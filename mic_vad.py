"""
mic_vad.py
----------
Pi 마이크에서 라이브로 녹음하면서 webrtcvad로 발화 시작/종료를 감지.
stt4vad.py 의 VAD 알고리즘을 스트리밍 버전으로 이식.

사용:
    from mic_vad import LiveRecorder
    rec = LiveRecorder()
    wav_path = rec.record_until_silence("input.wav")

말끝 감지가 잘 안 될 때(계속 녹음됨):
    VAD_DEBUG=1 python main.py 로 실행하면 프레임마다 에너지(e)·VAD 값이 찍힌다.
    조용히 있을 때 찍히는 e 값보다 살짝 높게 min_energy_threshold 를 잡으면 된다.
"""

import os
import queue
import struct
import threading
import wave
from collections import deque
from difflib import SequenceMatcher

import numpy as np
import pyaudio
import webrtcvad

# 켜면 프레임마다 에너지/VAD 판정을 출력한다(문턱값 보정용).
VAD_DEBUG = bool(os.getenv("VAD_DEBUG"))


def input_channels(pa: pyaudio.PyAudio, index: int | None) -> int:
    """마이크를 열 채널 수. ReSpeaker 6채널 펌웨어는 6으로 열어야 하고, 그중 0번
    (칩이 빔포밍·잡음 제거를 끝낸 소리)만 쓴다. 1~4번은 마이크 원음, 5번은 스피커로
    나간 소리다. 다른 마이크는 1채널로 연다."""
    info = (pa.get_device_info_by_index(index) if index is not None
            else pa.get_default_input_device_info())
    if "ReSpeaker" in info["name"] and info["maxInputChannels"] >= 6:
        return 6
    return 1


def first_channel(data: bytes, channels: int) -> bytes:
    """여러 채널이 번갈아 담긴 16bit 소리에서 0번 채널만 꺼낸다."""
    if channels == 1:
        return data
    return np.frombuffer(data, dtype=np.int16)[::channels].tobytes()


class ChipVAD:
    """ReSpeaker(XVF-3000) 칩에 USB 제어 전송으로 발화 여부를 묻는다.

    칩은 빔포밍·에코 제거를 거친 소리로 판정해서, 스피커가 ReSpeaker 잭에 꽂혀 있으면
    인형 목소리를 말로 잘못 잡는 일이 적다. 장치가 없거나 읽기에 실패하면 None 을
    돌려주고, 그때는 webrtcvad 로 판정한다. MIC_VAD=webrtc 면 칩을 아예 안 쓴다.
    일반 사용자로 읽으려면 udev 규칙이 있어야 한다(/etc/udev/rules.d/60-respeaker.rules).

    한 번 묻는 데 16ms 쯤 걸려서 녹음 루프(30ms)에서 바로 물으면 소리가 밀린다.
    start() 하면 다른 스레드가 계속 묻고, speech_detected() 는 마지막 값만 돌려준다.
    """

    VID, PID = 0x2886, 0x0018
    SPEECHDETECTED = (19, 22)   # (파라미터 번호, 오프셋) — Seeed tuning.py 표
    DOAANGLE = (21, 0)          # 소리가 들어온 방향(0~359도)
    TIMEOUT_MS = 50

    def __init__(self):
        self._dev = None
        self._latest: bool | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if os.getenv("MIC_VAD", "chip") != "chip":
            return
        try:
            import usb.core
            import usb.util
            self._util = usb.util
            self._dev = usb.core.find(idVendor=self.VID, idProduct=self.PID)
            if self._dev is not None:
                self._read(*self.SPEECHDETECTED)
        except Exception as e:
            print(f"⚠️  칩 VAD 못 씀({e}) — webrtcvad 로 판정")
            self._dev = None

    def _read(self, param_id: int, offset: int) -> int:
        u = self._util
        resp = self._dev.ctrl_transfer(
            u.CTRL_IN | u.CTRL_TYPE_VENDOR | u.CTRL_RECIPIENT_DEVICE,
            0, 0x80 | 0x40 | offset, param_id, 8, self.TIMEOUT_MS)  # 0x80 읽기, 0x40 정수
        return struct.unpack("ii", resp.tobytes())[0]

    def start(self) -> None:
        if self._dev is None or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        self._latest = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                self._latest = bool(self._read(*self.SPEECHDETECTED))
            except Exception as e:
                print(f"⚠️  칩 VAD 읽기 실패({e}) — webrtcvad 로 판정")
                self._dev = None
                self._latest = None
                return

    @property
    def available(self) -> bool:
        return self._dev is not None

    def speech_detected(self) -> bool | None:
        """마지막으로 읽은 칩 판정. start() 전이거나 칩을 못 쓰면 None."""
        return self._latest

    def doa_angle(self) -> int | None:
        if self._dev is None:
            return None
        try:
            return self._read(*self.DOAANGLE)
        except Exception:
            return None


class MicTap:
    """인형이 말하는 동안 마이크를 미리 열어 두고, 끝나면 그대로 녹음기에 넘긴다.

    말하는 동안에는 최근 tail_ms 만 남긴다(인형 목소리도 섞여 있다). end() 를 부르면
    그때까지의 꼬리를 tail 로 굳히고, 그 뒤 소리는 read() 로 하나도 빠짐없이 넘긴다.
    대답이 인형의 말끝과 겹쳐도 앞부분을 놓치지 않으려는 것이다.

    barge 를 주면 끼어들기를 듣는다. 인형이 말하는 동안 칩 VAD 로 듣다가 말씀이
    BARGE_IN_MS 쯤 이어지면 barge 를 켠다 — 부르는 쪽은 이걸 TTS 의 stop_event 에 묶어
    인형 말을 멈춘다. 꼬리(tail_ms)가 말씀의 시작을 담고 있어서 앞부분도 안 잘린다.
    인형 목소리를 말로 잡지 않으려면 칩이 그 소리를 알아야 한다(스피커가 ReSpeaker
    잭에 꽂혀 있어야 에코가 지워진다). 칩을 못 쓰면 끼어들기는 꺼진다.
    """

    # 이만큼(ms) 동안 대부분(BARGE_IN_RATIO)이 말이면 끼어드신 것으로 본다. 기침이나
    # 짧은 맞장구("응")에는 멈추지 않을 만큼 길게 잡았다. 스피커로 바꾸면 다시 맞춘다.
    BARGE_IN_MS = int(os.getenv("BARGE_IN_MS", "400"))
    BARGE_IN_RATIO = 0.8
    # 모리가 말을 시작한 직후엔 칩이 아직 그 목소리를 지우지 못해(에코 제거가 적응하는
    # 중) 모리 목소리를 말로 잡을 때가 있었다. 마이크를 연 뒤 이만큼은 끼어들기를 안 본다.
    BARGE_IN_GRACE_MS = int(os.getenv("BARGE_IN_GRACE_MS", "1000"))

    def __init__(self, rate: int = 16000, chunk_duration_ms: int = 30,
                 tail_ms: int = 1000, input_device_index: int | None = None,
                 barge: threading.Event | None = None):
        self.rate = rate
        self.chunk_size = int(rate * chunk_duration_ms / 1000)
        self.tail: list[bytes] = []
        self._tail = deque(maxlen=max(1, tail_ms // chunk_duration_ms))
        self._live: "queue.Queue[bytes]" = queue.Queue()
        self._ended = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._barge = barge
        self._chip = None
        self._window: deque = deque(maxlen=max(1, self.BARGE_IN_MS // chunk_duration_ms))
        self._grace_frames = self.BARGE_IN_GRACE_MS // chunk_duration_ms
        self._after_barge = 0        # 끼어드신 걸 알아챈 뒤 지난 프레임 수
        self._preroll = 200 // chunk_duration_ms
        if barge is not None:
            self._chip = ChipVAD()
            if not self._chip.available:
                self._chip = None
                print("⚠️  칩 VAD 를 못 써서 끼어들기는 끕니다")
            else:
                self._chip.start()
        self._pa = pyaudio.PyAudio()
        try:
            self._channels = input_channels(self._pa, input_device_index)
            self._stream = self._pa.open(
                format=pyaudio.paInt16, channels=self._channels, rate=rate, input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=input_device_index,
            )
        except Exception:
            self._pa.terminate()
            raise
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
            except Exception:
                break
            data = first_channel(data, self._channels)
            with self._lock:
                if self._ended:
                    self._live.put(data)
                else:
                    self._tail.append(data)
                    self._check_barge(data)

    def _check_barge(self, data: bytes) -> None:
        if self._barge is not None and self._barge.is_set():
            self._after_barge += 1
            return
        if self._chip is None:
            return
        if self._grace_frames > 0:
            self._grace_frames -= 1
            return
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        loud = float(np.sqrt(np.mean(x ** 2))) > LiveRecorder.CHIP_ENERGY_THRESHOLD
        self._window.append(bool(self._chip.speech_detected()) and loud)
        if (len(self._window) == self._window.maxlen
                and sum(self._window) >= self.BARGE_IN_RATIO * len(self._window)):
            print("✋ 끼어드심 — 모리 말을 멈추고 듣습니다")
            self._barge.set()

    @property
    def barged(self) -> bool:
        """모리 말 중에 끼어드셨나."""
        return self._barge is not None and self._barge.is_set()

    def _stop_chip(self) -> None:
        if self._chip is not None:
            self._chip.stop()
            self._chip = None

    def end(self) -> None:
        """인형 말이 끝났다. 지금까지의 꼬리를 굳히고 이후 소리를 모은다."""
        with self._lock:
            self._ended = True
            self.tail = list(self._tail)
            if self.barged:
                # 끼어드셨으면 말씀이 시작된 데부터만 남긴다(알아챈 시점에서 판정 창
                # 만큼, 거기서 조금 더 앞). 그 앞은 모리 목소리뿐이라 STT 를 헷갈리게 한다.
                keep = self._after_barge + self._window.maxlen + self._preroll
                self.tail = self.tail[-keep:]
        # 녹음기가 칩에 따로 묻는다. 둘이 함께 묻지 않게 여기선 그만둔다.
        self._stop_chip()

    def read(self, _n: int = 0) -> bytes:
        try:
            return self._live.get(timeout=2.0)
        except queue.Empty:
            raise OSError("마이크에서 소리가 안 들어옴")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._stop_chip()
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()


def strip_echo_prefix(text: str, spoken: str, min_chars: int = 3,
                      ratio: float = 0.75, anywhere: bool = False) -> str:
    """받아 적은 글이 인형이 막 한 말의 끝으로 시작하면 그 부분을 지운다.

    MicTap 의 꼬리에는 인형 목소리가 섞여 들어가서 "…드셨어요 네 먹었어요" 처럼
    나온다. 지우고 나서 남는 게 없으면 원래 글을 그대로 둔다 — 인형 말끝과
    같은 말로 대답하신 것("좋아요?" → "좋아요")일 수 있다.

    anywhere 면 인형 말의 끝이 아니라 어느 부분과 겹쳐도 지운다. 끼어드셨을 때는
    인형이 말하던 중간이 섞여 들어가기 때문이다.
    """
    kept = [(i, c) for i, c in enumerate(text) if c.isalnum()]
    head = "".join(c for _, c in kept)
    tail = "".join(c for c in spoken if c.isalnum())
    # 가장 잘 맞는 길이를 고른다(같으면 긴 쪽). 기준만 넘는 가장 긴 길이를 고르면
    # 대답 첫 글자까지 같이 지워진다("드셨어요 네" → "먹었어요").
    best = None
    for k in range(min_chars, min(len(head), len(tail), 40) + 1):
        parts = ([tail[i:i + k] for i in range(len(tail) - k + 1)] if anywhere
                 else [tail[-k:]])
        r = max(SequenceMatcher(None, p, head[:k]).ratio() for p in parts)
        if r >= ratio and (best is None or (r, k) >= best):
            best = (r, k)
    if best is None:
        return text
    rest = text[kept[best[1] - 1][0] + 1:].lstrip(" ,.?!~…")
    return rest if rest else text


class LiveRecorder:
    def __init__(
        self,
        rate: int = 16000,
        chunk_duration_ms: int = 30,
        vad_aggressiveness: int = 2,
        silence_end_ms: int = 1500,
        max_record_seconds: int = 10,
        min_energy_threshold: float = 500.0,
        input_device_index: int | None = None,
    ):
        self.rate = rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(rate * chunk_duration_ms / 1000)  # samples per frame
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.chip = ChipVAD()
        # 발화 종료 판단: 말끝 뒤로 이만큼 '연속 침묵'이 이어지면 종료.
        self.silence_end_ms = silence_end_ms
        self.max_record_seconds = max_record_seconds
        self.min_energy_threshold = min_energy_threshold
        self.input_device_index = input_device_index

    def record_until_silence(
        self,
        output_path: str = "input.wav",
        on_speech_start=None,
        on_speech_end=None,
        max_wait_seconds: float | None = None,
        on_level=None,
        on_pause=None,
        tap: "MicTap | None" = None,
        overlap_window_ms: int = 600,
        stop_event=None,
    ) -> str | None:
        """
        마이크에서 듣다가 발화 시작 감지 → 말끝 뒤 silence_end_ms 만큼 침묵 → wav 저장.
        반환: 저장된 wav 경로. 발화 미감지 시 None.

        on_speech_start() : 발화 시작이 감지된 '그 순간' 1회 호출 (감정 샘플링 시작용).
        on_speech_end()   : 발화 종료/녹음 종료 시 1회 호출 (감정 샘플링 종료용).
            두 콜백은 녹음 루프와 동시에 돌아가는 다른 작업(감정 인식)을 걸어두는 용도.
        on_level(v)       : 목소리 크기(0~1)를 약 0.1초마다 알린다. 듣는 중 화면 표시용.
        on_pause()        : 말씀 중간에 짧게 쉬실 때(PAUSE_MS) 알린다. 끄덕이는 맞장구용.
            모두 녹음 루프 안에서 불리니 바로 돌아와야 한다(오래 걸리면 소리가 끊긴다).
        tap               : 인형이 말하는 동안 미리 열어 둔 마이크(MicTap). 주면 그걸 이어
            쓰고, 끝나면 닫는다. 인형 말이 끝나고 overlap_window_ms 안에 말이 시작되면
            (= 말끝과 겹쳐 대답하셨으면) 그 꼬리를 녹음 앞에 붙인다. 붙였으면
            self.used_overlap 이 True — 받아 적은 글에서 인형 말을 지워야 한다.
        stop_event        : 켜지면 듣다가도 바로 그만두고 None 을 돌려준다.
        """
        self.used_overlap = False
        if tap is not None:
            pa = stream = None
            read = tap.read
        else:
            pa = pyaudio.PyAudio()
            channels = input_channels(pa, self.input_device_index)
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.input_device_index,
            )

            def read(n):
                return first_channel(stream.read(n, exception_on_overflow=False), channels)
        overlap_frames = overlap_window_ms // self.chunk_duration_ms

        start_win: deque = deque(maxlen=10)   # 발화 시작 판단용(최근 0.3초)
        pre_buffer: deque = deque(maxlen=10)  # 말 앞부분 잘림 방지

        frames: list[bytes] = []
        recording_started = False
        total_recorded_frames = 0
        max_frame_count = int(self.max_record_seconds * 1000 / self.chunk_duration_ms)

        waited_frames = 0
        max_wait_frames = (
            int(max_wait_seconds * 1000 / self.chunk_duration_ms)
            if max_wait_seconds is not None
            else None
        )

        # 말끝 침묵 카운터. 연속 침묵 프레임이 이만큼 쌓이면 종료.
        silence_end_frames = int(self.silence_end_ms / self.chunk_duration_ms)
        silence_run = 0   # 연속 침묵 프레임 수
        speech_run = 0    # 연속 음성 프레임 수 (단발 노이즈로 침묵 카운터가 리셋되는 것 방지)

        # 맞장구 타이밍. 말씀을 어느 정도(PAUSE_MIN_SPEECH_MS) 하신 뒤 PAUSE_MS 쉬시면
        # 한 번 알린다. 말끝(silence_end_ms)보다 한참 짧아서 '듣는 중' 에 끄덕일 수 있다.
        pause_frames = int(self.PAUSE_MS / self.chunk_duration_ms)
        pause_min_speech = int(self.PAUSE_MIN_SPEECH_MS / self.chunk_duration_ms)
        spoken_since_pause = 0
        frame_no = 0

        print("🎙️  마이크 대기 중... (말하면 녹음 시작)")
        self.chip.start()

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    print("⏹️  듣기 중단")
                    frames.clear()
                    break
                data = read(self.chunk_size)
                chunk = np.frombuffer(data, dtype=np.int16)

                # RMS 에너지
                audio_float = chunk.astype(np.float32)
                energy = float(np.sqrt(np.mean(audio_float ** 2)))
                if np.isnan(energy):
                    energy = 0.0

                # 칩 판정을 먼저 쓰고, 못 읽으면 webrtcvad 로 판정한다.
                is_speech = self.chip.speech_detected()
                vad_src = "chip"
                if is_speech is None:
                    is_speech = self.vad.is_speech(data, self.rate)
                    vad_src = "web"
                thr = (self.CHIP_ENERGY_THRESHOLD if vad_src == "chip"
                       else self.min_energy_threshold)
                valid_speech = is_speech and energy > thr

                frame_no += 1
                if on_level is not None and frame_no % 3 == 0:
                    _safe_call(on_level, self._level(energy) if valid_speech else 0.0)

                if VAD_DEBUG:
                    print(f"e={energy:6.0f}  vad[{vad_src}]={int(is_speech)}  valid={int(valid_speech)}"
                          f"  sil={silence_run}/{silence_end_frames}")

                if not recording_started:
                    waited_frames += 1

                    if (
                        max_wait_frames is not None
                        and waited_frames >= max_wait_frames
                    ):
                        print("💤 대화 대기시간 초과")
                        break

                    pre_buffer.append(data)
                    start_win.append(1 if valid_speech else 0)
                    if len(start_win) == start_win.maxlen and sum(start_win) > 5:
                        print("▶️  말 시작 감지")
                        recording_started = True
                        if tap is not None and tap.tail and waited_frames <= overlap_frames:
                            frames.extend(tap.tail)
                            self.used_overlap = True
                            print("↩️  말끝과 겹친 대답 — 앞부분을 붙여 녹음")
                        frames.extend(pre_buffer)
                        pre_buffer.clear()
                        silence_run = 0
                        speech_run = 0
                        if on_speech_start is not None:
                            try:
                                on_speech_start()
                            except Exception as e:
                                print(f"⚠️  on_speech_start 콜백 오류: {e}")
                else:
                    frames.append(data)
                    total_recorded_frames += 1

                    # 말끝 침묵 누적. 단, '지속된' 음성(3프레임 이상)일 때만 침묵 카운터 리셋
                    # → 침묵 구간에 끼는 짧은 VAD 오탐(1~2프레임)으로 종료가 안 되던 문제 해결.
                    if valid_speech:
                        speech_run += 1
                        spoken_since_pause += 1
                        if speech_run >= 3:
                            silence_run = 0
                    else:
                        speech_run = 0
                        silence_run += 1
                        if (on_pause is not None and silence_run == pause_frames
                                and spoken_since_pause >= pause_min_speech):
                            spoken_since_pause = 0
                            _safe_call(on_pause)

                    if silence_run >= silence_end_frames:
                        print("⏹️  말 종료 감지")
                        # 말끝 뒤 침묵은 STT_TAIL_MS 만 남긴다. 다 보내면 STT 가 0.2초쯤 더 걸린다.
                        drop = silence_run - self.STT_TAIL_MS // self.chunk_duration_ms
                        if drop > 0:
                            del frames[-drop:]
                        break

                    if total_recorded_frames >= max_frame_count:
                        print("⏹️  최대 녹음 길이 도달")
                        break
        finally:
            self.chip.stop()
            if on_level is not None:
                _safe_call(on_level, 0.0)
            if tap is not None:
                tap.close()
            else:
                stream.stop_stream()
                stream.close()
                pa.terminate()
            # 발화가 한 번이라도 시작됐다면 종료 콜백 보장(정상 종료/최대길이/예외 모두)
            if recording_started and on_speech_end is not None:
                try:
                    on_speech_end()
                except Exception as e:
                    print(f"⚠️  on_speech_end 콜백 오류: {e}")

        if not frames:
            print("⚠️  감지된 음성이 없습니다.")
            return None

        self._save_wav(output_path, frames)
        return output_path

    # 녹음 파일에 남길 말끝 뒤 침묵(ms)
    STT_TAIL_MS = 300

    # 칩이 말이라고 할 때 함께 보는 소리 크기 문턱. webrtcvad 는 주변 소리를 자주 말로
    # 잡아서 500 을 넘어야 인정했는데, 칩은 그보다 덜 속아서 낮춘다. 칩이 처리한 0번
    # 채널에서 배경은 40~200(TV), 말 첫소리는 250~750, 말하는 중엔 1500~2000 이었다.
    # 작게 말씀하시거나 말끝을 흐리셔도 말로 잡고, 먼 TV 소리는 거른다.
    CHIP_ENERGY_THRESHOLD = float(os.getenv("MIC_CHIP_ENERGY_THR", "300"))

    # 끄덕일 쉼의 길이와, 그 전에 말씀하셔야 하는 길이(ms)
    PAUSE_MS = 300
    PAUSE_MIN_SPEECH_MS = 600

    def _level(self, energy: float) -> float:
        """RMS 에너지를 0~1 로. 문턱값 바로 위가 0, 그 7배쯤이 1(크게 말씀하실 때)."""
        thr = self.min_energy_threshold
        return max(0.0, min(1.0, (energy - thr) / (thr * 6)))

    def _save_wav(self, output_path: str, frames: list[bytes]) -> None:
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))
        print(f"💾  녹음 저장: {output_path}")


def _safe_call(fn, *args) -> None:
    """화면 표시용 콜백이 실패해도 녹음은 계속한다."""
    try:
        fn(*args)
    except Exception as e:
        print(f"⚠️  녹음 콜백 오류: {e}")


if __name__ == "__main__":
    rec = LiveRecorder()
    path = rec.record_until_silence("mic_test.wav")
    print(f"결과: {path}")
