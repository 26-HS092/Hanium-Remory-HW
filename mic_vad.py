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
import threading
import wave
from collections import deque
from difflib import SequenceMatcher

import numpy as np
import pyaudio
import webrtcvad

# 켜면 프레임마다 에너지/VAD 판정을 출력한다(문턱값 보정용).
VAD_DEBUG = bool(os.getenv("VAD_DEBUG"))


class MicTap:
    """인형이 말하는 동안 마이크를 미리 열어 두고, 끝나면 그대로 녹음기에 넘긴다.

    말하는 동안에는 최근 tail_ms 만 남긴다(인형 목소리도 섞여 있다). end() 를 부르면
    그때까지의 꼬리를 tail 로 굳히고, 그 뒤 소리는 read() 로 하나도 빠짐없이 넘긴다.
    대답이 인형의 말끝과 겹쳐도 앞부분을 놓치지 않으려는 것이다.
    """

    def __init__(self, rate: int = 16000, chunk_duration_ms: int = 30,
                 tail_ms: int = 1000, input_device_index: int | None = None):
        self.rate = rate
        self.chunk_size = int(rate * chunk_duration_ms / 1000)
        self.tail: list[bytes] = []
        self._tail = deque(maxlen=max(1, tail_ms // chunk_duration_ms))
        self._live: "queue.Queue[bytes]" = queue.Queue()
        self._ended = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pa = pyaudio.PyAudio()
        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16, channels=1, rate=rate, input=True,
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
            with self._lock:
                if self._ended:
                    self._live.put(data)
                else:
                    self._tail.append(data)

    def end(self) -> None:
        """인형 말이 끝났다. 지금까지의 꼬리를 굳히고 이후 소리를 모은다."""
        with self._lock:
            self._ended = True
            self.tail = list(self._tail)

    def read(self, _n: int = 0) -> bytes:
        try:
            return self._live.get(timeout=2.0)
        except queue.Empty:
            raise OSError("마이크에서 소리가 안 들어옴")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._pa.terminate()


def strip_echo_prefix(text: str, spoken: str, min_chars: int = 3,
                      ratio: float = 0.75) -> str:
    """받아 적은 글이 인형이 막 한 말의 끝으로 시작하면 그 부분을 지운다.

    MicTap 의 꼬리에는 인형 목소리가 섞여 들어가서 "…드셨어요 네 먹었어요" 처럼
    나온다. 지우고 나서 남는 게 없으면 원래 글을 그대로 둔다 — 인형 말끝과
    같은 말로 대답하신 것("좋아요?" → "좋아요")일 수 있다.
    """
    kept = [(i, c) for i, c in enumerate(text) if c.isalnum()]
    head = "".join(c for _, c in kept)
    tail = "".join(c for c in spoken if c.isalnum())
    # 가장 잘 맞는 길이를 고른다(같으면 긴 쪽). 기준만 넘는 가장 긴 길이를 고르면
    # 대답 첫 글자까지 같이 지워진다("드셨어요 네" → "먹었어요").
    best = None
    for k in range(min_chars, min(len(head), len(tail), 40) + 1):
        r = SequenceMatcher(None, tail[-k:], head[:k]).ratio()
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
        min_energy_threshold: float = 300.0,
        input_device_index: int | None = None,
    ):
        self.rate = rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(rate * chunk_duration_ms / 1000)  # samples per frame
        self.vad = webrtcvad.Vad(vad_aggressiveness)
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
        """
        self.used_overlap = False
        if tap is not None:
            pa = stream = None
            read = tap.read
        else:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.input_device_index,
            )

            def read(n):
                return stream.read(n, exception_on_overflow=False)
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

        try:
            while True:
                data = read(self.chunk_size)
                chunk = np.frombuffer(data, dtype=np.int16)

                # RMS 에너지
                audio_float = chunk.astype(np.float32)
                energy = float(np.sqrt(np.mean(audio_float ** 2)))
                if np.isnan(energy):
                    energy = 0.0

                is_speech = self.vad.is_speech(data, self.rate)
                valid_speech = is_speech and energy > self.min_energy_threshold

                frame_no += 1
                if on_level is not None and frame_no % 3 == 0:
                    _safe_call(on_level, self._level(energy) if valid_speech else 0.0)

                if VAD_DEBUG:
                    print(f"e={energy:6.0f}  vad={int(is_speech)}  valid={int(valid_speech)}"
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
                        break

                    if total_recorded_frames >= max_frame_count:
                        print("⏹️  최대 녹음 길이 도달")
                        break
        finally:
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
