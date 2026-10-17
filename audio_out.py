"""
audio_out.py
------------
mp3 파일을 스피커로 재생. Pi(Linux)에서는 mpg123, Windows에서는 playsound.

Pi 설치: sudo apt install mpg123

스피커 장치 명시:
- ALSA default 가 마이크 카드를 가리키는 등 default 로 재생이 안 되는 경우
  SPEAKER_ALSA_DEVICE 에 USB 스피커 ALSA 이름을 지정.
- 예: "plughw:1,0"  (aplay -l 에서 본 card 번호 사용. plughw 가 자동 리샘플링)
- None 이면 시스템 default 로 재생.
"""
from __future__ import annotations

import math
import platform
import shutil
import queue
import struct
import wave
import threading
import subprocess
import time
import requests
from pathlib import Path

# 🔊 USB 스피커 ALSA 디바이스. None 이면 system default.
SPEAKER_ALSA_DEVICE = "plughw:CARD=UACDemoV10"

# 🔊 재생 볼륨(소프트웨어 증폭). 1.0=원음, 2.0=2배, 0.5=절반.
# 하드웨어 볼륨(amixer)을 먼저 올리고도 작으면 이 값을 키우세요(너무 크면 찢어짐).
SPEAKER_VOLUME = 1.0


def play_mp3(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    system = platform.system()

    if system == "Linux":
        if shutil.which("mpg123"):
            cmd = ["mpg123", "-q"]
            if SPEAKER_ALSA_DEVICE:
                cmd.extend(["-a", SPEAKER_ALSA_DEVICE])
            if SPEAKER_VOLUME != 1.0:                       # -f: 32768=원음
                cmd.extend(["-f", str(int(32768 * SPEAKER_VOLUME))])
            cmd.append(str(p))
            subprocess.run(cmd, check=True)
            return
        if shutil.which("ffplay"):
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            if SPEAKER_VOLUME != 1.0:
                cmd.extend(["-af", f"volume={SPEAKER_VOLUME}"])
            cmd.append(str(p))
            subprocess.run(cmd, check=True)
            return
        raise RuntimeError(
            "Linux: mpg123가 설치되어 있지 않습니다. `sudo apt install mpg123`"
        )

    if system == "Darwin":
        subprocess.run(["afplay", str(p)], check=True)
        return

    # Windows: playsound (pip install playsound==1.2.2 권장)
    try:
        from playsound import playsound
        playsound(str(p.resolve()))
    except ImportError as e:
        raise RuntimeError(
            "Windows: `pip install playsound==1.2.2` 가 필요합니다."
        ) from e


def _mp3_cmd(path: str, volume: float) -> list[str] | None:
    """mp3 를 한 번 재생하는 명령. 쓸 수 있는 재생기가 없으면 None."""
    system = platform.system()
    if system == "Linux":
        if shutil.which("mpg123"):
            cmd = ["mpg123", "-q"]
            if SPEAKER_ALSA_DEVICE:
                cmd.extend(["-a", SPEAKER_ALSA_DEVICE])
            cmd.extend(["-f", str(int(32768 * volume))])   # -f: 32768=원음
            return cmd + [path]
        if shutil.which("ffplay"):
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                    "-af", f"volume={volume}", path]
        return None
    if system == "Darwin":
        return ["afplay", "-v", str(volume), path]
    return None


def _wav_cmd(path: str) -> list[str] | None:
    """wav 를 한 번 재생하는 명령(aplay). 쓸 수 있는 재생기가 없으면 None."""
    if platform.system() == "Linux" and shutil.which("aplay"):
        cmd = ["aplay", "-q"]
        if SPEAKER_ALSA_DEVICE:
            cmd.extend(["-D", SPEAKER_ALSA_DEVICE])
        return cmd + [path]
    if platform.system() == "Darwin":
        return ["afplay", path]
    return None


class LoopPlayer:
    """wav(또는 mp3)를 멈출 때까지 되풀이해 튼다. 생각하는 동안 까는 음악에 쓴다.

    wav 는 aplay 로 튼다. mpg123 은 여는 순간 스피커가 바쁘면 장치가 풀린 뒤에도
    소리 없이 멈춰 버리는데, aplay 는 바로 실패해서 잠깐 뒤 다시 열 수 있다.
    wav 는 volume 을 따르지 않는다 — 파일 음량을 맞춰 두고 쓴다.

    stop() 은 재생기가 완전히 꺼질 때까지 기다린다. 스피커(ALSA hw 장치)는
    한 번에 하나만 열 수 있어서, 덜 꺼진 채로 TTS 가 aplay 를 열면 부딪친다.
    """

    RETRY_SEC = 0.2      # 스피커가 바쁠 때 다시 열어 보는 간격

    def __init__(self, path: str, volume: float = 1.0):
        self._path = str(path)
        self._volume = volume * SPEAKER_VOLUME
        self._stop = threading.Event()
        # 재생기를 띄우는 순간과 끄는 순간이 엇갈려 하나가 남지 않게 한다.
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._path.lower().endswith(".wav"):
            cmd = _wav_cmd(self._path)
        else:
            cmd = _mp3_cmd(self._path, self._volume)
        if not cmd or not Path(self._path).exists():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(cmd,), daemon=True)
        self._thread.start()

    def _run(self, cmd: list[str]) -> None:
        warned = False
        while True:
            with self._lock:
                if self._stop.is_set():
                    return
                try:
                    self._proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                except OSError as e:
                    print(f"⚠️  로딩 음악 재생 실패: {e}")
                    return
            _, err = self._proc.communicate()
            # 멈추라고 해서 꺼진 게 아니면, 스피커가 잠깐 바빴을 수 있으니 다시 연다.
            # 왜 못 틀었는지는 한 번만 남긴다.
            if self._proc.returncode != 0 and not self._stop.is_set():
                if not warned:
                    msg = (err or b"").decode(errors="replace").strip().splitlines()
                    print(f"⚠️  로딩 음악 재생 실패, 다시 시도(code {self._proc.returncode}): "
                          f"{msg[-1] if msg else ''}")
                    warned = True
                self._stop.wait(self.RETRY_SEC)

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None


def play_wav(path: str) -> None:
    """wav 파일을 스피커로 재생. Pi(Linux)에서는 aplay(ALSA), mac에서는 afplay.

    CosyVoice2 TTS 서버는 wav를 반환하므로, mp3용 mpg123이 아니라 wav용 aplay를 쓴다.
    SPEAKER_ALSA_DEVICE(예: "plughw:1,0")로 USB 스피커를 직접 지정.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    system = platform.system()

    if system == "Linux":
        if shutil.which("aplay"):
            cmd = ["aplay", "-q"]
            if SPEAKER_ALSA_DEVICE:
                cmd.extend(["-D", SPEAKER_ALSA_DEVICE])
            cmd.append(str(p))
            subprocess.run(cmd, check=True)
            return
        if shutil.which("ffplay"):
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            if SPEAKER_VOLUME != 1.0:
                cmd.extend(["-af", f"volume={SPEAKER_VOLUME}"])
            cmd.append(str(p))
            subprocess.run(cmd, check=True)
            return
        raise RuntimeError(
            "Linux: aplay가 없습니다. `sudo apt install alsa-utils`"
        )

    if system == "Darwin":
        subprocess.run(["afplay", str(p)], check=True)
        return

    # Windows: playsound (pip install playsound==1.2.2 권장)
    try:
        from playsound import playsound
        playsound(str(p.resolve()))
    except ImportError as e:
        raise RuntimeError(
            "Windows: `pip install playsound==1.2.2` 가 필요합니다."
        ) from e


# 효과음. 소리 하나에 뜻 하나만 둔다 — 같은 소리를 여러 곳에 쓰면 헷갈리신다.
# (음표, 울림) — 음표는 (시작 초, 주파수), 울림은 소리가 잦아드는 빠르기(초).
CHIMES = {
    "wake":    (((0.00, 1046.5),), 0.16),                          # 띵 — 불렀을 때
    "bye":     (((0.00, 1046.5), (0.11, 784.0)), 0.16),            # 또롱↓ — 이제 쉴게요
    "message": (((0.00, 659.3), (0.30, 523.3)), 0.30),             # 딩동 — 가족 메시지
    "sent":    (((0.00, 523.3), (0.09, 659.3), (0.18, 784.0), (0.27, 1046.5)), 0.16),  # 띠리링 — 보냈어요
    # 약 시간. 늘 같은 소리가 '약 드실 때' 라는 습관 신호가 되게, 다른 소리보다
    # 느리고 길게 둬서 확실히 구분되게 한다.
    "medicine": (((0.00, 1046.5), (0.22, 784.0), (0.44, 1046.5), (0.66, 1318.5)), 0.22),
    "tap":     (((0.00, 1318.5),), 0.035),                         # 톡 — 화면을 눌렀어요
    "poke":    (((0.00, 880.0), (0.06, 1318.5)), 0.06),            # 뿅 — 얼굴을 건드렸어요
    "pet":     (((0.00, 784.0), (0.20, 659.3), (0.40, 784.0)), 0.30),  # 랄라 — 쓰다듬어 줬어요
}


def make_chime_wav(path: str, notes, decay: float = 0.16,
                   sample_rate: int = 24000) -> str:
    """'또롱' 하고 올라가는 짧은 효과음 wav 를 만든다(0.5초 남짓).

    말씀을 다 들었다는 신호로 쓴다. 사람 목소리 맞장구는 매번 같은 말이면
    금방 기계 같고 실제 답과 이어 들으면 어색해서, 말이 아닌 소리로 둔다.
    파일을 저장소에 두지 않고 처음 한 번 만들어 쓴다. 이미 있으면 그대로 둔다.

    음을 살짝 겹쳐 올린다. 올라가는 음은 '알았어요' 처럼 들린다.
    notes·decay 를 바꾸면 다른 신호음이 된다(CHIMES 참고). 어르신 귀에 날카롭지 않게 배음을 거의 넣지 않고 소리도 작게 둔다.
    """
    p = Path(path)
    if p.exists():
        return str(p)

    length = notes[-1][0] + max(0.12, decay * 2.75)
    n = int(sample_rate * length)
    samples = [0.0] * n
    for start, freq in notes:
        s0 = int(start * sample_rate)
        for i in range(n - s0):
            t = i / sample_rate
            attack = min(1.0, t / 0.008)          # 딸깍 소리가 안 나게 살짝 연다
            env = attack * math.exp(-t / decay)
            w = math.sin(2 * math.pi * freq * t) + 0.12 * math.sin(4 * math.pi * freq * t)
            samples[s0 + i] += 0.2 * env * w

    fade = int(0.02 * sample_rate)                # 끝도 딸깍 안 나게 닫는다
    for i in range(fade):
        samples[n - 1 - i] *= i / fade

    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples
        ))
    return str(p)


def synthesize_and_play_stream(text, url, key, sample_rate=24000, buffer_bytes=96000,
                               on_start=None, spk_id=None):
    """PCM을 백그라운드 스레드로 미리 받아 쌓아두고 재생. 네트워크 흔들림을 흡수한다.

    spk_id: 어떤 화자(목소리)로 합성할지. None 이면 서버 기본 목소리(caregiver).
            기본 목소리로 등록한 가족 음성의 speaker_id 를 넘기면 그 목소리로 말한다.
    """
    def open_stream():
        return requests.post(url, headers={"x-api-key": key},
                             json={"text": text, "spk_id": spk_id},
                             stream=True, timeout=60)

    return _stream_and_play(open_stream, sample_rate, buffer_bytes, on_start, "TTS")


def synthesize_and_play_fish(text, key, reference_id=None, model=None,
                             sample_rate=24000, buffer_bytes=24000,
                             latency="balanced", on_start=None):
    """Fish Audio 클라우드 TTS(https://api.fish.audio/v1/tts)로 스트리밍 재생.

    reference_id: Fish Audio 에서 고른/만든 목소리 모델 ID. None 이면 Fish 기본 목소리.
    model       : 요청 헤더 'model'(예: "s1", "s2.1-pro"). None 이면 Fish 기본 모델.
    latency     : "low" | "balanced" | "normal". 낮을수록 첫소리가 빠르고 품질은 조금 떨어진다.
    PCM(16bit mono)으로 받아서 CosyVoice 와 같은 aplay 파이프로 재생한다.
    """
    headers = {"Authorization": f"Bearer {key}"}
    if model:
        headers["model"] = model
    body = {"text": text, "format": "pcm", "sample_rate": sample_rate, "latency": latency}
    if reference_id:
        body["reference_id"] = reference_id

    def open_stream():
        return requests.post("https://api.fish.audio/v1/tts", headers=headers,
                             json=body, stream=True, timeout=60)

    return _stream_and_play(open_stream, sample_rate, buffer_bytes, on_start, "TTS(Fish)")


def _stream_and_play(open_stream, sample_rate, buffer_bytes, on_start, label):
    """open_stream() 이 돌려준 PCM 스트리밍 응답을 백그라운드로 받으며 aplay 로 재생."""
    t0 = time.perf_counter()
    q = queue.Queue()
    box = {"ttfb": None, "dl_done": None, "err": None, "bytes": 0}

    def downloader():
        try:
            with open_stream() as r:
                if not r.ok:
                    # 인증/결제 오류 등은 본문에 이유가 담겨 온다.
                    raise RuntimeError(f"{label} HTTP {r.status_code}: {r.text[:200]}")
                for chunk in r.iter_content(chunk_size=4096):
                    if box["ttfb"] is None:
                        box["ttfb"] = time.perf_counter()
                    box["bytes"] += len(chunk)
                    q.put(chunk)
        except Exception as e:
            box["err"] = e
        finally:
            box["dl_done"] = time.perf_counter()
            q.put(None)

    threading.Thread(target=downloader, daemon=True).start()

    proc = subprocess.Popen(
        ["aplay", "-q", "-f", "S16_LE", "-r", str(sample_rate), "-c", "1",
         "-D", SPEAKER_ALSA_DEVICE, "-"],
        stdin=subprocess.PIPE
    )

    buf = b""
    started = False
    t_first = None
    while True:
        chunk = q.get()
        if chunk is None:
            break
        if not started:
            buf += chunk
            if len(buf) >= buffer_bytes:
                proc.stdin.write(buf)
                buf = b""
                started = True
                t_first = time.perf_counter()
                if on_start:
                    on_start()
        else:
            proc.stdin.write(chunk)

    if not started and buf:
        proc.stdin.write(buf)
        t_first = time.perf_counter()
        if on_start:
            on_start()

    proc.stdin.close()
    proc.wait()
    t_end = time.perf_counter()

    if box["err"]:
        raise box["err"]

    audio_sec = box["bytes"] / (sample_rate * 2)
    print(
        f"   ⏱  {label}  서버첫청크 {(box['ttfb'] - t0):.2f}s"
        f" | 첫소리 {(t_first - t0):.2f}s"
        f" | 다운로드완료 {(box['dl_done'] - t0):.2f}s"
        f" | 재생완료 {(t_end - t0):.2f}s"
        f" | 오디오길이 {audio_sec:.2f}s"
    )
    return {
        "first_sound": t_first - t0,
        "download_done": box["dl_done"] - t0,
        "total": t_end - t0,
        "audio_sec": audio_sec,
    }


if __name__ == "__main__":
    import sys
    play_mp3(sys.argv[1] if len(sys.argv) > 1 else "response.mp3")
