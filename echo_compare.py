"""
echo_compare.py
---------------
스피커를 어디에 꽂느냐(지금 USB 스피커 / ReSpeaker 3.5mm 잭)에 따라
인형 목소리가 마이크로 얼마나 되돌아오는지 비교한다.

인형 목소리를 틀면서 동시에 ReSpeaker 로 녹음하고, 30ms 프레임마다
  - 소리 크기(RMS)
  - webrtcvad 판정 (지금 mic_vad.py 방식)
  - 칩 판정 SPEECHDETECTED (USB 권한이 있을 때만)
을 모아서 요약한다. 녹음한 소리는 logs/echo_<speaker>_<mode>.wav 로 남겨 직접 들어 볼 수 있다.

사용 (main.py / main_fish.py 는 꺼 두기 — 마이크를 동시에 못 연다):
    python echo_compare.py usb               # 지금 USB 스피커
    python echo_compare.py respeaker         # ReSpeaker 잭에 꽂은 AUX 스피커
    python echo_compare.py usb --talk        # 틀어 놓는 동안 사람이 말한다(끼어들기)

조용히 두고 재는 기본 모드: 녹음에 남는 소리가 작고, 말로 판정한 비율이 낮을수록 좋다.
--talk 모드: 사람이 말한 구간을 얼마나 잘 잡는지, 녹음 wav 에서 사람 목소리가 또렷한지 본다.
"""

import argparse
import os
import struct
import subprocess
import threading
import time
import wave

import numpy as np
import pyaudio
import webrtcvad

from mic_vad import first_channel, input_channels

SPEAKERS = {
    "usb": "plughw:CARD=UACDemoV10",
    "respeaker": "plughw:CARD=ArrayUAC10",
}
RATE = 16000
FRAME_MS = 30
CHUNK = RATE * FRAME_MS // 1000
LEAD_S = 1.5      # 틀기 전 조용한 구간(바닥 소음 측정)
ENERGY_THR = 500.0  # mic_vad.LiveRecorder 기본값과 같게

# 인형 목소리 대신 쓸 소리: 짧은 안내 음성을 이어 붙여 반복한다
CUE_DIR = "audio"
CUE_SKIP = ("cue_tap", "cue_poke")


class Chip:
    """ReSpeaker(XVF-3000) 칩에 USB 제어 전송으로 SPEECHDETECTED 를 묻는다. 못 쓰면 None."""

    def __init__(self):
        self.dev = None
        try:
            import usb.core
            import usb.util
            self._usb = usb.util
            dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if dev is not None:
                self.dev = dev
                self.read(19, 22)   # 권한 확인
        except Exception as e:
            print(f"⚠️  칩 판정 못 씀 ({e}) — webrtcvad 만 비교합니다")
            self.dev = None

    def read(self, param_id, offset):
        u = self._usb
        resp = self.dev.ctrl_transfer(
            u.CTRL_IN | u.CTRL_TYPE_VENDOR | u.CTRL_RECIPIENT_DEVICE,
            0, 0x80 | 0x40 | offset, param_id, 8, 100)
        return struct.unpack("ii", resp.tobytes())[0]

    def speech(self):
        if self.dev is None:
            return None
        try:
            return bool(self.read(19, 22))
        except Exception:
            return None


def build_voice(path, seconds=10.0):
    """안내 음성들을 16kHz 로 이어 붙여 seconds 길이쯤 되는 wav 를 만든다."""
    parts = []
    for name in sorted(os.listdir(CUE_DIR)):
        if not name.endswith(".wav") or name.startswith(CUE_SKIP):
            continue
        with wave.open(os.path.join(CUE_DIR, name)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            src = w.getframerate()
        t = np.arange(int(len(x) * RATE / src)) * src / RATE
        parts.append(np.interp(t, np.arange(len(x)), x).astype(np.int16))
        parts.append(np.zeros(RATE // 4, dtype=np.int16))
    one = np.concatenate(parts)
    reps = max(1, int(np.ceil(seconds * RATE / len(one))))
    voice = np.tile(one, reps)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(voice.tobytes())
    return len(voice) / RATE


def find_mic(pa):
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if "ReSpeaker" in info["name"] and info["maxInputChannels"] > 0:
            return i
    raise SystemExit("ReSpeaker 마이크를 못 찾았습니다")


def summarize(label, rows):
    if not rows:
        return
    rms = np.array([r[0] for r in rows])
    web = np.array([r[1] for r in rows])
    chip = [r[2] for r in rows if r[2] is not None]
    line = (f"  {label:<10} 평균 RMS {rms.mean():7.0f} | "
            f"webrtcvad 말 판정 {100 * web.mean():5.1f}%")
    if chip:
        line += f" | 칩 말 판정 {100 * np.mean(chip):5.1f}%"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("speaker", choices=SPEAKERS)
    ap.add_argument("--talk", action="store_true", help="틀어 놓는 동안 사람이 말한다")
    ap.add_argument("--wav", help="인형 목소리 대신 틀 wav (기본: audio/ 안내 음성 이어 붙이기)")
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)
    voice = args.wav or "logs/echo_voice.wav"
    dur = build_voice(voice, args.seconds) if not args.wav else None
    if dur is None:
        with wave.open(voice) as w:
            dur = w.getnframes() / w.getframerate()

    vad = webrtcvad.Vad(2)
    chip = Chip()
    pa = pyaudio.PyAudio()
    mic = find_mic(pa)
    channels = input_channels(pa, mic)
    stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=RATE, input=True,
                     frames_per_buffer=CHUNK, input_device_index=mic)

    playing = threading.Event()
    done = threading.Event()

    def play():
        time.sleep(LEAD_S)
        playing.set()
        subprocess.run(["aplay", "-q", "-D", SPEAKERS[args.speaker], voice])
        playing.clear()
        time.sleep(1.0)
        done.set()

    if args.talk:
        print("🗣️  소리가 나오기 시작하면 평소처럼 말씀해 보세요 (예: '모리야, 오늘 날씨 어때?')")
    threading.Thread(target=play, daemon=True).start()

    quiet, during, after, frames = [], [], [], []
    started = False
    while not done.is_set():
        data = first_channel(stream.read(CHUNK, exception_on_overflow=False), channels)
        frames.append(data)
        x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(x ** 2)))
        web = vad.is_speech(data, RATE) and rms > ENERGY_THR
        row = (rms, web, chip.speech())
        if playing.is_set():
            started = True
            during.append(row)
        elif started:
            after.append(row)
        else:
            quiet.append(row)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    out = f"logs/echo_{args.speaker}_{'talk' if args.talk else 'quiet'}.wav"
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(frames))

    print(f"\n📊 스피커={args.speaker}  모드={'끼어들기' if args.talk else '조용히'}  "
          f"(재생 {dur:.1f}s)")
    summarize("틀기 전", quiet)
    summarize("트는 중", during)
    summarize("끝난 뒤", after)
    if quiet and during:
        base = np.mean([r[0] for r in quiet])
        echo = np.mean([r[0] for r in during])
        print(f"  → 트는 중 소리가 바닥 소음의 {echo / max(base, 1):.1f}배")
    print(f"💾 녹음: {out}")


if __name__ == "__main__":
    main()
