"""
vad_live_test.py
----------------
사람이 직접 말하면서 칩 VAD 와 webrtcvad 를 비교한다.

한 번 말씀하신 소리를 두 방식이 '동시에' 판정한다. 같은 소리를 두고 각자 어디서
녹음을 시작하고 끝냈을지 따로 계산한 뒤(LiveRecorder 와 같은 규칙), 각자 잘라 낸
부분을 CLOVA 로 받아 적어 정답과 비교한다. 같은 말을 두 번 하실 필요가 없다.

사용 (main.py / main_fish.py 는 꺼 두기 — 마이크를 동시에 못 연다):
    systemctl --user stop mori
    python vad_live_test.py
    systemctl --user start mori

매 문장 전에 조건을 고른다. 음악이나 TV 를 켜고 끄면서 조건만 바꿔 입력하면 된다.
    Enter = 지난 조건 그대로 / q = 조용 / m = 음악 / t = TV / r = 다시 / s = 건너뛰기 / x = 끝내기
문장 속 "…" 는 그 자리에서 1~2초 쉬었다가 이어 말하라는 뜻이다(쉼 감지 시험).

자동 진행(키 입력 없이): python vad_live_test.py --auto
    조용 → 음악 → TV 순서로 문장 목록을 한 바퀴씩 돈다. 문장마다 스피커에서 '띵' 하면
    다음 문장을 읽는다. 조건이 바뀔 때는 '띵띵띵' 하고 AUTO_SWITCH_S 초 동안 기다린다
    (그동안 음악이나 TV 를 켠다). 10초 안에 말이 없으면 그 문장은 건너뛴다.

결과: logs/vad_live/<시각>/ 에 문장별 녹음(전체·방식별)과 results.json 이 남는다.
"""

import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import pyaudio
import webrtcvad
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from mic_vad import ChipVAD, first_channel, input_channels   # noqa: E402
from stt.stt_clova import ClovaSTTHandler                    # noqa: E402

RATE = 16000
FRAME_MS = 30
CHUNK = RATE * FRAME_MS // 1000
SILENCE_END_MS = 1500        # main.MIC_SILENCE_END_MS 와 같게
ENERGY_THR = 500.0           # LiveRecorder.min_energy_threshold 기본값
MAX_WAIT_S = 10              # 이만큼 말이 없으면 이번 문장은 넘어간다
MAX_RECORD_S = 15            # LiveRecorder 기본(10초)보다 넉넉히 — 끝을 못 찾는지 보려고

SENTENCES = [
    "오늘 점심에 딸이랑 짜장면 먹었어",
    "요즘 무릎이 자꾸 아파서 병원에 가 봐야겠어",
    "손주가 다음 주 토요일에 놀러 온대",
    "아침 약은 먹었는데 저녁 약은 아직 안 먹었어",
    "그러니까 말이야 … 어제 시장에 가서 고등어를 샀거든",
    "날씨가 추워져서 … 내복을 꺼내 입었어",
    "모리야 오늘 무슨 요일이야",
    "우리 영감이 젊었을 때 노래를 참 잘 불렀어",
    "음 … 그게 뭐더라 … 아 맞다 김장해야 돼",
    "텔레비전에서 트로트 경연 프로그램을 봤는데 재밌더라",
]
CONDITIONS = {"q": "조용", "m": "음악", "t": "TV"}
AUTO_ROUNDS = ["조용", "음악", "TV"]
AUTO_SWITCH_S = 15


def beep(times: int = 1) -> None:
    """스피커(.env 의 SPEAKER_ALSA_DEVICE)로 짧게 '띵'. ReSpeaker 잭이면 칩이 에코로 지운다."""
    import audio_out
    t = np.arange(int(RATE * 0.12)) / RATE
    tone = (np.sin(2 * np.pi * 880 * t) * np.hanning(len(t)) * 12000).astype(np.int16)
    gap = np.zeros(int(RATE * 0.12), np.int16)
    pcm = np.concatenate([np.concatenate([tone, gap]) for _ in range(times)])
    subprocess.run(["aplay", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1",
                    "-D", audio_out.SPEAKER_ALSA_DEVICE, "-"], input=pcm.tobytes())


class Endpointer:
    """LiveRecorder.record_until_silence 와 같은 규칙으로 시작·끝 프레임만 계산한다.
    - 시작: 최근 10프레임 중 6프레임 이상이 말이면 시작, 그 10프레임부터 녹음
    - 끝: 연속 침묵이 SILENCE_END_MS 이어지면 끝. 3프레임 넘게 이어진 말만 침묵을 되돌린다
    """

    def __init__(self):
        self.win = deque(maxlen=10)
        self.start = None        # 녹음이 시작되는 프레임 번호(앞 버퍼 포함)
        self.end = None          # 녹음이 끝나는 프레임 번호
        self.hit_max = False
        self._silence = 0
        self._speech = 0
        self._end_frames = SILENCE_END_MS // FRAME_MS
        self._max_frames = MAX_RECORD_S * 1000 // FRAME_MS

    @property
    def done(self):
        return self.end is not None

    def feed(self, i: int, valid: bool) -> None:
        if self.done:
            return
        if self.start is None:
            self.win.append(1 if valid else 0)
            if len(self.win) == self.win.maxlen and sum(self.win) > 5:
                self.start = max(0, i - 9)
            return
        if valid:
            self._speech += 1
            if self._speech >= 3:
                self._silence = 0
        else:
            self._speech = 0
            self._silence += 1
        if self._silence >= self._end_frames:
            self.end = i
        elif i - self.start >= self._max_frames:
            self.end, self.hit_max = i, True


def cer(ref: str, hyp: str) -> float:
    r = [c for c in ref if c.isalnum()]
    h = [c for c in hyp if c.isalnum()]
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def save_wav(path: str, frames: list[bytes]) -> None:
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(frames))


def listen_once(stream, channels, vad, chip):
    """두 방식이 모두 끝을 찾을 때까지 듣는다. (프레임들, {방식: Endpointer})"""
    eps = {"web": Endpointer(), "chip": Endpointer()}
    frames: list[bytes] = []
    chip_frames = 0
    chip.start()
    try:
        stream.read(stream.get_read_available(), exception_on_overflow=False)  # 묵은 소리 버림
        i = 0
        while True:
            data = first_channel(stream.read(CHUNK, exception_on_overflow=False), channels)
            frames.append(data)
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            loud = float(np.sqrt(np.mean(x ** 2))) > ENERGY_THR
            web = vad.is_speech(data, RATE)
            c = chip.speech_detected()
            if c is None:
                c = web
            else:
                chip_frames += 1
            eps["web"].feed(i, web and loud)
            eps["chip"].feed(i, c and loud)
            i += 1
            if all(e.done for e in eps.values()):
                break
            if i >= MAX_WAIT_S * 1000 // FRAME_MS and all(e.start is None for e in eps.values()):
                break
    finally:
        chip.stop()
    if chip_frames == 0:
        print("⚠️  칩 판정을 못 읽었습니다 — chip 결과는 webrtcvad 와 같습니다")
    return frames, eps


def run_trial(n, cond, text, out_dir, stream, channels, vad, chip, stt):
    """한 문장을 듣고 두 방식으로 잘라 받아 적는다. 결과 한 줄(dict)."""
    print(f"📜 [{n}] ({cond}) 「{text}」", flush=True)
    print("🎙️  지금 읽어 주세요...", flush=True)
    frames, eps = listen_once(stream, channels, vad, chip)
    truth = text.replace("…", " ")
    save_wav(os.path.join(out_dir, f"{n:02d}_all.wav"), frames)

    row = {"n": n, "cond": cond, "text": truth}
    for mode, ep in eps.items():
        if ep.start is None:
            row[mode] = {"hyp": "", "cer": 1.0, "hit_max": False}
            print(f"   {mode:<4} 말 시작을 못 잡음")
            continue
        end = ep.end if ep.end is not None else len(frames) - 1
        path = os.path.join(out_dir, f"{n:02d}_{mode}.wav")
        save_wav(path, frames[ep.start:end + 1])
        hyp = stt.transcribe(path)[0]
        row[mode] = {
            "start_s": ep.start * FRAME_MS / 1000,
            "end_s": end * FRAME_MS / 1000,
            "hit_max": ep.hit_max,
            "hyp": hyp,
            "cer": cer(truth, hyp),
        }
        flag = " ⏱️ 끝 못 찾음(최대 길이)" if ep.hit_max else ""
        print(f"   {mode:<4} {row[mode]['start_s']:5.2f}~{row[mode]['end_s']:5.2f}s  "
              f"CER {row[mode]['cer']:.2f}  「{hyp}」{flag}")
    w, c = row["web"], row["chip"]
    if "end_s" in w and "end_s" in c:
        diff = c["end_s"] - w["end_s"]
        print(f"   → 칩이 {abs(diff):.2f}s {'먼저' if diff < 0 else '늦게'} 끝냄"
              if abs(diff) >= 0.03 else "   → 끝난 시점 같음", flush=True)
    return row


def main():
    auto = "--auto" in sys.argv
    out_dir = os.path.join("logs", "vad_live", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    stt = ClovaSTTHandler()
    vad = webrtcvad.Vad(2)
    chip = ChipVAD()

    pa = pyaudio.PyAudio()
    channels = input_channels(pa, None)
    stream = pa.open(format=pyaudio.paInt16, channels=channels, rate=RATE, input=True,
                     frames_per_buffer=CHUNK)
    trial = (out_dir, stream, channels, vad, chip, stt)

    rows = []
    try:
        if auto:
            n = 0
            for r, cond in enumerate(AUTO_ROUNDS):
                # 첫 조건 앞에서도 기다린다 — 준비할 시간
                wait = AUTO_SWITCH_S * 2 if r == 0 else AUTO_SWITCH_S
                print(f"\n🔔 조건: 「{cond}」 — {wait}초 뒤 시작", flush=True)
                beep(3)
                time.sleep(wait)
                for text in SENTENCES:
                    n += 1
                    print()
                    beep(1)
                    rows.append(run_trial(n, cond, text, *trial))
                    time.sleep(1.0)
        else:
            cond = "조용"
            idx = 0
            while idx < len(SENTENCES) * 3:          # 문장 목록을 최대 세 바퀴
                key = input(f"\n조건 [{cond}] (Enter 그대로 / q 조용 / m 음악 / t TV / "
                            f"s 건너뛰기 / x 끝) > ").strip().lower()
                if key == "x":
                    break
                if key == "s":
                    idx += 1
                    continue
                cond = CONDITIONS.get(key, cond)
                row = run_trial(len(rows) + 1, cond, SENTENCES[idx % len(SENTENCES)], *trial)
                if input("   이 결과 쓸까요? (Enter 예 / r 다시) > ").strip().lower() == "r":
                    continue
                rows.append(row)
                idx += 1
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)

    summarize(rows)
    print(f"\n💾 {out_dir}")


def summarize(rows):
    if not rows:
        return
    print("\n== 요약 ==")
    for cond in dict.fromkeys(r["cond"] for r in rows):
        rs = [r for r in rows if r["cond"] == cond]
        print(f"[{cond}] {len(rs)}문장")
        for mode in ("web", "chip"):
            cers = [r[mode]["cer"] for r in rs]
            miss = sum("start_s" not in r[mode] for r in rs)
            maxed = sum(r[mode].get("hit_max", False) for r in rs)
            print(f"   {mode:<4} CER 평균 {np.mean(cers):.3f} | 시작 못 잡음 {miss} | 끝 못 찾음 {maxed}")
        diffs = [r["chip"]["end_s"] - r["web"]["end_s"] for r in rs
                 if "end_s" in r["chip"] and "end_s" in r["web"]]
        if diffs:
            print(f"   끝난 시점: 칩 - webrtcvad 평균 {np.mean(diffs):+.2f}s "
                  f"(음수면 칩이 먼저 끝냄)")


if __name__ == "__main__":
    main()
