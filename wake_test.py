"""
wake_test.py
------------
"모리야" 웨이크워드만 따로 잰다. 깨어나도 대화로 넘어가지 않아서 기다릴 필요가 없다.
main.py 는 꺼 두기 — 마이크를 같이 못 연다.

1) 인식률: 화면에 "모리야!" 가 뜨면 한 번 부른다. 그걸 N번.
     python wake_test.py                  # 50번
     python wake_test.py -n 20            # 20번
   한 번마다 그 몇 초 동안 나온 가장 높은 점수를 남긴다. 그래서 한 번만 해 두면
   문턱 0.7 / 0.8 / 0.9 / 0.95 일 때의 인식률이 한꺼번에 나온다.

2) 오인식: 사람 없이 TV 를 틀어 두고, 그동안 몇 번 잘못 깨는지 센다.
     python wake_test.py --tv             # 60분
     python wake_test.py --tv 10          # 10분
   몇 분을 듣든 '시간당 몇 번' 으로 바꿔서도 보여준다.

결과는 logs/wake_test/ 에 csv 로 남는다. 못 알아들은 소리와 잘못 깬 소리는 wav 로도 남겨
나중에 들어 볼 수 있다. Ctrl+C 로 언제 멈춰도 거기까지의 결과를 낸다.

점수 계산은 main.py 와 같다(wakeword.py 의 같은 모델·2.1초 창·0.25초 간격).
"""

import argparse
import csv
import queue
import sys
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from wakeword import (
    BUFFER_SAMPLES,
    DETECTION_COOLDOWN_SEC,
    INFER_INTERVAL_SEC,
    SAMPLE_RATE,
    MoriyaWakeWordDetector,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "logs" / "wake_test"

# 결과를 볼 문턱들. 기기는 사람이 있으면 0.7, 카메라에 안 보이면 0.85, 레이더도 없다고 하면 0.95.
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
# 문턱을 넘은 뒤 이만큼 더 들어서 그 '모리야'의 가장 높은 점수를 마저 본다.
PEAK_EXTRA_SEC = 0.75


class Mic:
    """마이크를 계속 열어 두고 조각(1280 샘플)을 큐로 받는다."""

    def __init__(self, device=None):
        import sounddevice as sd

        self.q = queue.Queue()
        dev = sd.query_devices(device, "input")
        # ReSpeaker 6채널 펌웨어는 6으로 열고 0번(칩이 처리한 소리)만 쓴다 — wakeword.py 와 같다.
        ch = 6 if "ReSpeaker" in dev["name"] and dev["max_input_channels"] >= 6 else 1
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=ch, dtype="int16", blocksize=1280,
            device=device, callback=lambda indata, *_: self.q.put(indata[:, 0].copy()),
        )
        self.stream.start()
        print(f"🎙  마이크: {dev['name']}")

    def read(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self):
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                return

    def close(self):
        self.stream.stop()
        self.stream.close()


class Scorer:
    """조각을 받아 0.25초 분량이 쌓일 때마다 최근 2.1초 소리의 점수를 낸다."""

    def __init__(self, det: MoriyaWakeWordDetector):
        self.det = det
        self.hop = int(SAMPLE_RATE * INFER_INTERVAL_SEC)
        self.reset()

    def reset(self):
        # 창을 0(완전한 무음)으로 채워 두면 안 된다. 실제 마이크엔 늘 방 소리가 깔려 있어서,
        # 0 이 섞인 창은 모델이 처음 보는 소리가 되고 점수가 낮게 나왔다. 그래서 비워 두고
        # 실제 소리로 2.1초가 다 찰 때까지는 점수를 내지 않는다(main.py 와 같다).
        self.buf = deque(maxlen=BUFFER_SAMPLES)
        self.since = 0
        self.audio = None

    def fill(self, mic: "Mic"):
        """지금 방 소리로 창을 새로 채운다(2.1초). 앞 차례의 '모리야' 가 남지 않게."""
        self.reset()
        mic.drain()
        while len(self.buf) < BUFFER_SAMPLES:
            chunk = mic.read(timeout=0.5)
            if chunk is not None:
                self.buf.extend(chunk.tolist())
        self.since = 0

    def feed(self, chunk: np.ndarray) -> float | None:
        self.buf.extend(chunk.tolist())
        self.since += len(chunk)
        if self.since < self.hop or len(self.buf) < BUFFER_SAMPLES:
            return None
        self.since = 0
        self.audio = np.array(self.buf, dtype=np.int16)
        return self.det.predict_score(self.audio)


def save_wav(path: Path, audio: np.ndarray):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype(np.int16).tobytes())


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"📄 {path.relative_to(ROOT)}")


def thresholds_with(thr: float) -> list[float]:
    return sorted(set(THRESHOLDS + [thr]))


# ── 1) 인식률 ──────────────────────────────────────────────────────────────────

def one_call(mic: Mic, scorer: Scorer, window: float, thr: float):
    """한 번 부르는 동안 들은 가장 높은 점수와 그때의 소리. 창은 미리 채워 둔다(fill)."""
    peak, peak_audio = 0.0, None
    heard = 0                       # 이번 차례에 받은 샘플 수
    stop_at = int(window * SAMPLE_RATE)
    deadline = time.monotonic() + window + 3   # 마이크가 멈춰도 갇히지 않게
    while heard < stop_at and time.monotonic() < deadline:
        chunk = mic.read(timeout=0.5)
        if chunk is None:
            continue
        heard += len(chunk)
        s = scorer.feed(chunk)
        if s is None:
            continue
        if s > peak:
            peak, peak_audio = s, scorer.audio
        if s >= thr:
            # 넘었다. 조금만 더 들어서 최고점을 보고 끝낸다.
            stop_at = min(stop_at, heard + int(PEAK_EXTRA_SEC * SAMPLE_RATE))
    if peak_audio is None:
        peak_audio = np.array(scorer.buf, dtype=np.int16)
    return peak, peak_audio


def run_calls(mic: Mic, scorer: Scorer, n: int, window: float, thr: float, stamp: str):
    print(f"\n화면에 '🗣  모리야!' 가 뜨면 한 번 불러 주세요. {n}번, 한 번에 {window:.0f}초씩 기다려요.")
    print(f"감지 기준은 {thr:.2f} 이에요. 차례 사이 2초쯤은 조용히 기다려 주세요. 멈추려면 Ctrl+C.")
    input("준비되면 Enter ▶ ")

    miss_dir = OUT_DIR / f"calls_{stamp}_miss"
    rows = []
    try:
        for i in range(1, n + 1):
            scorer.fill(mic)    # 2.1초 동안 방 소리로 창을 채운다 — 이 동안은 조용히
            print(f"[{i:2d}/{n}] 🗣  모리야!", end="   ", flush=True)
            peak, audio = one_call(mic, scorer, window, thr)
            hit = peak >= thr
            if not hit:
                miss_dir.mkdir(parents=True, exist_ok=True)
                save_wav(miss_dir / f"{i:02d}_{peak:.3f}.wav", audio)
            rows.append({"trial": i, "peak": round(peak, 4), "hit": int(hit)})
            got = sum(r["hit"] for r in rows)
            mark = "✅" if hit else "❌ 못 들음"
            print(f"{mark}  점수 {peak:.3f}   (지금까지 {got}/{i})")
    except KeyboardInterrupt:
        print("\n⏹  멈춤")

    if not rows:
        return
    n_done = len(rows)
    print(f"\n━━ 인식률 ({n_done}번 부름) ━━")
    for t in thresholds_with(thr):
        k = sum(r["peak"] >= t for r in rows)
        now = "   ← 이번 기준" if t == thr else ""
        print(f"  문턱 {t:.2f}:  {k:3d}/{n_done}   {k / n_done * 100:5.1f}%{now}")
    peaks = sorted(r["peak"] for r in rows)
    print(f"  점수 중앙값 {np.median(peaks):.3f} · 가장 낮은 것 {peaks[0]:.3f}")
    if miss_dir.exists():
        print(f"  못 들은 소리: {miss_dir.relative_to(ROOT)}/")
    write_csv(OUT_DIR / f"calls_{stamp}.csv", rows)


# ── 2) 오인식 ──────────────────────────────────────────────────────────────────

def run_tv(mic: Mic, scorer: Scorer, minutes: float, thr: float, stamp: str):
    ths = thresholds_with(thr)
    low = ths[0]
    counts = {t: 0 for t in ths}
    # 문턱마다 마지막으로 넘은 시각. 2초 넘게 안 넘다가 다시 넘어야 새로 한 번 깬 걸로 센다
    # (길게 이어진 소리 하나를 여러 번 세지 않게).
    last = {t: -1e9 for t in ths}
    events = []
    ev_dir = OUT_DIR / f"tv_{stamp}_wake"

    total = int(minutes * 60 * SAMPLE_RATE)
    heard = 0
    next_report = 60 * SAMPLE_RATE
    print(f"\n📺 {minutes:g}분 동안 듣습니다. 사람은 자리를 비워 주세요. 도중에 멈추려면 Ctrl+C.")
    try:
        while heard < total:
            chunk = mic.read()
            if chunk is None:
                continue
            heard += len(chunk)
            if heard >= next_report:
                next_report += 60 * SAMPLE_RATE
                print(f"⏱  {heard // (60 * SAMPLE_RATE):.0f}/{minutes:g}분 — "
                      f"문턱 {thr:.2f} 에서 잘못 깬 횟수 {counts[thr]}", flush=True)
            s = scorer.feed(chunk)
            if s is None:
                continue
            now = heard / SAMPLE_RATE
            for t in ths:
                if s < t:
                    continue
                new = now - last[t] >= DETECTION_COOLDOWN_SEC
                last[t] = now
                if new:
                    counts[t] += 1
                # 가장 낮은 문턱을 넘은 소리 하나하나를 기록한다(점수는 그 소리의 최고점).
                if t == low:
                    if new:
                        events.append({"time_sec": round(now, 2),
                                       "clock": datetime.now().strftime("%H:%M:%S"),
                                       "peak": round(s, 4)})
                    else:
                        events[-1]["peak"] = round(max(events[-1]["peak"], s), 4)
                if t == thr and new:
                    ev_dir.mkdir(parents=True, exist_ok=True)
                    save_wav(ev_dir / f"{int(now):05d}s_{s:.3f}.wav", scorer.audio)
                    print(f"🚨 잘못 깸  {datetime.now():%H:%M:%S}  점수 {s:.3f}", flush=True)
    except KeyboardInterrupt:
        print("\n⏹  멈춤")

    hours = heard / SAMPLE_RATE / 3600
    if hours <= 0:
        return
    print(f"\n━━ 오인식 ({hours * 60:.1f}분 들음) ━━")
    for t in ths:
        now = "   ← 이번 기준" if t == thr else ""
        print(f"  문턱 {t:.2f}:  {counts[t]:3d}번   시간당 {counts[t] / hours:5.1f}번{now}")
    print("  (기기는 사람이 있으면 0.7, 카메라에 안 보이면 0.85, 레이더도 없다고 하면 0.95 를 써요)")
    if ev_dir.exists():
        print(f"  잘못 깬 소리: {ev_dir.relative_to(ROOT)}/")
    write_csv(OUT_DIR / f"tv_{stamp}.csv",
              events or [{"time_sec": "", "clock": "", "peak": ""}])


def main():
    ap = argparse.ArgumentParser(description="'모리야' 웨이크워드 인식률·오인식 재기")
    ap.add_argument("-n", type=int, default=50, help="부를 횟수 (기본 50)")
    ap.add_argument("--window", type=float, default=4.0, help="한 번에 기다리는 초 (기본 4)")
    ap.add_argument("--tv", type=float, nargs="?", const=60, metavar="분",
                    help="오인식 모드: 이만큼 듣고 잘못 깬 횟수를 셈 (숫자 없으면 60분)")
    ap.add_argument("--thr", type=float, default=0.7, help="감지 기준 (기본 0.7, main.py 와 같음)")
    ap.add_argument("--model", default=str(ROOT / "models" / "moriya_v5.onnx"))
    ap.add_argument("--device", type=int, default=None, help="마이크 번호 (기본: 시스템 기본 입력)")
    a = ap.parse_args()

    det = MoriyaWakeWordDetector(
        moriya_model_path=Path(a.model), models_dir=ROOT / "models",
        threshold=a.thr, input_device_index=a.device,
    )
    try:
        mic = Mic(a.device)
    except Exception as e:
        print(f"❌ 마이크를 못 열었어요: {e}")
        print("   main.py 가 켜져 있으면 끄고 다시 해 주세요.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scorer = Scorer(det)
    try:
        if a.tv:
            run_tv(mic, scorer, a.tv, a.thr, stamp)
        else:
            run_calls(mic, scorer, a.n, a.window, a.thr, stamp)
    finally:
        mic.close()


if __name__ == "__main__":
    main()
