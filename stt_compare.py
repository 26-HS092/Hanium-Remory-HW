"""
stt_compare.py
--------------
같은 녹음으로 STT 엔진·설정을 견주어 본다.

준비: 기기에서 SAVE_STT_WAV=1 로 대화해 stt_samples/ 에 녹음을 모은다(stt/stt_samples.py).

1) 정답 적기
     python stt_compare.py label
   stt_samples/labels.tsv 를 만들거나 새 녹음만 덧붙인다. 칸은
     id · truth(정답) · keywords(핵심 단어, 쉼표) · live_stt(그때 받아 적은 글)
   truth 는 LLM 이 바로잡은 말(없으면 STT 원문)로 미리 채워 두니, 녹음을 들어 보며
   틀린 데만 고친다. 들어 보려면 stt_samples/wav/{id}.wav.
   - 말이 아닌 소음이었으면 truth 를 비워 둔다(-) → 비교에서 뺀다.
   - keywords 를 비우면 부스팅 목록(stt/stt_keywords.py) 중 truth 에 든 말을 쓴다.

2) 엔진 비교
     python stt_compare.py run                         # 쓸 수 있는 엔진 모두
     python stt_compare.py run -e clova_raw,clova,clova_speech
   엔진 이름
     clova_raw          CLOVA CSR, 소리 크기 맞추기 없이(지금까지의 방식)
     clova              CLOVA CSR + 소리 크기 맞추기
     clova_speech_raw   CLOVA Speech, 부스팅·크기 맞추기 없이
     clova_speech       CLOVA Speech + 이름 부스팅 + 크기 맞추기
     groq               Groq Whisper large-v3 + 이름 힌트 + 크기 맞추기
   엔진마다 표본 수만큼 API 를 부른다(요금 주의). -n 으로 개수를 줄일 수 있다.

3) 결과
   - CER        글자 오류율(띄어쓰기·문장부호 빼고). 낮을수록 좋다.
   - 핵심단어   keywords 를 몇 개나 맞게 받아 적었나
   - 완전일치   정답과 글자가 똑같은 비율
   - 지연       평균 / 가장 느린 것(초)
   그리고 기기에서 실제로 쓴 결과(live STT)와 LLM 바로잡기(heard)도 같은 기준으로 잰다.
   heard 가 live STT 보다 CER 이 낮으면 바로잡기가 도움이 된 것이고,
   '잘못 고침' 은 STT 는 맞았는데 LLM 이 틀리게 바꾼 수다.
   표본별 결과는 stt_samples/results_{시각}.csv 로 남는다.
"""

import argparse
import csv
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "stt"))
load_dotenv(ROOT / ".env")

from stt_keywords import boost_words  # noqa: E402

PATIENT_ID = os.getenv("PATIENT_ID", "P001")
SAMPLE_DIR = Path(os.getenv("STT_SAMPLE_DIR", str(ROOT / "stt_samples")))
LABELS = "labels.tsv"
FIELDS = ["id", "truth", "keywords", "live_stt"]


# ───────────────────────────── 기록 읽기 ─────────────────────────────

def load_samples(base: Path) -> dict[str, dict]:
    """samples.jsonl → {id: {stt_text, prev, heard, ...}} (기록 순서대로)."""
    out: dict[str, dict] = {}
    path = base / "samples.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        sid = r.get("id")
        if not sid:
            continue
        if r.get("kind") == "stt":
            out.setdefault(sid, {}).update(r)
        elif r.get("kind") == "heard" and sid in out:
            out[sid]["heard"] = r.get("heard", "")
    return out


def load_labels(base: Path) -> list[dict]:
    path = base / LABELS
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [{k: (row.get(k) or "") for k in FIELDS}
                for row in csv.DictReader(f, delimiter="\t")]


def write_labels(base: Path, rows: list[dict]) -> None:
    with (base / LABELS).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def cmd_label(args) -> None:
    base = SAMPLE_DIR
    samples = load_samples(base)
    if not samples:
        sys.exit(f"녹음 기록이 없습니다: {base / 'samples.jsonl'} (기기에서 SAVE_STT_WAV=1)")
    rows = load_labels(base)
    have = {r["id"] for r in rows}
    added = 0
    for sid, s in samples.items():
        if sid in have or not (base / "wav" / f"{sid}.wav").exists():
            continue
        stt = s.get("stt_text", "")
        guess = s.get("heard") or stt
        rows.append({"id": sid, "truth": guess or "-", "keywords": "", "live_stt": stt})
        added += 1
    write_labels(base, rows)
    print(f"{base / LABELS}: 새로 {added}개 추가, 모두 {len(rows)}개")
    print("녹음을 들어 보며 truth 칸의 틀린 곳을 고쳐 주세요(소음이면 '-').")


# ───────────────────────────── 채점 ─────────────────────────────

def _norm(text: str) -> str:
    """띄어쓰기·문장부호를 빼고 비교한다(한국어는 띄어쓰기가 사람마다 달라서)."""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text or "").lower()


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> tuple[int, int]:
    """(틀린 글자 수, 정답 글자 수)."""
    r, h = _norm(ref), _norm(hyp)
    return _edit_distance(h, r), len(r)


def keywords_for(row: dict, boost: list[str]) -> list[str]:
    if row["keywords"].strip():
        return [k.strip() for k in row["keywords"].split(",") if k.strip()]
    truth = _norm(row["truth"])
    return [w for w in boost if _norm(w) and _norm(w) in truth]


class Score:
    def __init__(self):
        self.err = self.chars = self.exact = self.n = 0
        self.kw_hit = self.kw_total = 0
        self.lat: list[float] = []

    def add(self, hyp: str, ref: str, kws: list[str], latency: float | None = None):
        e, c = cer(hyp, ref)
        self.err += e
        self.chars += c
        self.n += 1
        self.exact += (e == 0)
        hn = _norm(hyp)
        self.kw_total += len(kws)
        self.kw_hit += sum(_norm(k) in hn for k in kws)
        if latency is not None:
            self.lat.append(latency)

    def row(self, name: str) -> list[str]:
        cer_s = f"{self.err / self.chars:.1%}" if self.chars else "-"
        kw = f"{self.kw_hit}/{self.kw_total}" if self.kw_total else "-"
        ex = f"{self.exact / self.n:.0%}" if self.n else "-"
        lat = (f"{sum(self.lat) / len(self.lat):.2f} / {max(self.lat):.2f}"
               if self.lat else "-")
        return [name, str(self.n), cer_s, kw, ex, lat]


# ───────────────────────────── 엔진 ─────────────────────────────

ENGINES = ["clova_raw", "clova", "clova_speech_raw", "clova_speech", "groq"]


def make_engine(name: str, boost: list[str]):
    """엔진을 만든다. 키가 없으면 None(건너뜀)."""
    try:
        if name in ("clova_raw", "clova"):
            from stt_clova import ClovaSTTHandler
            return ClovaSTTHandler(normalize=(name == "clova"))
        if name in ("clova_speech_raw", "clova_speech"):
            from stt_clova_speech import ClovaSpeechSTTHandler
            full = name == "clova_speech"
            return ClovaSpeechSTTHandler(normalize=full, keywords=boost if full else [])
        if name == "groq":
            from stt_groq import GroqSTTHandler
            return GroqSTTHandler(normalize=True, keywords=boost)
    except KeyError as e:
        print(f"  ↳ {name}: .env 에 {e} 가 없어 건너뜀")
        return None
    except ImportError as e:
        print(f"  ↳ {name}: {e} — 건너뜀")
        return None
    sys.exit(f"모르는 엔진: {name} (가능: {', '.join(ENGINES)})")


def print_table(rows: list[list[str]]) -> None:
    head = ["엔진", "표본", "CER", "핵심단어", "완전일치", "지연 평균/최대(s)"]
    allr = [head] + rows

    def width(s: str) -> int:   # 한글은 두 칸
        return sum(2 if ord(ch) > 0x1100 else 1 for ch in s)

    cols = [max(width(r[i]) for r in allr) for i in range(len(head))]
    for k, r in enumerate(allr):
        print("  ".join(c + " " * (cols[i] - width(c)) for i, c in enumerate(r)))
        if k == 0:
            print("  ".join("-" * w for w in cols))


def cmd_run(args) -> None:
    base = SAMPLE_DIR
    samples = load_samples(base)
    rows = [r for r in load_labels(base)
            if r["truth"].strip() not in ("", "-")
            and (base / "wav" / f"{r['id']}.wav").exists()]
    if not rows:
        sys.exit(f"정답이 적힌 표본이 없습니다. 먼저: python stt_compare.py label ({base / LABELS})")
    if args.n:
        rows = rows[: args.n]
    boost = boost_words(PATIENT_ID)
    print(f"표본 {len(rows)}개 · 부스팅 단어 {len(boost)}개: {', '.join(boost[:15])}"
          + (" …" if len(boost) > 15 else ""))

    names = [e.strip() for e in args.engines.split(",")] if args.engines else ENGINES
    engines = {}
    for name in names:
        h = make_engine(name, boost)
        if h:
            engines[name] = h

    scores = {name: Score() for name in engines}
    live, heard = Score(), Score()
    over = fixed = n_heard = 0
    per_sample = []
    for i, row in enumerate(rows, 1):
        wav = str(base / "wav" / f"{row['id']}.wav")
        ref = row["truth"]
        kws = keywords_for(row, boost)
        rec = {"id": row["id"], "truth": ref, "keywords": ",".join(kws)}

        s = samples.get(row["id"], {})
        live_text = row["live_stt"] or s.get("stt_text", "")
        live.add(live_text, ref, kws)
        rec["live_stt"] = live_text
        if "heard" in s:
            h = s["heard"]
            heard.add(h, ref, kws)
            n_heard += 1
            rec["heard"] = h
            e_live, _ = cer(live_text, ref)
            e_heard, _ = cer(h, ref)
            over += e_heard > e_live
            fixed += e_heard < e_live

        for name, eng in engines.items():
            t = time.time()
            text, _ = eng.transcribe(wav)
            lat = time.time() - t
            scores[name].add(text, ref, kws, lat)
            e, c = cer(text, ref)
            rec[name] = text
            rec[f"{name}_cer"] = f"{e / c:.3f}" if c else ""
        per_sample.append(rec)
        print(f"  [{i}/{len(rows)}] {row['id']}", end="\r")
    print()

    table = [scores[n].row(n) for n in engines]
    table.append(live.row("live STT(기기 기록)"))
    if n_heard:
        table.append(heard.row("live heard(LLM 바로잡기)"))
    print_table(table)
    if n_heard:
        print(f"\nLLM 바로잡기: {n_heard}개 중 나아짐 {fixed} · 잘못 고침 {over}")

    out = base / f"results_{datetime.datetime.now():%Y%m%d-%H%M%S}.csv"
    keys = list(dict.fromkeys(k for r in per_sample for k in r))
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(per_sample)
    print(f"표본별 결과: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="STT 엔진 비교")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("label", help="labels.tsv 만들기/새 녹음 덧붙이기")
    r = sub.add_parser("run", help="엔진 비교")
    r.add_argument("-e", "--engines", help=f"쉼표로 구분 (기본: {','.join(ENGINES)})")
    r.add_argument("-n", type=int, help="앞에서부터 n개만")
    args = ap.parse_args()
    {"label": cmd_label, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
