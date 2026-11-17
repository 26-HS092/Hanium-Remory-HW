"""안전 규칙이 멀쩡한 대화를 위험으로 잡는 비율(오탐)을 감성대화 말뭉치로 잰다.

    python eval_corpus_fp.py                       # 전체 문장에 규칙을 돌려 요약
    python eval_corpus_fp.py --dump hits.csv       # 걸린 문장을 검토용으로 저장
    python eval_corpus_fp.py --show 20             # 종류별로 걸린 문장 20개씩 보기

말뭉치(data/text_emotion/*.csv)는 감정 정답만 있고 위험 정답은 없다. 그래서
여기서 나온 숫자는 '오탐률' 이 아니라 '걸린 비율' 이고, 걸린 문장 중 진짜
위험은 사람이 보고 가려야 한다(--dump). 대신 거의 다 일상 대화라 걸린 비율이
갑자기 오르면 규칙이 넓어졌다는 신호로 쓴다. 데이터는 라이선스 때문에 git 에
올리지 않는다(.gitignore 의 *.csv).
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import safety

DATA_DIR = Path(__file__).parent / "data" / "text_emotion"


def load_sentences(data_dir: Path = DATA_DIR) -> list[dict]:
    """어르신 쪽 문장(사람문장1~4)만 꺼낸다. 시스템 응답은 모리 말이라 뺀다."""
    out = []
    for name in ("train.csv", "test.csv"):
        path = data_dir / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig") as f:
            for row_i, r in enumerate(csv.DictReader(f)):
                for i in range(1, 5):
                    text = (r.get(f"사람문장{i}") or "").strip()
                    if text:
                        out.append({"split": name[:-4], "row": row_i, "turn": i,
                                    "emotion": r.get("감정_대분류", ""), "text": text})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="걸린 문장을 저장할 CSV 경로")
    ap.add_argument("--show", type=int, default=0, help="종류별로 보여줄 문장 수")
    args = ap.parse_args()

    sents = load_sentences()
    if not sents:
        raise SystemExit(f"말뭉치가 없습니다: {DATA_DIR}")

    hits = []
    for s in sents:
        r = safety.classify(s["text"])
        if r:
            hits.append({**s, "kind": r.kind, "matched": r.matched})

    by_kind = collections.Counter(h["kind"] for h in hits)
    by_pat = collections.Counter((h["kind"], h["matched"]) for h in hits)
    n = len(sents)
    print(f"문장 {n:,}개 중 {len(hits):,}개 걸림 ({len(hits) / n:.2%})\n")
    print("종류별")
    for kind in safety._SEVERITY:
        c = by_kind.get(kind, 0)
        print(f"  {kind:12s} {c:5d}  ({c / n:.3%})")
    print("\n많이 걸린 표현")
    for (kind, m), c in by_pat.most_common(15):
        print(f"  {c:4d}  {kind:12s} {m!r}")

    if args.show:
        for kind in safety._SEVERITY:
            rows = [h for h in hits if h["kind"] == kind][: args.show]
            if rows:
                print(f"\n## {kind}")
                for h in rows:
                    print(f"  [{h['emotion']}] {h['text'][:80]}  ← {h['matched']!r}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["split", "row", "turn", "emotion",
                                              "kind", "matched", "text", "is_risk"])
            w.writeheader()
            for h in hits:
                w.writerow({**h, "is_risk": ""})   # 사람이 채운다(1/0)
        print(f"\n저장: {args.dump}  (is_risk 칸을 채워 실제 오탐률을 낸다)")


if __name__ == "__main__":
    main()
