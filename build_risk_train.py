"""소형 위험 판별 모델을 학습할 데이터를 만든다 — 감성대화 말뭉치에 Claude 로 라벨을 붙인다.

    python build_risk_train.py submit        # 배치로 보낸다 (Message Batches, 값이 반)
    python build_risk_train.py collect       # 끝났으면 결과를 받아 저장한다 (안 끝났으면 상태만)

결과: data/risk_train/labels.jsonl  {id, prev, text, kind, level, frame, note}

평가셋(build_risk_eval.py)에 들어간 대화는 통째로 뺀다. 같은 대화의 다른 문장이
학습에 들어가면 평가 점수가 부풀려진다. 라벨 기준(_SYSTEM)은 평가셋과 같다.

뽑는 문장
  위험 낱말이 든 문장(규칙·넓은 낱말)은 전부 — 위험 쪽 예가 드물어서 다 쓴다
  나머지 일상 문장은 무작위로 채운다. 일상 문장도 3% 쯤은 우회 표현 위험이 섞여
  있어서(평가셋 random 150 중 5) 라벨 없이 '정상' 으로 두지 않고 함께 라벨을 붙인다.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import safety
from build_risk_eval import _BROAD, _SCHEMA, _SYSTEM, CHUNK, LABEL_MODEL, OUT_CSV as EVAL_CSV
from eval_corpus_fp import load_sentences

HERE = Path(__file__).parent
OUT_DIR = HERE / "data" / "risk_train"
LABELS = OUT_DIR / "labels.jsonl"
BATCH_ID = OUT_DIR / "batch_id.txt"
TOTAL = 4000
SEED = 1


def pick() -> list[dict]:
    with open(EVAL_CSV, encoding="utf-8-sig") as f:
        held_out = {(r["split"], int(r["row"])) for r in csv.DictReader(f)}
    sents = load_sentences()
    by_dialog: dict = {}
    for s in sents:
        by_dialog.setdefault((s["split"], s["row"]), {})[s["turn"]] = s["text"]

    risky, plain, seen = [], [], set()
    for s in sents:
        if (s["split"], s["row"]) in held_out or s["text"] in seen:
            continue
        seen.add(s["text"])
        s = {"id": f"{s['split']}-{s['row']}-{s['turn']}", "text": s["text"],
             "prev": by_dialog[(s["split"], s["row"])].get(s["turn"] - 1, "")}
        (risky if safety.classify(s["text"]) or _BROAD.search(s["text"]) else plain).append(s)
    random.Random(SEED).shuffle(plain)
    return risky + plain[: max(0, TOTAL - len(risky))]


def submit() -> None:
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if BATCH_ID.exists():
        raise SystemExit(f"이미 보낸 배치가 있습니다: {BATCH_ID.read_text().strip()} (collect 로 받기)")

    items = pick()
    (OUT_DIR / "items.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    requests = []
    for i in range(0, len(items), CHUNK):
        chunk = [{"id": x["id"], "prev": x["prev"], "text": x["text"]}
                 for x in items[i:i + CHUNK]]
        requests.append({
            "custom_id": f"c{i // CHUNK:04d}",
            "params": {
                "model": LABEL_MODEL,
                "max_tokens": 8000,
                "system": [{"type": "text", "text": _SYSTEM,
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": "다음 문장들을 판정하세요.\n"
                              + json.dumps(chunk, ensure_ascii=False)}],
                "output_config": {"effort": "medium",
                                  "format": {"type": "json_schema", "schema": _SCHEMA}},
            },
        })
    batch = anthropic.Anthropic().messages.batches.create(requests=requests)
    BATCH_ID.write_text(batch.id)
    print(f"보냄: 문장 {len(items)}개, 요청 {len(requests)}개 → 배치 {batch.id}")


def collect() -> None:
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    client = anthropic.Anthropic()
    batch_id = BATCH_ID.read_text().strip()
    b = client.messages.batches.retrieve(batch_id)
    print(f"배치 {batch_id}: {b.processing_status}  {b.request_counts}")
    if b.processing_status != "ended":
        return
    items = {json.loads(l)["id"]: json.loads(l)
             for l in (OUT_DIR / "items.jsonl").read_text(encoding="utf-8").splitlines()}
    n_ok = n_bad = 0
    with open(LABELS, "w", encoding="utf-8") as f:
        for res in client.messages.batches.results(batch_id):
            if res.result.type != "succeeded" or res.result.message.stop_reason != "end_turn":
                n_bad += 1
                print(f"  ⚠️  {res.custom_id}: {res.result.type}", file=sys.stderr)
                continue
            text = next(b.text for b in res.result.message.content if b.type == "text")
            for lab in json.loads(text)["labels"]:
                if lab["id"] in items:
                    f.write(json.dumps({**items[lab["id"]], **lab}, ensure_ascii=False) + "\n")
                    n_ok += 1
    print(f"저장: {LABELS}  라벨 {n_ok}개, 실패한 요청 {n_bad}개")


if __name__ == "__main__":
    {"submit": submit, "collect": collect}[sys.argv[1] if len(sys.argv) > 1 else "collect"]()
