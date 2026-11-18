"""위험 판별 모델 학습용 보강 데이터를 Claude 로 만든다 (합성).

    python build_risk_augment.py submit
    python build_risk_augment.py collect

말뭉치에서 위험 예가 드물다(4천 문장 중 자해 96, 학대 79, 의료 54, 타해 1). 작은 모델이
'뜻' 을 배우게 하려고, 학습 세트(labels.jsonl)의 위험 문장마다 짝을 만든다.
  positive  같은 위험을 어르신 말투로 다르게 (우회·체념·사투리 섞어서) 3개
  stt       음성 인식이 잘못 받아 적은 듯한 버전 1개 ("영감" → "영광" 처럼)
  negative  비슷한 낱말을 쓰지만 대응할 필요 없는 말 2개 (남 이야기·부정·지난 일·관용)
평가셋에 든 대화는 씨앗으로 쓰지 않으니(labels.jsonl 이 이미 뺐다) 평가가 새지 않는다.
결과: data/risk_train/augment.jsonl  {text, prev:"", kind, level, frame, synthetic:true}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DIR = HERE / "data" / "risk_train"
LABELS = DIR / "labels.jsonl"
OUT = DIR / "augment.jsonl"
BATCH_ID = DIR / "augment_batch_id.txt"
MODEL = "claude-opus-5-5"
CHUNK = 8

sys.path.insert(0, str(HERE / "risk_model"))
from train import action_of  # noqa: E402

_SYSTEM = """당신은 노인 돌봄 인형의 안전 판별 모델을 학습시킬 데이터를 만듭니다.
어르신(70~80대) 한국어 구어체로 씁니다. 짧고 자연스러운 혼잣말·대화체, 가끔 사투리와 맞춤법이 틀린 말투도 섞으세요.

라벨 기준
kind: none | self_harm | harm_others | medical | abuse
level: 0 없음 | 1 관용·푸념 | 2 수동적 바람·체념·지속 고통 | 3 능동적 생각·방법·시점·계획·지금 당장 위험
frame: now(본인 지금) | past(지나간 일) | other(남·TV 이야기) | negated(부정) | hypothetical(가정)
medical 은 약·병원·수술을 스스로 판단하려 하거나 판단을 묻는 말만 해당합니다(단순히 아프다는 말은 none).

씨앗 문장마다 다음을 만드세요.
- positive 3개: 씨앗과 같은 kind 의 실제 위험(frame now, level 2 이상, medical 은 1 이상). 씨앗의 낱말을 되도록 피하고 우회 표현을 쓰세요 (예: "죽고 싶다" 대신 "이제 그만 영감 곁으로 가고 싶어", "자식들 짐 안 되게 조용히 가야지").
- stt 1개: positive 하나를 음성 인식이 잘못 받아 적은 듯이 (발음이 비슷한 낱말로 한두 군데 바뀜, 띄어쓰기 틀림). 라벨은 원래 뜻대로.
- negative 2개: 비슷한 낱말을 쓰지만 대응할 필요가 없는 말. 남 이야기, 부정, 지나간 일, 관용 강조 중에서 고르고 그에 맞는 라벨을 다세요.
모든 문장은 서로 달라야 하고 씨앗을 그대로 베끼지 마세요."""

_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "seed_id": {"type": "string"},
            "role": {"type": "string", "enum": ["positive", "stt", "negative"]},
            "text": {"type": "string"},
            "kind": {"type": "string", "enum": ["none", "self_harm", "harm_others",
                                               "medical", "abuse"]},
            "level": {"type": "integer", "enum": [0, 1, 2, 3]},
            "frame": {"type": "string", "enum": ["now", "past", "other", "negated",
                                                "hypothetical"]},
        },
        "required": ["seed_id", "role", "text", "kind", "level", "frame"],
        "additionalProperties": False,
    }}},
    "required": ["items"],
    "additionalProperties": False,
}

# 말뭉치에 타해가 하나뿐이라 씨앗을 직접 둔다(평가셋 문장과 겹치지 않게 새로 씀).
_EXTRA_SEEDS = [
    {"id": "x-ho-1", "text": "저 옆집 놈을 확 죽여 버리고 싶어", "kind": "harm_others"},
    {"id": "x-ho-2", "text": "며느리가 또 그러면 칼로 어떻게 해 버릴지도 몰라", "kind": "harm_others"},
    {"id": "x-ho-3", "text": "그 사기꾼 찾아가서 가만 안 둘 거야, 몽둥이 들고", "kind": "harm_others"},
    {"id": "x-ho-4", "text": "영감이 술 먹고 들어오면 이번엔 내가 패 버릴 거야", "kind": "harm_others"},
]


def seeds() -> list[dict]:
    out = []
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        act = action_of(d["kind"], int(d["level"]), d["frame"])
        if act != "none":
            out.append({"id": d["id"], "text": d["text"], "kind": act})
    return out + _EXTRA_SEEDS


def submit() -> None:
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    if BATCH_ID.exists():
        raise SystemExit(f"이미 보낸 배치: {BATCH_ID.read_text().strip()}")
    ss = seeds()
    reqs = []
    for i in range(0, len(ss), CHUNK):
        chunk = [{"seed_id": s["id"], "text": s["text"], "kind": s["kind"]}
                 for s in ss[i:i + CHUNK]]
        reqs.append({"custom_id": f"a{i // CHUNK:04d}", "params": {
            "model": MODEL, "max_tokens": 12000,
            "system": [{"type": "text", "text": _SYSTEM,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "씨앗 문장:\n"
                          + json.dumps(chunk, ensure_ascii=False)}],
            "output_config": {"effort": "medium",
                              "format": {"type": "json_schema", "schema": _SCHEMA}},
        }})
    b = anthropic.Anthropic().messages.batches.create(requests=reqs)
    BATCH_ID.write_text(b.id)
    print(f"보냄: 씨앗 {len(ss)}개, 요청 {len(reqs)}개 → {b.id}")


def collect() -> None:
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env")
    client = anthropic.Anthropic()
    bid = BATCH_ID.read_text().strip()
    b = client.messages.batches.retrieve(bid)
    print(f"배치 {bid}: {b.processing_status}  {b.request_counts}")
    if b.processing_status != "ended":
        return
    n = bad = 0
    seen = set()
    with open(OUT, "w", encoding="utf-8") as f:
        for res in client.messages.batches.results(bid):
            if res.result.type != "succeeded" or res.result.message.stop_reason != "end_turn":
                bad += 1
                continue
            text = next(x.text for x in res.result.message.content if x.type == "text")
            for it in json.loads(text)["items"]:
                t = it["text"].strip()
                if not t or t in seen:
                    continue
                seen.add(t)
                f.write(json.dumps({"id": f"aug-{it['seed_id']}-{n}", "text": t, "prev": "",
                                    "kind": it["kind"], "level": it["level"],
                                    "frame": it["frame"], "role": it["role"],
                                    "synthetic": True}, ensure_ascii=False) + "\n")
                n += 1
    print(f"저장: {OUT}  {n}문장, 실패한 요청 {bad}개")


if __name__ == "__main__":
    {"submit": submit, "collect": collect}[sys.argv[1] if len(sys.argv) > 1 else "collect"]()
