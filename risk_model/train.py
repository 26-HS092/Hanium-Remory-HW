"""소형 위험 판별 모델(KoELECTRA-small) 학습 → ONNX 내보내기.

Claude Opus 가 감성대화 말뭉치에 붙인 라벨(build_risk_train.py)을 작은 모델에 옮겨 담는다
(지식 증류). 파이에서 인터넷 없이 수십 ms 에 돈다. 규칙과 LLM 사이의 두 번째 방어선이다.

    python risk_model/train.py                    # 학습 + ONNX 내보내기
    python risk_model/train.py --epochs 1 --limit 300   # 빠르게 한 바퀴 확인

입력  "어르신 말 [SEP] 바로 앞에 하신 말"  (앞말은 절반은 지우고 학습해 없을 때도 쓴다)
출력  세 머리를 함께 배운다(다중 과제)
  action  none | self_harm | harm_others | medical | abuse   — 모리가 대응할 위험(eval 정답과 같은 규칙)
  level   0~3                                              — 강도
  frame   now | past | other | negated | hypothetical      — 누구의 언제 이야기인가
결과
  models/risk_ko.onnx       파이용 (risk_model/runtime.py 가 읽는다)
  models/risk_ko/           토크나이저와 라벨 이름
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "data" / "risk_train" / "labels.jsonl"
AUGMENT = ROOT / "data" / "risk_train" / "augment.jsonl"   # build_risk_augment.py (합성)
EVAL = ROOT / "data" / "risk_eval" / "candidates.csv"
OUT_DIR = ROOT / "models" / "risk_ko"
OUT_ONNX = ROOT / "models" / "risk_ko.onnx"
BASE = "monologg/koelectra-small-v3-discriminator"
MAX_LEN = 96

ACTIONS = ["none", "self_harm", "harm_others", "medical", "abuse"]
FRAMES = ["now", "past", "other", "negated", "hypothetical"]


def action_of(kind: str, level: int, frame: str) -> str:
    """모리가 대응해야 하는 위험. build_risk_eval.export_eval 과 같은 규칙이다."""
    need = 1 if kind == "medical" else 2
    return kind if kind != "none" and frame == "now" and level >= need else "none"


class RiskModel(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.action = nn.Linear(h, len(ACTIONS))
        self.level = nn.Linear(h, 4)
        self.frame = nn.Linear(h, len(FRAMES))

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.enc(input_ids=input_ids, attention_mask=attention_mask,
                     token_type_ids=token_type_ids).last_hidden_state[:, 0]
        x = self.drop(x)
        return self.action(x), self.level(x), self.frame(x)


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        lv = int(d["level"])
        rows.append({"id": d.get("id", ""), "text": d["text"], "prev": d.get("prev", ""),
                     "action": ACTIONS.index(action_of(d["kind"], lv, d["frame"])),
                     "level": lv, "frame": FRAMES.index(d["frame"])})
    random.Random(0).shuffle(rows)
    return rows[:limit] if limit else rows


def encode(tok, rows, drop_prev: float, rng: random.Random):
    a = [r["text"] for r in rows]
    b = [r["prev"] if r["prev"] and rng.random() >= drop_prev else "" for r in rows]
    enc = tok(a, b, truncation="only_second", max_length=MAX_LEN, padding=True,
              return_tensors="pt")
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
    return enc


def macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        if tp + fp + fn:
            f.append(2 * tp / (2 * tp + fp + fn))
    return sum(f) / len(f) if f else 0.0


@torch.no_grad()
def predict(model, tok, rows, bs=64):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        enc = encode(tok, rows[i:i + bs], 0.0, random.Random(0))
        a, _, _ = model(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
        out += torch.softmax(a, -1).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--augment", type=Path, default=AUGMENT, help="없으면 건너뛴다")
    ap.add_argument("--out", type=Path, default=OUT_ONNX)
    ap.add_argument("--threads", type=int, default=3)   # 모리가 돌 몫을 남긴다
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    rows = load_rows(args.labels, args.limit)
    n_val = max(1, len(rows) // 10)
    val, tr = rows[:n_val], rows[n_val:]
    # 합성 문장은 학습에만 넣는다. 검증 문장을 씨앗으로 만든 것은 빼야 검증이 새지 않는다
    # (id: aug-<씨앗 id>-<번호>).
    if args.augment and args.augment.exists() and not args.limit:
        val_ids = {r["id"] for r in val}
        aug = [r for r in load_rows(args.augment, None)
               if r["id"].rsplit("-", 1)[0].removeprefix("aug-") not in val_ids]
        print(f"합성 {len(aug)}문장을 학습에 더함")
        tr += aug
    cnt = np.bincount([r["action"] for r in tr], minlength=len(ACTIONS))
    print(f"학습 {len(tr)} / 검증 {len(val)}   action 분포 {dict(zip(ACTIONS, cnt.tolist()))}")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = RiskModel(BASE)
    # 위험 쪽이 드물다. 놓치는 게 더 비싸니 드문 쪽에 무게를 준다(너무 크면 오탐이 는다).
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) ** 0.5, dtype=torch.float)
    w = w / w[0]
    ce_a = nn.CrossEntropyLoss(weight=w)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(tr) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    rng = random.Random(0)
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(tr)
        t0, tot = time.time(), 0.0
        for i in range(0, len(tr), args.bs):
            batch = tr[i:i + args.bs]
            enc = encode(tok, batch, 0.5, rng)
            a, l, f = model(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
            loss = (ce_a(a, torch.tensor([r["action"] for r in batch]))
                    + 0.5 * ce(l, torch.tensor([r["level"] for r in batch]))
                    + 0.5 * ce(f, torch.tensor([r["frame"] for r in batch])))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(batch)
        probs = predict(model, tok, val)
        pred = [int(np.argmax(p)) for p in probs]
        y = [r["action"] for r in val]
        f1 = macro_f1(y, pred, len(ACTIONS))
        risk_y = [v != 0 for v in y]
        risk_p = [v != 0 for v in pred]
        rec = sum(a and b for a, b in zip(risk_y, risk_p)) / max(1, sum(risk_y))
        fpr = sum(b and not a for a, b in zip(risk_y, risk_p)) / max(1, risk_y.count(False))
        print(f"epoch {ep + 1}  loss {tot / len(tr):.3f}  검증 macro-F1 {f1:.3f}  "
              f"위험 재현율 {rec:.2f}  오탐율 {fpr:.3f}  ({time.time() - t0:.0f}s)")
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(OUT_DIR)
    (OUT_DIR / "labels.json").write_text(json.dumps(
        {"action": ACTIONS, "frame": FRAMES, "max_len": MAX_LEN, "val_macro_f1": best},
        ensure_ascii=False, indent=1), encoding="utf-8")
    enc = encode(tok, [{"text": "죽고 싶어", "prev": "요즘 힘들어"}], 0.0, random.Random(0))
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
        str(args.out), input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["action", "level", "frame"],
        dynamic_axes={k: {0: "b", 1: "t"} for k in
                      ["input_ids", "attention_mask", "token_type_ids"]}
        | {k: {0: "b"} for k in ["action", "level", "frame"]},
        opset_version=17, dynamo=False,
    )
    print(f"저장: {args.out} ({args.out.stat().st_size / 1e6:.1f}MB), 검증 macro-F1 {best:.3f}")


if __name__ == "__main__":
    main()
