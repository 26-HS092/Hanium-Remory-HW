"""소형 위험 판별 모델을 파이에서 돌린다 (ONNX, 인터넷 필요 없음).

    from risk_model.runtime import RiskClassifier
    clf = RiskClassifier()               # 모델이 없으면 RiskModelMissing
    r = clf.classify("그냥 영감을 따라 갈까 봐", prev="먼저 간 영감이 보고 싶어")
    r.action, r.prob, r.level, r.frame   # 'self_harm', 0.93, 2, 'now'

학습은 risk_model/train.py. transformers 를 불러오지 않고 tokenizers 만 쓴다 —
모리가 켜질 때 몇 초씩 더 걸리지 않게.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "risk_ko"
ONNX_PATH = Path(os.getenv("RISK_MODEL_PATH", ROOT / "models" / "risk_ko.int8.onnx"))


class RiskModelMissing(RuntimeError):
    pass


@dataclass
class RiskGuess:
    action: str          # none | self_harm | harm_others | medical | abuse
    prob: float          # action 의 확률
    risk_prob: float     # none 이 아닐 확률 (위험 쪽 전부 합)
    top_risk: str        # none 을 뺀 것 중 가장 그럴듯한 위험 종류
    level: int
    frame: str


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class RiskClassifier:
    def __init__(self, onnx_path: Path = ONNX_PATH, model_dir: Path = MODEL_DIR,
                 threads: int = 1):
        if not onnx_path.exists() or not (model_dir / "tokenizer.json").exists():
            raise RiskModelMissing(f"모델 없음: {onnx_path}")
        import onnxruntime as ort
        from tokenizers import Tokenizer

        meta = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        self.actions, self.frames = meta["action"], meta["frame"]
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(meta.get("max_len", 96))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads    # 대화 중엔 STT·LLM 이 바쁘다. 한 코어면 된다.
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), opts,
                                         providers=["CPUExecutionProvider"])

    def classify(self, text: str, prev: str = "") -> RiskGuess:
        enc = self.tok.encode(text, prev) if prev else self.tok.encode(text)
        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        a, lv, fr = self.sess.run(["action", "level", "frame"], feed)
        pa = _softmax(a[0])
        i = int(pa.argmax())
        j = int(pa[1:].argmax()) + 1
        return RiskGuess(action=self.actions[i], prob=float(pa[i]),
                         risk_prob=float(1.0 - pa[0]), top_risk=self.actions[j], level=int(lv[0].argmax()),
                         frame=self.frames[int(fr[0].argmax())])


def quantize(src: Path, dst: Path) -> None:
    """fp32 ONNX 를 int8 로 줄인다(크기 1/4, CPU 에서 더 빠름)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
