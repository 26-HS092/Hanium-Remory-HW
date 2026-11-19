"""
같은 사진, 같은 5감정 기준으로 모델들을 비교한다.

  python evaluate.py --data data/crops/test
  python evaluate.py --data ~/elder_faces --models ../../models/emotion_ko5.onnx

--models 를 안 주면 지금 인형 모델(HSEmotion)과 새로 학습한 모델을 둘 다 잰다.
HSEmotion 처럼 8개를 내는 모델은 labels.HSEMOTION_TO_TARGET 으로 5감정에 합쳐서 채점한다.

--data 는 감정 이름 폴더 아래 얼굴 사진이 있는 구조면 된다(AI Hub 7감정 이름도, 5감정 이름도 됨).
이미 crop 된 사진이라고 가정한다. 원본 사진이면 --detect 를 붙이면 인형과 똑같이 잘라서 잰다.
"""
import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from labels import (HSEMOTION_LABELS, HSEMOTION_TO_TARGET, MODELS_DIR, TARGET_LABELS,
                    list_samples, print_confusion)
import face_crop


def hsemotion_path():
    """HSEmotion ONNX. models/ 에 없으면 인형이 처음 켤 때 받아 두는 ~/.hsemotion/ 을 쓴다."""
    p = os.path.join(MODELS_DIR, "enet_b0_8_best_afew.onnx")
    home = os.path.expanduser("~/.hsemotion/enet_b0_8_best_afew.onnx")
    return p if os.path.exists(p) or not os.path.exists(home) else home


class OnnxModel:
    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map
        n_out = self.sess.get_outputs()[0].shape[-1]
        if "labels" in meta:
            labels = meta["labels"].split(",")
        elif n_out == len(HSEMOTION_LABELS):
            labels = HSEMOTION_LABELS
        else:
            labels = TARGET_LABELS
        # 모델 출력 인덱스 → 5감정 인덱스
        self.to_target = np.array([TARGET_LABELS.index(HSEMOTION_TO_TARGET.get(l, l)) for l in labels])

    def predict(self, batch):
        logits = self.sess.run(None, {self.input: batch})[0]
        return self.to_target[logits.argmax(1)]


def iter_batches(samples, detect, batch):
    """(x, y) 배치를 차례로. 테스트셋이 수만 장이라 한꺼번에 올리지 않는다."""
    detector = face_crop.create_detector(os.path.join(MODELS_DIR, "yunet.onnx")) if detect else None
    xs, ys, dropped = [], [], 0
    for path, y in samples:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None and detector is not None:
            det = face_crop.detect_largest_face(detector, img)
            img = face_crop.crop_square(img, det) if det else None
        if img is None:
            dropped += 1
            continue
        xs.append(face_crop.to_input(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        ys.append(y)
        if len(xs) == batch:
            yield np.stack(xs), np.array(ys)
            xs, ys = [], []
    if xs:
        yield np.stack(xs), np.array(ys)
    if dropped:
        print(f"  읽기/얼굴 검출 실패로 제외: {dropped}장")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crops/test")
    ap.add_argument("--models", nargs="*", default=[
        hsemotion_path(),
        os.path.join(MODELS_DIR, "emotion_ko5.onnx"),
    ])
    ap.add_argument("--detect", action="store_true", help="원본 사진이면 YuNet 으로 잘라서 평가")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    samples = list_samples(args.data)
    if not samples:
        raise SystemExit(f"{args.data} 에 감정 폴더/사진이 없습니다.")
    models = {}
    for path in args.models:
        if os.path.exists(path):
            models[os.path.basename(path)] = OnnxModel(path)
        else:
            print(f"(없음, 건너뜀) {path}")

    ys, preds = [], {name: [] for name in models}
    for x, y in tqdm(iter_batches(samples, args.detect, args.batch),
                     total=-(-len(samples) // args.batch)):
        ys.append(y)
        for name, model in models.items():
            preds[name].append(model.predict(x))
    y = np.concatenate(ys)
    print(f"평가 사진 {len(y)}장:", dict(zip(TARGET_LABELS, np.bincount(y, minlength=5).tolist())))

    for name, pred in preds.items():
        pred = np.concatenate(pred)
        print(f"\n━━ {name} ━━")
        print(f"  accuracy={accuracy_score(y, pred):.3f}  macro-F1={f1_score(y, pred, average='macro'):.3f}")
        print(classification_report(y, pred, labels=range(5), target_names=TARGET_LABELS,
                                    digits=3, zero_division=0))
        print_confusion(confusion_matrix(y, pred, labels=range(5)))


if __name__ == "__main__":
    main()
