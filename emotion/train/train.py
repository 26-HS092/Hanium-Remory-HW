"""
5감정 얼굴 분류기 파인튜닝 → ONNX 내보내기.

지금 인형에 들어간 HSEmotion(enet_b0_8_best_afew, AffectNet 학습) 가중치에서 출발해
마지막 층만 5감정으로 바꾸고 한국인 얼굴(AI Hub)로 다시 학습한다. 처음부터 학습하는 것보다
훨씬 적은 데이터·시간으로 수렴하고, 구조가 같아 파이에서의 속도도 지금과 같다.

  python train.py --data data/crops
  python train.py --data data/crops --epochs 3 --limit 2000   # 빠르게 한 바퀴 확인

결과:
  ../../models/emotion_ko5_best.pt   val macro-F1 이 가장 좋았던 가중치
  ../../models/emotion_ko5.onnx      인형용. 이 파일을 파이 models/ 에 넣으면 vision.py 가 자동으로 쓴다
"""
import argparse
import os
import random
import time
import unicodedata

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

import timm
from tqdm import tqdm

from labels import MODELS_DIR, TARGET_LABELS, folder_to_target, list_samples, print_confusion
import face_crop

HSEMOTION_PT = os.path.join(MODELS_DIR, "enet_b0_8_best_afew.pt")
HSEMOTION_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/"
                 "models/affectnet_emotions/enet_b0_8_best_afew.pt")


class FaceDataset(Dataset):
    def __init__(self, samples, augment=None):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.augment is not None:
            # 증강은 uint8 이미지에서 하고, 마지막 리사이즈·정규화는 인형과 같은 함수로.
            rgb = self.augment(torch.from_numpy(rgb).permute(2, 0, 1)).permute(1, 2, 0).numpy()
        return torch.from_numpy(face_crop.to_input(rgb)), y


def build_augment():
    # 인형 앞 어르신: 조명이 어둡거나 한쪽에서 들어오고, 고개를 살짝 기울이거나
    # 얼굴이 정면에서 비껴 있고, 카메라 초점이 흐린 경우가 많다.
    return T.Compose([
        T.RandomResizedCrop(256, scale=(0.80, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(12)], p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.03),
        T.RandomGrayscale(p=0.1),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
    ])


def build_model(init, num_classes):
    model = timm.create_model("tf_efficientnet_b0", pretrained=(init == "imagenet"),
                              num_classes=num_classes)
    if init == "hsemotion":
        if not os.path.exists(HSEMOTION_PT):
            os.makedirs(MODELS_DIR, exist_ok=True)
            torch.hub.download_url_to_file(HSEMOTION_URL, HSEMOTION_PT)
        # 옛 timm 으로 pickle 된 모델이라 그대로는 forward 가 깨진다 → 가중치만 옮긴다.
        old = torch.load(HSEMOTION_PT, map_location="cpu", weights_only=False)
        sd = {k: v for k, v in old.state_dict().items() if not k.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert set(missing) == {"classifier.weight", "classifier.bias"} and not unexpected, \
            (missing, unexpected)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return accuracy_score(y, p), f1_score(y, p, average="macro"), confusion_matrix(
        y, p, labels=range(len(TARGET_LABELS)))


def export_onnx(model, path):
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, face_crop.INPUT_SIZE, face_crop.INPUT_SIZE)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    import onnx
    import onnxruntime as ort
    m = onnx.load(path)
    # 라벨 순서를 모델 안에 같이 넣어 둔다 → 인형 config 와 어긋날 일이 없다.
    m.metadata_props.add(key="labels", value=",".join(TARGET_LABELS))
    onnx.save(m, path)
    x = np.random.randn(2, 3, face_crop.INPUT_SIZE, face_crop.INPUT_SIZE).astype(np.float32)
    out = ort.InferenceSession(path).run(None, {"input": x})[0]
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    diff = float(np.abs(out - ref).max())
    assert diff < 1e-3, f"ONNX 출력이 PyTorch 와 다릅니다 (최대 차이 {diff})"
    print(f"ONNX 저장: {path} (PyTorch 와 최대 차이 {diff:.1e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crops")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze-epochs", type=int, default=1,
                    help="처음 몇 epoch 은 마지막 층만 학습(새 층이 무작위라 본체를 망가뜨리지 않게)")
    ap.add_argument("--init", choices=["hsemotion", "imagenet"], default="hsemotion")
    ap.add_argument("--limit", type=int, default=0, help="train 장수 제한(빠른 확인용)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="emotion_ko5")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    train_s = list_samples(os.path.join(args.data, "train"))
    val_s = list_samples(os.path.join(args.data, "val"))
    if args.limit:
        random.shuffle(train_s)
        train_s = train_s[:args.limit]
    if not train_s or not val_s:
        raise SystemExit(f"{args.data}/train, val 에 사진이 없습니다. prepare_aihub.py 를 먼저 돌리세요.")

    counts = np.bincount([y for _, y in train_s], minlength=len(TARGET_LABELS))
    print(f"device={device}  train={len(train_s)}  val={len(val_s)}")
    print("  train 분포:", dict(zip(TARGET_LABELS, counts.tolist())))
    # 5감정으로 합치면 불안(당황+불안)·슬픔(상처+슬픔)이 두 배가 된다 → 손실에 역빈도 가중.
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(counts), dtype=torch.float32)

    train_dl = DataLoader(FaceDataset(train_s, build_augment()), batch_size=args.batch,
                          shuffle=True, num_workers=args.workers, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(FaceDataset(val_s), batch_size=args.batch * 2,
                        num_workers=args.workers, persistent_workers=args.workers > 0)

    model = build_model(args.init, len(TARGET_LABELS)).to(device)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    # 본체는 이미 표정을 잘 아는 상태라 작은 lr, 새 층은 큰 lr.
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.lr * 10}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.lr * 10], total_steps=args.epochs * len(train_dl), pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    os.makedirs(MODELS_DIR, exist_ok=True)
    best_path = os.path.join(MODELS_DIR, f"{args.out}_best.pt")
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        for p in body:
            p.requires_grad_(not frozen)
        model.train()
        t0, total, n = time.time(), 0.0, 0
        bar = tqdm(train_dl, desc=f"[{epoch:2d}/{args.epochs}]", leave=False, dynamic_ncols=True)
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y); n += len(y)
            bar.set_postfix(loss=f"{total / n:.4f}")
        acc, f1, cm = evaluate(model, val_dl, device)
        mark = ""
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"state_dict": model.state_dict(), "labels": TARGET_LABELS,
                        "epoch": epoch, "val_f1": f1}, best_path)
            mark = "  ← best"
        print(f"[{epoch:2d}/{args.epochs}] {'(본체 고정) ' if frozen else ''}loss={total / n:.4f} "
              f"val acc={acc:.3f} macro-F1={f1:.3f}  {time.time() - t0:.0f}s{mark}")

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    acc, f1, cm = evaluate(model.to(device), val_dl, device)
    print(f"\nbest epoch {ckpt['epoch']}: val acc={acc:.3f} macro-F1={f1:.3f}")
    print_confusion(cm)
    export_onnx(model, os.path.join(MODELS_DIR, f"{args.out}.onnx"))
    print("\n최종 성능은 test 로: python evaluate.py --data data/crops/test")


if __name__ == "__main__":
    main()
