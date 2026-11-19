"""
라벨 정의. 매핑을 바꾸고 싶으면 여기만 고치고 train.py 를 다시 돌리면 된다
(prepare_aihub.py 는 원래 7감정 이름으로 폴더를 나눠 두므로 다시 자를 필요 없음).
"""
import os
import sys
import unicodedata

# emotion/face_crop.py 를 학습 스크립트에서도 그대로 쓰기 위해.
EMOTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, EMOTION_DIR)
MODELS_DIR = os.path.abspath(os.path.join(EMOTION_DIR, "..", "models"))

# 인형이 내보낼 5감정. 백엔드 emotion_codes.py 가 모두 알아듣는 한글 라벨이다
# (기쁨→happy, 슬픔→sad, 분노→angry, 불안→anxious, 중립→calm).
TARGET_LABELS = ["기쁨", "슬픔", "분노", "불안", "중립"]

# AI Hub '한국인 감정인식을 위한 복합 영상' 7감정 → 5감정.
# 상처는 슬픔 쪽 표정이고, 당황은 눈이 커지고 굳는 표정이라 불안에 가깝다.
AIHUB_LABELS = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]
AIHUB_TO_TARGET = {
    "기쁨": "기쁨",
    "당황": "불안",
    "분노": "분노",
    "불안": "불안",
    "상처": "슬픔",
    "슬픔": "슬픔",
    "중립": "중립",
}

# 지금 인형에 들어간 HSEmotion 8감정 → 5감정. 기준선(학습 전) 성능을 같은 잣대로 재기 위함.
# 백엔드가 못마땅함·불쾌를 angry, 두려움·놀람을 anxious 로 모으는 것과 같은 규칙.
HSEMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                    "Happiness", "Neutral", "Sadness", "Surprise"]
HSEMOTION_TO_TARGET = {
    "Anger": "분노", "Contempt": "분노", "Disgust": "분노", "Fear": "불안",
    "Happiness": "기쁨", "Neutral": "중립", "Sadness": "슬픔", "Surprise": "불안",
}


def folder_to_target(name):
    """crop 폴더 이름(AI Hub 7감정 또는 5감정 그대로) → 5감정 라벨. 모르면 None."""
    if name in TARGET_LABELS:
        return name
    return AIHUB_TO_TARGET.get(name)


# 아래 둘은 학습(train.py)과 평가(evaluate.py)가 같이 쓴다. 여기 두는 건 evaluate.py 가
# torch 없이(파이에서도) 돌게 하려고 — train.py 는 import 만 해도 torch 가 필요하다.
def list_samples(split_dir):
    """crop 폴더 → [(경로, 5감정 인덱스)]"""
    samples = []
    for name in sorted(os.listdir(split_dir)):
        target = folder_to_target(unicodedata.normalize("NFC", name))
        d = os.path.join(split_dir, name)
        if target is None or not os.path.isdir(d):
            continue
        idx = TARGET_LABELS.index(target)
        samples += [(os.path.join(d, f), idx) for f in sorted(os.listdir(d))
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return samples


def print_confusion(cm):
    w = max(max(len(l) for l in TARGET_LABELS), len(str(cm.max()))) + 2
    print("  정답\\예측".ljust(w + 2) + "".join(l.rjust(w) for l in TARGET_LABELS))
    for l, row in zip(TARGET_LABELS, cm):
        print(f"  {l}".ljust(w + 2) + "".join(str(v).rjust(w) for v in row))
