"""
얼굴 검출(YuNet) + crop + 분류기 입력 전처리.

인형(vision.py)과 학습 스크립트(train/)가 '같은 함수'를 쓴다.
학습 때 자른 얼굴과 실제로 카메라에서 자른 얼굴이 조금이라도 다르게 생기면
(여백, 정사각형 여부, 리사이즈 방식) 테스트 정확도만 높고 현장에서는 떨어진다.
그래서 이 파일은 picamera2 / hsemotion 같은 파이 전용 의존성 없이 cv2+numpy만 쓴다.
"""
import cv2
import numpy as np

# HSEmotion 과 직접 학습한 모델 모두 ImageNet 정규화 + 224 입력.
INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 얼굴 박스 대비 crop 반폭 비율. 0.65 → 박스 긴 변의 1.3배 정사각형.
CROP_HALF_RATIO = 0.65


def create_detector(model_path, score_threshold=0.5, nms_threshold=0.3):
    return cv2.FaceDetectorYN.create(
        model_path, "", (320, 320), score_threshold, nms_threshold, 5000,
    )


def detect_largest_face(detector, frame_bgr):
    """가장 큰 얼굴 1개의 (x, y, w, h, score). 없으면 None."""
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)
    if faces is None or len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, fw, fh = faces[0][:4]
    return int(x), int(y), int(fw), int(fh), float(faces[0][14])


def crop_square(frame_bgr, det):
    """정사각형으로 확장 + 여백 → 비율 왜곡 없이 crop. 비면 None."""
    x, y, fw, fh = det[:4]
    cx, cy = x + fw / 2.0, y + fh / 2.0
    half = int(max(fw, fh) * CROP_HALF_RATIO)
    H, W = frame_bgr.shape[:2]
    x1, y1 = max(0, int(cx - half)), max(0, int(cy - half))
    x2, y2 = min(W, int(cx + half)), min(H, int(cy + half))
    face = frame_bgr[y1:y2, x1:x2]
    return face if face.size else None


def to_input(face_rgb):
    """RGB 얼굴 → (3, 224, 224) float32. HSEmotion 전처리와 동일."""
    x = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)
