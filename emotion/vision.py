"""
비전 파이프라인: 카메라 → 얼굴 검출(YuNet) → crop → 감정 분류.

- 얼굴 검출: cv2.FaceDetectorYN (YuNet). 가볍고 각도/측면에 비교적 강함.
- 감정 분류: models/ 에 직접 학습한 모델(config.CUSTOM_EMOTION_PATH)이 있으면 그것,
  없으면 HSEmotion(AffectNet 8감정 EfficientNet-B0).
  ※ 둘 다 RGB 입력, 224 리사이즈 + ImageNet 정규화. crop/전처리는 face_crop.py 공용.
원본 이미지는 저장하지 않고 메모리에서만 처리합니다.
"""
import os
import threading

import cv2
import numpy as np

import config
import face_crop


class Camera:
    """picamera2 래퍼. 한 장씩 빠르게 캡처.

    감정 샘플링과 얼굴 트래킹(face_tracker.py)이 카메라 하나를 같이 쓴다.
    두 스레드가 동시에 capture_array 를 부르지 않도록 락으로 줄을 세운다.
    """

    def __init__(self, size=config.CAM_SIZE):
        from picamera2 import Picamera2  # 파이 전용. import는 여기서.
        from libcamera import Transform
        self._picam = Picamera2()
        cfg = self._picam.create_preview_configuration(
            main={"size": size, "format": "RGB888"},
            transform=Transform(hflip=config.CAM_HFLIP, vflip=config.CAM_VFLIP),
        )
        self._picam.configure(cfg)
        self._picam.start()
        self._lock = threading.Lock()

    def capture(self):
        """현재 프레임을 BGR ndarray로 반환."""
        # picamera2의 "RGB888"은 실제로 BGR 순서로 배열을 반환 → OpenCV에 그대로 사용.
        with self._lock:
            return self._picam.capture_array()

    def close(self):
        try:
            self._picam.stop()
        except Exception:
            pass


class _HSEmotionClassifier:
    """HSEmotion(AffectNet 8감정). 모델은 ~/.hsemotion/ 에서 로드."""

    def __init__(self):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        self._fer = HSEmotionRecognizer(model_name=config.HSEMOTION_MODEL)

    def __call__(self, face_rgb):
        label, scores = self._fer.predict_emotions(face_rgb, logits=False)
        return label, float(np.max(scores))


class _CustomClassifier:
    """emotion/train/ 으로 직접 학습해 내보낸 ONNX. 라벨은 모델 메타데이터에 들어 있다."""

    def __init__(self, path):
        import onnxruntime as ort
        self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        meta = self._sess.get_modelmeta().custom_metadata_map
        self.labels = meta["labels"].split(",") if "labels" in meta else config.CUSTOM_EMOTION_LABELS

    def __call__(self, face_rgb):
        x = face_crop.to_input(face_rgb)[None]
        probs = face_crop.softmax(self._sess.run(None, {self._input: x})[0][0])
        i = int(np.argmax(probs))
        return self.labels[i], float(probs[i])


class EmotionAnalyzer:
    """YuNet으로 얼굴을 찾고 감정을 분류."""

    def __init__(self):
        self._detector = face_crop.create_detector(
            config.YUNET_PATH, config.FACE_SCORE_THRESHOLD, config.FACE_NMS_THRESHOLD,
        )
        if os.path.exists(config.CUSTOM_EMOTION_PATH):
            self._fer = _CustomClassifier(config.CUSTOM_EMOTION_PATH)
            print(f"🙂 감정 모델: 직접 학습 ({'/'.join(self._fer.labels)})")
        else:
            self._fer = _HSEmotionClassifier()
            print("🙂 감정 모델: HSEmotion (AffectNet 8감정)")

    def analyze_frame(self, frame_bgr):
        """프레임 1장 → (label, confidence) 또는 None(얼굴 없음)."""
        det = face_crop.detect_largest_face(self._detector, frame_bgr)
        if det is None:
            return None
        face = face_crop.crop_square(frame_bgr, det)
        if face is None:
            return None
        return self._fer(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    def best_emotion(self, frames):
        """
        발화 중 캡처한 여러 프레임 → 대표 감정 1개 (신뢰도 가중 다수결).
        반환: (label, label_ko, confidence) 또는 None.
        """
        votes = {}
        for f in frames:
            res = self.analyze_frame(f)
            if res is None:
                continue
            label, conf = res
            votes[label] = votes.get(label, 0.0) + conf
        if not votes:
            return None
        label = max(votes, key=votes.get)
        total = sum(votes.values())
        conf = votes[label] / total if total else 0.0
        return label, config.EMOTION_KO.get(label, label), conf
