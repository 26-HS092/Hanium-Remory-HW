"""
wakeword.py
-----------
"모리야" 웨이크워드 감지 모듈.

흐름:
    마이크 입력 → openWakeWord embedding → moriya_v0.onnx → score
"""

import os
import time
import queue
import urllib.request
import inspect
from pathlib import Path
from collections import deque

import numpy as np
import sounddevice as sd
import onnxruntime as ort


SAMPLE_RATE = 16000
TARGET_SECONDS = 2.1
BUFFER_SAMPLES = int(SAMPLE_RATE * TARGET_SECONDS)

INFER_INTERVAL_SEC = 0.25
DETECTION_COOLDOWN_SEC = 2.0

# 켜면 1초마다 그동안의 최고 점수와 마이크 음량(RMS)을 찍는다. 안 깨어날 때 원인 찾기용.
WAKE_DEBUG = bool(os.getenv("WAKE_DEBUG"))


def load_audio_features_class():
    try:
        from openwakeword.utils import AudioFeatures
        return AudioFeatures
    except Exception:
        pass

    try:
        from openwakeword.audio_features import AudioFeatures
        return AudioFeatures
    except Exception as e:
        raise ImportError(
            "AudioFeatures를 import하지 못했습니다. "
            "openwakeword 설치 상태를 확인하세요."
        ) from e


def download_file(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        return

    print(f"다운로드 중: {out_path.name}")
    urllib.request.urlretrieve(url, out_path)
    print(f"다운로드 완료: {out_path}")


def ensure_openwakeword_models(models_dir: Path):
    melspec_path = models_dir / "melspectrogram.onnx"
    embedding_path = models_dir / "embedding_model.onnx"

    base_url = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

    download_file(
        f"{base_url}/melspectrogram.onnx",
        melspec_path
    )

    download_file(
        f"{base_url}/embedding_model.onnx",
        embedding_path
    )

    return melspec_path, embedding_path


def create_audio_features(
    melspec_model_path: Path,
    embedding_model_path: Path,
):
    AudioFeatures = load_audio_features_class()

    print("AudioFeatures:", AudioFeatures)

    try:
        print("AudioFeatures signature:", inspect.signature(AudioFeatures))
    except Exception:
        pass

    # (melspec_model_path='', embedding_model_path='', sr=16000, ncpu=1, ...)
    try:
        return AudioFeatures(
            melspec_model_path=str(melspec_model_path),
            embedding_model_path=str(embedding_model_path),
            sr=SAMPLE_RATE,
            ncpu=1,
        )
    except TypeError as e:
        print("AudioFeatures init 방식 1 실패:", e)

    # Colab에서 쓰던 형태
    try:
        return AudioFeatures(
            melspec_onnx_model_path=str(melspec_model_path),
            embedding_onnx_model_path=str(embedding_model_path),
            sr=SAMPLE_RATE,
            ncpu=1,
        )
    except TypeError as e:
        print("AudioFeatures init 방식 2 실패:", e)

    # 최후 fallback
    try:
        print("경고: AudioFeatures() 기본 생성자로 시도합니다.")
        return AudioFeatures()
    except Exception as e:
        raise RuntimeError("AudioFeatures 초기화 실패") from e


class MoriyaWakeWordDetector:
    def __init__(
        self,
        moriya_model_path: Path,
        models_dir: Path,
        threshold: float = 0.90,
        input_device_index: int | None = None,
    ):
        self.moriya_model_path = Path(moriya_model_path)
        self.models_dir = Path(models_dir)
        self.threshold = threshold
        self.input_device_index = input_device_index

        if not self.moriya_model_path.exists():
            raise FileNotFoundError(f"moriya_v0.onnx 없음: {self.moriya_model_path}")

        data_path = Path(str(self.moriya_model_path) + ".data")
        if not data_path.exists():
            print("주의: moriya_v0.onnx.data 파일이 없을 수 있습니다.")
            print("ONNX가 외부 data 파일을 요구하면 로드 에러가 납니다.")
            print("확인 경로:", data_path)

        melspec_path, embedding_path = ensure_openwakeword_models(self.models_dir)

        self.feature_extractor = create_audio_features(
            melspec_model_path=melspec_path,
            embedding_model_path=embedding_path,
        )

        self.session = ort.InferenceSession(
            str(self.moriya_model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("\n✅ 웨이크워드 모델 로드 완료")
        print("input:", self.session.get_inputs()[0].shape, self.input_name)
        print("output:", self.session.get_outputs()[0].shape, self.output_name)
        print("threshold:", self.threshold)

    def _get_embeddings(self, audio_int16: np.ndarray) -> np.ndarray:
        audio_int16 = np.asarray(audio_int16, dtype=np.int16).reshape(-1)

        if hasattr(self.feature_extractor, "_get_embeddings"):
            return self.feature_extractor._get_embeddings(audio_int16)

        if hasattr(self.feature_extractor, "get_embeddings"):
            return self.feature_extractor.get_embeddings(audio_int16)

        if hasattr(self.feature_extractor, "embed_clips"):
            return self.feature_extractor.embed_clips(audio_int16)

        raise AttributeError("openWakeWord embedding 추출 함수를 찾지 못했습니다.")

    def _audio_to_feature(self, audio_int16: np.ndarray) -> np.ndarray:
        emb = self._get_embeddings(audio_int16)
        emb = np.asarray(emb, dtype=np.float32)

        if emb.ndim == 3:
            emb = emb[0]

        if emb.ndim != 2:
            raise ValueError(f"Unexpected embedding shape: {emb.shape}")

        if emb.shape[0] >= 16:
            emb = emb[-16:, :]
        else:
            pad = np.zeros((16 - emb.shape[0], emb.shape[1]), dtype=np.float32)
            emb = np.concatenate([pad, emb], axis=0)

        if emb.shape != (16, 96):
            raise ValueError(f"Expected feature shape (16, 96), got {emb.shape}")

        return emb[np.newaxis, :, :].astype(np.float32)

    def predict_score(self, audio_int16: np.ndarray) -> float:
        x = self._audio_to_feature(audio_int16)

        logits = self.session.run(
            [self.output_name],
            {self.input_name: x},
        )[0]

        #logit = float(logits.reshape(-1)[0])
        #score = 1.0 / (1.0 + np.exp(-logit))

        #return float(score)

        logit = float(logits.reshape(-1)[0])
        score = 1.0 / (1.0 + np.exp(-logit))

        return float(score)

    def wait_for_wakeword(self, stop_event=None) -> bool:
        """
        "모리야"가 감지될 때까지 마이크를 듣습니다.
        감지되면 True를 반환합니다.

        stop_event 가 켜지면 마이크를 놓고 False 를 돌려줍니다. 다른 일(가족
        메시지 답장 녹음 등)이 마이크를 써야 할 때 기다림을 끊는 데 씁니다.
        """

        audio_queue = queue.Queue()
        audio_buffer = deque(maxlen=BUFFER_SAMPLES)

        last_infer_time = 0.0
        last_detection_time = 0.0
        debug_t = time.time()
        debug_max = 0.0

        def audio_callback(indata, frames, time_info, status):
            if status:
                print("마이크 상태:", status)

            mono = indata[:, 0].copy()
            audio_queue.put(mono)

        print("\n👂 웨이크워드 대기 중... ('모리야'라고 말하세요)")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1280,
            callback=audio_callback,
            device=self.input_device_index,
        ):
            while True:
                if stop_event is not None and stop_event.is_set():
                    return False

                try:
                    chunk = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                audio_buffer.extend(chunk.tolist())

                if len(audio_buffer) < BUFFER_SAMPLES:
                    continue

                now = time.time()

                if now - last_infer_time < INFER_INTERVAL_SEC:
                    continue

                last_infer_time = now

                audio_np = np.array(audio_buffer, dtype=np.int16)

                try:
                    score = self.predict_score(audio_np)
                except Exception as e:
                    print("웨이크워드 추론 오류:", repr(e))
                    continue

                if WAKE_DEBUG:
                    debug_max = max(debug_max, score)
                    if now - debug_t >= 1.0:
                        rms = float(np.sqrt(np.mean(audio_np[-SAMPLE_RATE:].astype(np.float32) ** 2)))
                        print(f"wake max={debug_max:.3f} rms={rms:.0f}", flush=True)
                        debug_t, debug_max = now, 0.0

                if score >= self.threshold:
                    if now - last_detection_time >= DETECTION_COOLDOWN_SEC:
                        print(f"\n✅ 모리야 감지됨! score={score:.3f}\n")
                        last_detection_time = now
                        return True

