"""
감정 서비스 — 마이크 VAD의 '발화 시작/끝' 이벤트에 걸어 쓰는 얇은 래퍼.

같은 파이 한 대, 같은 프로세스 안에서 동작.
오디오/STT 담당이 발화 시작 시 on_speech_start(), 끝났을 때 on_speech_end()를
호출해주면, 그 사이 동안 얼굴을 샘플링해 대표 감정 라벨을 돌려줍니다.

  audio/STT 쪽          emotion 쪽(이 모듈)
  ───────────          ──────────────────
  VAD: 발화 시작  ──▶  on_speech_start()      # 백그라운드로 프레임 샘플 시작
       (녹음/STT 진행)
  VAD: 발화 끝    ──▶  on_speech_end() → 라벨  # 샘플 종료, FER 다수결, 라벨 반환

카메라는 이 모듈만 엽니다. (마이크는 오디오 담당만 — 디바이스 중복 오픈 금지)
"""
import threading
import time

import config
from vision import Camera, EmotionAnalyzer


class EmotionService:
    """
    발화 구간 동안 얼굴을 일정 간격으로 샘플링하고, '그 자리에서 바로' FER를 돌려
    감정 투표를 누적한다(프레임을 모았다가 끝에 한꺼번에 분석하지 않음).

    이렇게 하면:
      - 발화 길이에 맞춰 샘플 수 n이 자동으로 늘어남(짧으면 적게, 길면 많이).
      - on_speech_end()가 즉시 반환됨 → 턴 끝 지연 0.
      - 원본 프레임을 보관하지 않음(분석 직후 폐기) → 메모리/프라이버시 이점.
    """

    def __init__(self, sample_interval=0.7, max_samples=30):
        self._cam = Camera()
        self._fer = EmotionAnalyzer()
        self._interval = sample_interval   # 프레임 샘플 간격(초)
        self._max_samples = max_samples    # 안전 상한(아주 긴 발화 대비)
        self._votes = {}                   # {label: 누적 신뢰도}
        self._n = 0                        # 얼굴을 실제로 잡은 샘플 수
        self._lock = threading.Lock()
        self._stop = threading.Event()     # set → 샘플 루프 종료(중단 가능한 sleep용)
        self._thread = None

    @property
    def camera(self):
        """얼굴 트래킹(face_tracker.py)이 같은 카메라를 쓰도록 내어준다."""
        return self._cam

    # ── 발화 시작: 프레임 샘플링 백그라운드 시작 ──────────
    def on_speech_start(self):
        with self._lock:
            self._votes = {}
            self._n = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        # 발화 동안 간격을 두고 캡처 → 즉시 분석 → 투표 누적.
        while not self._stop.is_set() and self._n < self._max_samples:
            res = self._fer.analyze_frame(self._cam.capture())   # (label, conf) or None
            if res is not None:
                label, conf = res
                with self._lock:
                    self._votes[label] = self._votes.get(label, 0.0) + conf
                    self._n += 1
            # 중단 가능한 sleep: 종료 신호가 오면 즉시 깨어남.
            self._stop.wait(self._interval)

    # ── 발화 끝: 샘플 종료 → 누적 투표 다수결 → 라벨 반환 ──────
    def on_speech_end(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

        with self._lock:
            votes = dict(self._votes)
            n = self._n

        # 발화가 너무 짧아 한 장도 못 잡았으면 마지막으로 한 장만 시도.
        if not votes:
            res = self._fer.analyze_frame(self._cam.capture())
            if res is not None:
                votes[res[0]] = res[1]
                n = 1

        if not votes:
            return {"label": "unknown", "label_ko": "알수없음",
                    "confidence": 0.0, "n": 0}

        label = max(votes, key=votes.get)
        total = sum(votes.values())
        conf = votes[label] / total if total else 0.0
        return {"label": label, "label_ko": config.EMOTION_KO.get(label, label),
                "confidence": conf, "n": n}

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cam.close()


# ── 오디오/STT 담당이 이렇게 호출하면 됨 (예시) ───────────
if __name__ == "__main__":
    svc = EmotionService()
    try:
        # 실제로는 마이크 VAD 콜백 안에서 아래 두 줄이 각각 호출됨
        svc.on_speech_start()          # ← VAD: 발화 시작
        time.sleep(3)                  #   (이 자리에 STT 녹음/인식이 진행)
        result = svc.on_speech_end()   # ← VAD: 발화 끝
        print(result)                  # {'label': 'sadness', 'label_ko': '슬픔', ...}
        # 이제 result['label_ko'] + STT텍스트 를 LLM 담당에게 넘기면 끝
    finally:
        svc.close()
