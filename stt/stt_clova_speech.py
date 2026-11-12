"""
stt_clova_speech.py
-------------------
NAVER Cloud CLOVA Speech(장문 인식 API) — STT_ENGINE=clova_speech

CSR(stt_clova.py)보다 인식 모델이 좋고, 키워드 부스팅(boostings)으로
어르신·가족 이름처럼 잘 틀리는 말을 알려 줄 수 있다.

.env
  CLOVA_SPEECH_INVOKE_URL  콘솔 > CLOVA Speech > 도메인 > 빌더 > 'Invoke URL'
  CLOVA_SPEECH_SECRET      같은 화면의 'Secret Key'
  STT_BOOST_WORDS          (선택) 부스팅에 더할 말, 쉼표로 구분
  STT_NORMALIZE=0          (선택) 보내기 전 소리 크기 맞추기를 끈다
"""

import json
import os
import time

import requests

from audio_prep import prepared_wav_bytes

MIN_AUDIO_SEC = 0.4


class ClovaSpeechSTTHandler:
    """stt4vad_hat.STTHandler 와 같은 인터페이스: transcribe(path) -> (text, elapsed)."""

    def __init__(self, timeout=15.0, normalize=None, keywords=None):
        if normalize is None:
            normalize = os.getenv("STT_NORMALIZE", "1") != "0"
        self.normalize = normalize
        self.url = os.environ["CLOVA_SPEECH_INVOKE_URL"].rstrip("/") + "/recognizer/upload"
        self.secret = os.environ["CLOVA_SPEECH_SECRET"]
        self.timeout = timeout
        self.session = requests.Session()
        # 부스팅 단어: 목록이나, 매번 새로 읽을 함수(가족 정보가 동기화로 바뀌므로).
        self._keywords = keywords
        print("STT(CLOVA Speech) 준비 완료")

    def set_keywords(self, keywords) -> None:
        """list[str] 또는 list[str] 을 돌려주는 함수."""
        self._keywords = keywords

    def _boost_words(self) -> list[str]:
        kw = self._keywords
        try:
            words = kw() if callable(kw) else (kw or [])
        except Exception as e:
            print(f"⚠️  STT 부스팅 단어 읽기 실패: {e}")
            words = []
        return [w for w in words if w]

    def transcribe(self, audio_path):
        start_time = time.time()
        data, duration, gain_db = prepared_wav_bytes(audio_path, self.normalize)
        if duration < MIN_AUDIO_SEC:
            print(f"⚠️  STT: 너무 짧은 발화({duration:.2f}s) — 건너뜀")
            return "", time.time() - start_time
        if gain_db >= 3:
            print(f"🔊 STT: 작은 목소리 → {gain_db:.0f}dB 키워 보냄")

        params = {
            "language": "ko-KR",
            "completion": "sync",
            "wordAlignment": False,
            "fullText": True,
            "diarization": {"enable": False},
        }
        words = self._boost_words()
        if words:
            params["boostings"] = [{"words": ",".join(words)}]

        try:
            res = self.session.post(
                self.url,
                headers={"X-CLOVASPEECH-API-KEY": self.secret,
                         "Accept": "application/json"},
                files={
                    "media": ("speech.wav", data, "audio/wav"),
                    "params": (None, json.dumps(params, ensure_ascii=False),
                               "application/json"),
                },
                timeout=self.timeout,
            )
            res.raise_for_status()
            body = res.json()
            if body.get("result") not in (None, "COMPLETED"):
                print(f"⚠️  STT(CLOVA Speech) 실패: {body.get('result')} {body.get('message')}")
                text = ""
            else:
                text = (body.get("text") or "").strip()
        except (requests.RequestException, ValueError) as e:
            # 네트워크/인증 오류로 대화 루프가 죽지 않도록 빈 문자열로 처리한다.
            resp = getattr(e, "response", None)
            print(f"⚠️  STT(CLOVA Speech) 실패: {e} {getattr(resp, 'text', '')[:200]}")
            text = ""

        return text, time.time() - start_time


if __name__ == "__main__":
    # 파이에서 연결·부스팅 확인: python stt/stt_clova_speech.py input.wav [더 많은 wav...]
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    from stt_keywords import boost_words

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    paths = sys.argv[1:] or [str(Path(__file__).resolve().parent.parent / "input.wav")]
    words = boost_words(os.getenv("PATIENT_ID", "P001"))
    print(f"부스팅 단어: {', '.join(words)}")
    plain = ClovaSpeechSTTHandler(normalize=False, keywords=[])
    full = ClovaSpeechSTTHandler(normalize=True, keywords=words)
    for p in paths:
        print(f"\n▶ {p}")
        for name, h in (("기본           ", plain), ("부스팅+크기맞춤", full)):
            text, sec = h.transcribe(p)
            print(f"  {name}: {text!r}  ({sec:.2f}s)")
