import os
import time

import requests
import soundfile as sf

# NAVER Cloud Platform — CLOVA Speech Recognition(CSR, 짧은 문장 인식).
# 녹음된 WAV 바이트를 그대로 보내면 {"text": "..."} 로 돌려준다. (최대 60초)
CSR_URL = os.getenv(
    "CLOVA_CSR_URL", "https://naveropenapi.apigw.ntruss.com/recog/v1/stt"
)
# 이보다 짧은 클립은 의미 있는 발화가 아닐 확률이 높아 API 호출 자체를 건너뛴다.
MIN_AUDIO_SEC = 0.4
MAX_AUDIO_SEC = 60.0


class ClovaSTTHandler:
    """stt4vad_hat.STTHandler 와 같은 인터페이스: transcribe(path) -> (text, elapsed)."""

    def __init__(self, lang="Kor", timeout=10.0):
        self.client_id = os.environ["CLOVA_CLIENT_ID"]
        self.client_secret = os.environ["CLOVA_CLIENT_SECRET"]
        self.lang = lang
        self.timeout = timeout
        self.session = requests.Session()
        print("STT(CLOVA) 준비 완료")

    def transcribe(self, audio_path):
        start_time = time.time()

        info = sf.info(audio_path)
        if info.duration < MIN_AUDIO_SEC:
            print(f"⚠️  STT: 너무 짧은 발화({info.duration:.2f}s) — 건너뜀")
            return "", time.time() - start_time
        if info.duration > MAX_AUDIO_SEC:
            print(f"⚠️  STT: 60초 초과({info.duration:.1f}s) — CSR 한도라 잘릴 수 있음")

        with open(audio_path, "rb") as f:
            data = f.read()

        try:
            res = self.session.post(
                CSR_URL,
                params={"lang": self.lang},
                headers={
                    "X-NCP-APIGW-API-KEY-ID": self.client_id,
                    "X-NCP-APIGW-API-KEY": self.client_secret,
                    "Content-Type": "application/octet-stream",
                },
                data=data,
                timeout=self.timeout,
            )
            res.raise_for_status()
            text = res.json().get("text", "").strip()
        except requests.RequestException as e:
            # 네트워크/인증 오류로 대화 루프가 죽지 않도록 빈 문자열로 처리한다.
            body = getattr(getattr(e, "response", None), "text", "")
            print(f"⚠️  STT(CLOVA) 실패: {e} {body[:200]}")
            text = ""

        return text, time.time() - start_time
