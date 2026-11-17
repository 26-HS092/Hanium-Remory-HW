"""
stt_groq.py
-----------
Groq Whisper(large-v3) — STT_ENGINE=groq. 주로 stt_compare.py 에서 CLOVA 와
견주어 보려고 둔다. 키워드는 prompt 힌트로 넘긴다.

.env
  GROQ_API_KEY
  GROQ_STT_MODEL  (선택) 기본 whisper-large-v3
"""

import os
import time

from groq import Groq

from audio_prep import prepared_wav_bytes

MIN_AUDIO_SEC = 0.4


class GroqSTTHandler:
    def __init__(self, normalize=None, keywords=None, model=None):
        if normalize is None:
            normalize = os.getenv("STT_NORMALIZE", "1") != "0"
        self.normalize = normalize
        self.model = model or os.getenv("GROQ_STT_MODEL", "whisper-large-v3")
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self._keywords = keywords
        print(f"STT(Groq {self.model}) 준비 완료")

    def set_keywords(self, keywords) -> None:
        self._keywords = keywords

    def transcribe(self, audio_path):
        start_time = time.time()
        data, duration, _ = prepared_wav_bytes(audio_path, self.normalize)
        if duration < MIN_AUDIO_SEC:
            return "", time.time() - start_time
        kw = self._keywords
        words = (kw() if callable(kw) else kw) or []
        # Whisper 는 prompt 를 '앞서 나온 글'로 보고 그 말투·단어를 따른다.
        prompt = ("어르신과 인형 모리의 대화입니다. " + ", ".join(words))[:400] if words else None
        try:
            res = self.client.audio.transcriptions.create(
                file=("speech.wav", data),
                model=self.model,
                language="ko",
                temperature=0,
                prompt=prompt,
            )
            text = (res.text or "").strip()
        except Exception as e:
            print(f"⚠️  STT(Groq) 실패: {e}")
            text = ""
        return text, time.time() - start_time
