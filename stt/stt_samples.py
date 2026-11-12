"""
stt_samples.py
--------------
STT 비교용으로 매 턴 녹음을 남긴다. SAVE_STT_WAV=1 일 때만 켜진다.

  {STT_SAMPLE_DIR}/wav/{id}.wav   STT 에 넣은 녹음 그대로
  {STT_SAMPLE_DIR}/samples.jsonl  한 줄 = 한 사건
      {"kind": "stt",   "id", "time", "engine", "stt_text", "prev", "where"}
      {"kind": "heard", "id", "heard"}      LLM 이 바로잡은 말(고친 게 없으면 원문)

prev 는 바로 앞에 모리가 한 말이다(heard 바로잡기를 따로 재 볼 때 쓴다).
기본 위치는 저장소의 stt_samples/ — 어르신 목소리라 .gitignore 로 올리지 않는다.
stt_compare.py 가 이 폴더를 읽어 정답 적기(label)·엔진 비교(run)를 한다.
저장에 실패해도 대화는 이어 간다.
"""

import datetime
import json
import os
import shutil
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "stt_samples"


class SampleSaver:
    def __init__(self, base_dir, enabled: bool, engine: str = ""):
        self.base = Path(base_dir)
        self.enabled = enabled
        self.engine = engine
        self._lock = threading.Lock()
        if enabled:
            print(f"🎙️  STT 녹음 저장 켜짐 → {self.base}")

    @classmethod
    def from_env(cls) -> "SampleSaver":
        return cls(
            os.getenv("STT_SAMPLE_DIR", str(DEFAULT_DIR)),
            os.getenv("SAVE_STT_WAV", "0") == "1",
            os.getenv("STT_ENGINE", "hailo").lower(),
        )

    def _append(self, record: dict) -> None:
        with self._lock:
            self.base.mkdir(parents=True, exist_ok=True)
            with (self.base / "samples.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save(self, wav_path: str, stt_text: str, prev: str = "",
             where: str = "conversation") -> str | None:
        """녹음을 복사하고 기록을 남긴다. 샘플 id(꺼져 있거나 실패하면 None)."""
        if not self.enabled or not wav_path:
            return None
        now = datetime.datetime.now()
        sid = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        try:
            (self.base / "wav").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(wav_path, self.base / "wav" / f"{sid}.wav")
            self._append({
                "kind": "stt", "id": sid,
                "time": now.astimezone().isoformat(timespec="seconds"),
                "engine": self.engine, "stt_text": stt_text or "",
                "prev": prev or "", "where": where,
            })
        except Exception as e:
            print(f"⚠️  STT 녹음 저장 실패: {e}")
            return None
        return sid

    def note_heard(self, sid: str | None, heard: str) -> None:
        """LLM 이 바로잡은(또는 그대로 둔) 말을 덧붙인다."""
        if not self.enabled or not sid:
            return
        try:
            self._append({"kind": "heard", "id": sid, "heard": heard or ""})
        except Exception as e:
            print(f"⚠️  STT 녹음 기록 실패: {e}")
