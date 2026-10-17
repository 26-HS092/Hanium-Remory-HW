"""
conversation_log.py
--------------------
대화 한 턴(어르신 발화 STT + 모리 응답)을 로컬에 append-only JSONL로 저장한다.
백엔드가 나중에 이 파일들을 읽어(pull) 가져간다. (전송/수집은 백엔드 담당)

설계:
- 한 줄 = 한 턴(JSON object). UTF-8 + ensure_ascii=False → 한글 그대로 저장.
- 파일은 환자별·날짜별로 분리: {base_dir}/{patient_id}/{YYYY-MM-DD}.jsonl
  (append-only라 백엔드가 tail/오프셋으로 안전하게 가져갈 수 있음)
- 웨이크워드~대화 종료까지를 하나의 세션(session_id)으로 묶는다.
- 여러 스레드에서 불러도 안전하도록 락으로 쓰기를 감싼다.
- 저장 실패가 대화를 끊지 않도록 예외는 삼키고 경고만 출력한다.
"""

import json
import threading
import datetime
import uuid
from pathlib import Path


class ConversationLogger:
    def __init__(self, base_dir, patient_id: str, device_id=None):
        self.base_dir = Path(base_dir)
        self.patient_id = patient_id
        self.device_id = device_id
        self._lock = threading.Lock()

    def new_session_id(self) -> str:
        """웨이크워드~대화 종료까지 한 세션을 묶는 id를 만든다."""
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{self.patient_id}-{ts}-{uuid.uuid4().hex[:6]}"

    def log_turn(
        self,
        user_text: str,
        reply: str,
        *,
        session_id: str | None = None,
        expression: str | None = None,
        emotion: dict | None = None,
    ) -> dict | None:
        """대화 한 턴을 JSONL 한 줄로 저장한다. 실패해도 None을 반환하고 넘어간다."""
        now = datetime.datetime.now().astimezone()
        record = {
            "id": uuid.uuid4().hex,                 # 턴 고유 id(백엔드 중복 방지용)
            "session_id": session_id,               # 같은 대화 세션 묶음
            "timestamp": now.isoformat(timespec="seconds"),
            "patient_id": self.patient_id,
            "device_id": self.device_id,
            "user_text": user_text,                 # STT로 인식된 어르신 발화
            "reply": reply,                         # 모리(LLM) 응답
            "expression": expression,               # 로봇 표정
            "emotion": emotion,                     # 감정 인식 결과(dict) 또는 None
            "synced": False,                        # 백엔드가 가져간 뒤 표시용(선택)
        }
        try:
            path = self._path_for(now.date())
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with path.open("a", encoding="utf-8") as f:
                    # 감정 신뢰도가 numpy 실수로 올 수 있어 default 로 풀어 준다.
                    f.write(json.dumps(record, ensure_ascii=False, default=_jsonable) + "\n")
                    f.flush()
        except Exception as e:
            print(f"⚠️  대화 저장 실패: {e}")
            return None
        return record

    def recent_turns(self, days: int = 2, limit: int = 30) -> list[dict]:
        """최근 days 일(오늘 포함) 동안 남긴 턴을 오래된 것부터 최대 limit 개 돌려준다.

        앞선 대화에서 하신 말("오늘 짜장면 먹었어")을 다음 대화에서도 기억하려고
        쓴다. 못 읽은 파일·줄은 건너뛴다 — 기억이 좀 빠져도 대화는 해야 한다.
        """
        today = datetime.date.today()
        turns: list[dict] = []
        for back in range(days - 1, -1, -1):
            path = self._path_for(today - datetime.timedelta(days=back))
            try:
                with self._lock:
                    lines = path.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"⚠️  이전 대화 읽기 실패: {e}")
                continue
            for line in lines:
                try:
                    turns.append(json.loads(line))
                except ValueError:
                    continue   # 쓰다 끊긴 줄
        return turns[-limit:]

    def _path_for(self, date: datetime.date) -> Path:
        return self.base_dir / self.patient_id / f"{date.isoformat()}.jsonl"


def _jsonable(obj):
    """json 이 모르는 값(numpy 스칼라 등)을 기본 타입으로 바꾼다."""
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


if __name__ == "__main__":
    # 간단 동작 확인
    logger = ConversationLogger("conversations", "P001", device_id="1")
    sid = logger.new_session_id()
    rec = logger.log_turn(
        "날씨가 좋아",
        "그러게요 어르신, 오늘 날씨가 참 좋네요.",
        session_id=sid,
        expression="행복",
        emotion={"label_ko": "기쁨", "confidence": 0.82, "n": 5},
    )
    print("저장됨:", rec)
