"""
stt_keywords.py
---------------
음성 인식이 잘 틀리는 고유명사(어르신·가족 이름 등)를 모은다.

CLOVA Speech 의 키워드 부스팅, Whisper 의 prompt 힌트, stt_compare.py 의
핵심 단어 정답률이 모두 같은 목록을 쓴다.

- rag_sync 가 남기는 rag/data/patient_{id}.json 의 성함·가족 이름
- 이름은 성을 뺀 부분도 넣는다(어르신은 "민수야" 처럼 이름만 부르신다)
- .env 의 STT_BOOST_WORDS(쉼표로 구분): 약 이름·동네 이름처럼 직접 넣을 말
"""

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALWAYS = ["모리"]          # 인형 이름


def load_patient_info(patient_id: str) -> dict:
    try:
        with open(ROOT / "rag" / "data" / f"patient_{patient_id}.json",
                  encoding="utf-8") as f:
            return json.load(f).get("patient_info") or {}
    except (OSError, ValueError):
        return {}


def _clean(word) -> str:
    # 부스팅 목록은 쉼표로 이어 붙이므로 쉼표·줄바꿈을 지운다.
    return re.sub(r"[,\n\r\t]+", " ", " ".join(str(word or "").split()))[:20].strip()


def _name_variants(name: str) -> list[str]:
    name = _clean(name)
    if not name or name == "어르신":
        return []
    out = [name]
    # 세 글자 한글 이름이면 성을 뺀 두 글자도("김민수" → "민수").
    if re.fullmatch(r"[가-힣]{3}", name):
        out.append(name[1:])
    return out


def boost_words(patient_id: str) -> list[str]:
    """부스팅할 단어 목록(중복 없이, 순서 유지)."""
    info = load_patient_info(patient_id)
    words = list(ALWAYS)
    words += _name_variants(info.get("name"))
    for f in info.get("family") or []:
        words += _name_variants(f.get("name"))
    words += [_clean(w) for w in os.getenv("STT_BOOST_WORDS", "").split(",")]
    seen, out = set(), []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out
