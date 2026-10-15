"""
retriever.py
------------
역할: 환자 발화를 받아 관련 기억을 검색해서 반환
사용: LLM 담당 팀원이 이 파일의 retrieve_memories() 함수만 호출하면 됨
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ── 설정 ──────────────────────────────────────────────────────────────────────
DB_DIR          = "./chroma_db"
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
# ─────────────────────────────────────────────────────────────────────────────


# 임베딩 모델 + DB 모두 캐시 (검색할 때마다 새로 로드하지 않음)
_embedding_cache = None
_db_cache        = {}   # { "P001": Chroma객체, "P002": Chroma객체, ... }

def _get_embeddings():
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedding_cache


def _load_db(patient_id: str) -> Chroma:
    """벡터 DB 로드 (캐시 적용 - 처음 한 번만 로드)"""
    global _db_cache

    # 이미 로드된 DB면 바로 반환
    if patient_id in _db_cache:
        return _db_cache[patient_id]

    db_path = os.path.join(DB_DIR, patient_id)

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"환자 {patient_id}의 DB가 없습니다.\n"
            f"먼저 memory_builder.py를 실행해서 DB를 생성하세요."
        )

    db = Chroma(
        collection_name=f"patient_{patient_id}",
        embedding_function=_get_embeddings(),
        persist_directory=db_path
    )

    _db_cache[patient_id] = db   # 캐시에 저장
    return db


# ── LLM 담당자가 사용할 메인 함수 ─────────────────────────────────────────────

def retrieve_memories(patient_id: str, query: str, top_k: int = 3) -> list[dict]:
    """
    환자 발화와 유사한 기억을 검색해서 반환

    Args:
        patient_id : 환자 ID (예: "P001")
        query      : 환자 발화 또는 검색 문장
        top_k      : 반환할 기억 수 (기본 3개)

    Returns:
        [
            {
                "content"       : "기억 내용 텍스트",
                "category"      : "family",
                "importance"    : "high",
                "source_type"   : "text",
                "similarity"    : 0.92,       # 유사도 점수 (높을수록 관련성 높음)
                "keywords"      : "딸, 수진, 일요일",
                "trigger_phrases": "딸, 수진이, 큰애"
            },
            ...
        ]

    Example:
        memories = retrieve_memories("P001", "우리 딸이 보고싶어", top_k=3)
        for m in memories:
            print(m["content"])
    """
    vectordb = _load_db(patient_id)

    # 거리 점수와 함께 검색 (distance: 낮을수록 유사 → 1-distance로 유사도 변환)
    results = vectordb.similarity_search_with_score(
        query=query,
        k=top_k
    )

    memories = []
    for doc, distance in results:
        similarity = round(max(0.0, 1.0 - distance), 4)   # 코사인: distance=0~2 → similarity=1~-1, max(0)로 클리핑
        memories.append({
            "content"        : doc.page_content,
            "category"       : doc.metadata.get("category", ""),
            "importance"     : doc.metadata.get("importance", ""),
            "source_type"    : doc.metadata.get("source_type", "text"),
            "similarity"     : similarity,
            "keywords"       : doc.metadata.get("keywords", ""),
            "trigger_phrases": doc.metadata.get("trigger_phrases", ""),
            "memory_id"      : doc.metadata.get("memory_id", "")
        })

    return memories


def retrieve_by_category(patient_id: str, query: str,
                          category: str, top_k: int = 3) -> list[dict]:
    """
    특정 카테고리 안에서만 검색 (카테고리 필터 적용)

    Args:
        category : "family" / "history" / "routine" / "food" /
                   "hobby" / "health" / "emotion" / "place" / "photo"

    Example:
        # 음식 관련 기억만 검색
        memories = retrieve_by_category("P001", "뭐가 맛있어", "food")
    """
    vectordb = _load_db(patient_id)

    results = vectordb.similarity_search_with_score(
        query=query,
        k=top_k,
        filter={"category": category}
    )

    memories = []
    for doc, distance in results:
        similarity = round(max(0.0, 1.0 - distance), 4)
        memories.append({
            "content"    : doc.page_content,
            "category"   : doc.metadata.get("category", ""),
            "importance" : doc.metadata.get("importance", ""),
            "source_type": doc.metadata.get("source_type", "text"),
            "similarity" : similarity,
            "memory_id"  : doc.metadata.get("memory_id", "")
        })

    return memories


def retrieve_high_importance(patient_id: str, query: str,
                              top_k: int = 3) -> list[dict]:
    """
    중요도 높은(high) 기억만 검색
    → 핵심 정보(가족, 건강 등)를 우선 찾을 때 사용

    Example:
        memories = retrieve_high_importance("P001", "집에 가고싶어")
    """
    vectordb = _load_db(patient_id)

    results = vectordb.similarity_search_with_score(
        query=query,
        k=top_k,
        filter={"importance": "high"}
    )

    memories = []
    for doc, distance in results:
        similarity = round(max(0.0, 1.0 - distance), 4)
        memories.append({
            "content"    : doc.page_content,
            "category"   : doc.metadata.get("category", ""),
            "importance" : doc.metadata.get("importance", ""),
            "source_type": doc.metadata.get("source_type", "text"),
            "similarity" : similarity,
            "memory_id"  : doc.metadata.get("memory_id", "")
        })

    return memories


def build_context_prompt(patient_id: str, query: str, top_k: int = 3,
                         mark: str | None = None) -> str:
    """
    LLM에게 넘길 컨텍스트 프롬프트 문자열 생성
    → LLM 담당자가 이 함수를 바로 사용하면 됨

    Args:
        patient_id : 환자 ID
        query      : 환자 발화
        top_k      : 참고할 기억 수
        mark       : 있으면 기억 본문의 단어 사이를 이 표식으로 바꾼다(datamark.py).
                     가족 글에 심어 둔 지시문을 모델이 자료로 알아보게 하는 용도.

    Returns:
        LLM 프롬프트에 삽입할 컨텍스트 문자열

    Example:
        context = build_context_prompt("P001", "딸이 보고싶어")
        prompt  = system_prompt + context + patient_utterance
    """
    memories = retrieve_memories(patient_id, query, top_k)

    if not memories:
        return "[관련 기억 없음]"

    lines = ["[관련 기억]"]
    for i, mem in enumerate(memories, 1):
        source_label = "📷 사진기억" if mem["source_type"] == "photo" else "💬 기억"
        content = mem["content"]
        if mark:
            # 줄 단위가 아니라 본문 통째로 바꾼다. 줄바꿈이 남으면 둘째 줄부터
            # 표식 없이 들어가고, "2. (💬 기억 ...)" 처럼 시스템 줄을 흉내 낼 수 있다.
            content = mark.join(content.split())
        lines.append(
            f"{i}. ({source_label} | {mem['category']} | 유사도: {mem['similarity']:.2f})\n"
            f"   {content}"
        )

    return "\n".join(lines)


def get_patient_info(patient_id: str) -> dict:
    """
    DB에 저장된 기억 통계 반환

    Returns:
        {"total": 75, "by_category": {"family": 12, ...}}
    """
    vectordb  = _load_db(patient_id)
    all_docs  = vectordb.get()

    total = len(all_docs["ids"])
    by_category = {}

    for meta in all_docs["metadatas"]:
        cat = meta.get("category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        "patient_id" : patient_id,
        "total"      : total,
        "by_category": by_category
    }
