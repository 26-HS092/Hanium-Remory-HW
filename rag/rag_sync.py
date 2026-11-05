"""
rag_sync.py
-----------
백엔드에서 어르신 데이터(프로필·가족·사진 추억)를 받아 RAG(chroma)를 갱신한다.

증분(incremental) 방식:
- 각 기억에 안정적인 ID 를 부여하고, 지난번과 비교해서
  '새로 생긴 것 / 바뀐 것' 만 임베딩해서 add, '없어진 것' 은 delete 한다.
- 안 바뀐 기억은 다시 임베딩하지 않는다(사진 수백 장이어도 빠름).
- 사진 비전분석 결과는 캐시하여 재분석하지 않는다.

memory_builder / image_processor / retriever 와 같은 폴더(rag)에 둔다.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import requests

RAG_DIR       = Path(__file__).resolve().parent
DATA_DIR      = RAG_DIR / "data"
PHOTO_DIR     = RAG_DIR / "photos"
CHROMA_DIR    = RAG_DIR / "chroma_db"
CACHE_FILE    = RAG_DIR / "rag_sync_cache.json"      # 사진 비전분석 캐시
MANIFEST_FILE = RAG_DIR / "rag_manifest.json"        # {chroma_id: 내용해시}
# {chroma_id: 최신 S3 주소}. 대화 중 관련 사진을 화면에 띄울 때 쓴다(retriever.py).
# 주소는 만료되므로 chroma 메타데이터(글이 바뀔 때만 다시 씀)가 아니라 여기에
# 동기화 때마다 새로 쓴다.
PHOTO_URL_FILE = RAG_DIR / "photo_urls.json"
# 화면에 띄울 사진. 동기화 때 받아 화면 크기로 줄여 둔다. S3 에서 원본을 그때그때
# 받으면 한 장에 1~4초 걸려서(미국 서버, 큰 건 2~3MB), 모리가 말을 시작해도 안 뜬다.
DISPLAY_DIR    = PHOTO_DIR / "display"
DISPLAY_SIZE   = (800, 480)                          # 인형 화면
# 앱에서 지울 수 없는 추억을 인형 쪽에서만 빼 둘 memoryId 목록(JSON 배열).
# 동기화마다 다시 읽으므로 고친 뒤 재시작하지 않아도 다음 동기화부터 빠진다.
EXCLUDE_FILE   = RAG_DIR / "excluded_memories.json"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"⚠️  파일 저장 실패({path.name}): {e}")


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def display_photo_path(url: str) -> Path:
    """사진 주소에 해당하는 화면용 파일 자리. 서명(쿼리)은 동기화마다 바뀌어서 빼고
    경로로만 이름을 짓는다 — 가족이 사진을 바꾸면 경로가 바뀌어 새로 받는다."""
    return DISPLAY_DIR / f"{_hash(url.split('?', 1)[0])}.jpg"


def _prepare_display_photos(urls) -> None:
    """화면용 사진이 없는 것만 받아 화면 크기로 줄여 둔다."""
    from PIL import Image, ImageOps
    made = 0
    for url in urls:
        dest = display_photo_path(url)
        if dest.exists():
            continue
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            from io import BytesIO
            img = ImageOps.exif_transpose(Image.open(BytesIO(resp.content)))
            img.thumbnail(DISPLAY_SIZE)
            DISPLAY_DIR.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            img.convert("RGB").save(tmp, "JPEG", quality=88)
            tmp.replace(dest)
            made += 1
        except Exception as e:
            print(f"⚠️  화면용 사진 만들기 실패: {e}")
    if made:
        print(f"🖼️  화면용 사진 {made}장 준비")


def _clean_meta(meta: dict) -> dict:
    """chromadb 는 빈 문자열/None 메타데이터를 거부하므로 그런 값은 뺀다."""
    return {k: v for k, v in meta.items() if v not in ("", None)}


# ── 문서(임베딩 단위) 만들기 ─────────────────────────────────────
def _family_docs(patient_id: str, patient_name: str, family: list, note: Optional[str]) -> dict:
    """가족·메모 → {chroma_id: (text, metadata)}."""
    docs = {}
    for f in family:
        rel = (f.get("relation") or "가족").strip()
        name = (f.get("name") or "").strip()
        if not name:
            continue
        text = f"{rel} 이름은 {name}이다."
        docs[f"fam_{_hash(rel + name)}"] = (text, {
            "memory_id": f"family_{name}", "patient_id": patient_id, "patient_name": patient_name,
            "category": "family", "importance": "high", "source_type": "text",
            "keywords": f"{rel}, {name}", "trigger_phrases": f"{rel}, {name}",
        })
    if note:
        # 노트는 여러 사실이 섞여 있으므로 문장 단위로 쪼개 각각 문서로 만든다.
        # (한 덩어리로 넣으면 "별명이 뭐야?" 같은 질문에 검색이 흐려짐)
        import re
        for s in re.split(r"[.\n]", note):
            s = s.strip()
            if len(s) < 2:
                continue
            docs[f"note_{_hash(s)}"] = (s, {
                "memory_id": f"note_{_hash(s)}", "patient_id": patient_id,
                "patient_name": patient_name, "category": "profile",
                "importance": "medium", "source_type": "text",
                "keywords": "", "trigger_phrases": "",
            })
    return docs


def _photo_doc(memory: dict, patient_name: str, patient_id: str, vision_model, cache: dict,
               seen: set):
    """추억(사진) 1개 → (chroma_id, text, metadata). 캐시 우선.

    앱에서 같은 추억을 여러 번 올리면 memoryId 만 다르고 제목은 똑같다. 이미 넣은
    추억과 제목이 같으면 None 을 돌려 빼고, 사진을 받지도 분석하지도 않는다.
    seen 에 지금까지 넣은 추억의 제목을 모은다. 제목이 없으면 빼지 않는다.
    """
    mid = str(memory.get("memoryId"))
    title = " ".join((memory.get("title") or "").split())
    if title:
        if title in seen:
            return None
        seen.add(title)
    # 가족이 적은 추억 원문. 비전분석 설명이 되고, 사진을 띄울 때 모리가 이 추억을
    # 꺼내 "이 기억 떠오르세요?" 하고 여쭌다.
    story = " ".join(
        x for x in [memory.get("title"), memory.get("period"), memory.get("description")] if x
    ).strip()
    pm = cache.get(mid) or {}
    dest = PHOTO_DIR / f"memory_{mid}.jpg"
    if "content" not in pm and memory.get("imageUrl"):   # 아직 안 받은 사진
        try:
            PHOTO_DIR.mkdir(exist_ok=True)
            resp = requests.get(memory["imageUrl"], timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        except Exception as e:
            print(f"⚠️  사진 받기 실패(memory {mid}): {e}")

    if "content" not in pm:
        result = None
        if vision_model is not None and dest.exists():
            try:
                from image_processor import analyze_image
                result = analyze_image(vision_model, dest, patient_name, story)
            except Exception as e:
                print(f"⚠️  사진 분석 실패(memory {mid}): {e}")
        if result:
            pm = {
                "content": result.get("content", "") or story,
                "importance": result.get("importance", "medium"),
                "source": dest.name,
                "keywords": ", ".join(result.get("keywords", [])),
                "trigger_phrases": ", ".join(result.get("trigger_phrases", [])),
            }
        else:   # 비전분석 없음/실패 → 보호자 설명 사용
            pm = {
                "content": story or (memory.get("title") or "가족 사진"),
                "importance": "medium", "source": "",
                "keywords": ", ".join(x for x in [memory.get("title"), memory.get("period")] if x),
                "trigger_phrases": "",
            }
        cache[mid] = pm

    text = pm["content"]
    meta = {
        "memory_id": f"photo_{mid}", "patient_id": patient_id, "patient_name": patient_name,
        "category": "photo", "importance": pm["importance"], "source_type": "photo",
        "source_file": pm["source"], "keywords": pm["keywords"], "trigger_phrases": pm["trigger_phrases"],
        "story": story,
    }
    return f"photo_{mid}", text, meta


def _embeddings():
    """retriever 가 쓰는 것과 같은 임베딩 모델(싱글턴)을 재사용한다."""
    import retriever
    return retriever._get_embeddings()


# ── 메인 ─────────────────────────────────────────────────────────
def sync_rag(patient_id: str, api_base: str, device_id, device_token: str,
             vision_model=None) -> bool:
    """백엔드 데이터로 chroma 를 갱신한다. 바뀐 게 있어 갱신했으면 True."""
    r = requests.get(
        f"{api_base}/devices/{device_id}/memories",
        headers={"X-Device-Token": device_token}, timeout=60,
    )
    r.raise_for_status()
    data = (r.json() or {}).get("data") or {}
    user = data.get("user") or {}
    patient_name = user.get("name") or "어르신"
    family = data.get("family") or []
    memories = data.get("memories") or []
    try:
        excluded = {str(x) for x in json.loads(EXCLUDE_FILE.read_text(encoding="utf-8"))}
    except Exception:
        excluded = set()
    memories = [m for m in memories if str(m.get("memoryId")) not in excluded]
    _save_json(PHOTO_URL_FILE, {
        f"photo_{m.get('memoryId')}": m["imageUrl"] for m in memories if m.get("imageUrl")
    })
    _prepare_display_photos(m["imageUrl"] for m in memories if m.get("imageUrl"))

    cache = _load_json(CACHE_FILE)

    # 현재 문서 전체 구성: {chroma_id: {"text", "metadata", "hash"}}
    docs = {}
    for cid, (text, meta) in _family_docs(patient_id, patient_name, family, user.get("note")).items():
        docs[cid] = {"text": text, "metadata": _clean_meta(meta), "hash": _hash(text)}
    # 같은 추억이 여러 개면 먼저 올린 것(작은 memoryId)을 남긴다. 순서를 고정해야
    # 동기화마다 남는 쪽이 바뀌어 다시 임베딩하는 일이 없다.
    def _mid_order(m):
        mid = str(m.get("memoryId"))
        return (0, int(mid), "") if mid.isdigit() else (1, 0, mid)
    seen = set()
    for m in sorted(memories, key=_mid_order):
        got = _photo_doc(m, patient_name, patient_id, vision_model, cache, seen)
        if got is None:
            continue   # 이미 넣은 추억과 제목이 같음 → 건너뜀
        cid, text, meta = got
        # 추억 글만 고쳐도 다시 넣도록 해시에 함께 넣는다.
        docs[cid] = {"text": text, "metadata": _clean_meta(meta),
                     "hash": _hash(text + "\n" + meta["story"])}
    _save_json(CACHE_FILE, cache)

    if not docs:
        return False   # 넣을 게 없으면 기존 chroma 그대로 둔다.

    from langchain_chroma import Chroma
    db_path = CHROMA_DIR / patient_id
    manifest = _load_json(MANIFEST_FILE)
    first_build = (not manifest) or (not db_path.exists())

    # 참고용 JSON 도 남겨둔다(디버깅). patient_info 는 main 이 매 턴 프롬프트에 넣는
    # 기본 정보(성함·나이·가족)다 — 검색에 맡기면 유사도가 낮아 빠질 때가 있다.
    DATA_DIR.mkdir(exist_ok=True)
    _save_json(DATA_DIR / f"patient_{patient_id}.json", {
        "patient_info": {
            "id": patient_id, "name": patient_name,
            "birthDate": user.get("birthDate"), "gender": user.get("gender"),
            # 가족 생년월일은 아직 서버가 안 준다. 주면 그대로 담긴다.
            "family": [{"relation": f.get("relation"), "name": f.get("name"),
                        "birthDate": f.get("birthDate")}
                       for f in family if f.get("name")],
        },
        "docs": {cid: d["text"] for cid, d in docs.items()},
    })

    if first_build:
        # 처음(또는 chroma 없음/매니페스트 없음) → 안정적 ID 로 통째 새로 만든다.
        if db_path.exists():
            shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True)
        ids = list(docs.keys())
        Chroma.from_texts(
            texts=[docs[i]["text"] for i in ids],
            metadatas=[docs[i]["metadata"] for i in ids],
            ids=ids,
            embedding=_embeddings(),
            collection_name=f"patient_{patient_id}",
            persist_directory=str(db_path),
            collection_metadata={"hnsw:space": "cosine"},
        )
        _save_json(MANIFEST_FILE, {i: docs[i]["hash"] for i in ids})
        print(f"🧠 RAG 전체 빌드: {len(ids)}개")
        _clear_retriever_cache(patient_id)
        return True

    # 증분: 새로 생긴 것 / 바뀐 것만 add, 없어진 것은 delete.
    cur, old = set(docs), set(manifest)
    to_add = cur - old
    to_del = old - cur
    to_upd = {i for i in (cur & old) if docs[i]["hash"] != manifest[i]}

    if not (to_add or to_del or to_upd):
        return False   # 바뀐 게 없음 → 아무것도 안 함.

    os.makedirs(db_path, exist_ok=True)
    db = Chroma(
        collection_name=f"patient_{patient_id}",
        embedding_function=_embeddings(),
        persist_directory=str(db_path),
    )
    del_ids = list(to_del | to_upd)
    if del_ids:
        db.delete(ids=del_ids)
    add_ids = list(to_add | to_upd)
    if add_ids:
        db.add_texts(
            texts=[docs[i]["text"] for i in add_ids],
            metadatas=[docs[i]["metadata"] for i in add_ids],
            ids=add_ids,
        )
    _save_json(MANIFEST_FILE, {i: docs[i]["hash"] for i in docs})
    print(f"🧠 RAG 증분 갱신: +{len(to_add)} ~{len(to_upd)} -{len(to_del)}")
    _clear_retriever_cache(patient_id)
    return True


def _clear_retriever_cache(patient_id: str) -> None:
    """retriever 가 캐시한 옛 DB 를 버리게 한다(다음 검색부터 새 DB 로드)."""
    try:
        import retriever
        retriever._db_cache.pop(patient_id, None)
    except Exception:
        pass
