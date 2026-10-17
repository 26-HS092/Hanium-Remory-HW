"""대화와 관련된 사진 찾기 테스트. 관련도가 높을 때만, 주소가 있을 때만 띄우는지 본다.

python test_related_photo.py 로 그냥 돌린다(인형에 pytest 를 깔지 않는다).
임베딩 검색(chroma)은 부르지 않고 가짜 결과로 바꿔 끼운다.
"""

import json
import os
import sys
import tempfile
import types

# 이 테스트는 임베딩 모델 없이 돈다. retriever 가 import 하는 모듈만 빈 껍데기로 둔다.
for name, attr in [("langchain_chroma", "Chroma"),
                   ("langchain_huggingface", "HuggingFaceEmbeddings")]:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        setattr(mod, attr, object)
        sys.modules[name] = mod

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag"))
import retriever  # noqa: E402

OK = "\033[32m✓\033[0m"
NG = "\033[31m✗\033[0m"

fails = 0


def check(good: bool, what: str, detail: str = ""):
    global fails
    if not good:
        fails += 1
    print(f"  {OK if good else NG} {what}" + ("" if good else f"  ({detail})"))


def fake_search(similarity: float | None):
    """사진 검색 결과를 하나(또는 없음)로 고정한다."""
    def search(patient_id, query, category, top_k=3):
        assert category == "photo", "사진 기억만 찾아야 한다"
        if similarity is None:
            return []
        return [{"content": "바닷가에서 수진이와", "category": "photo",
                 "importance": "medium", "source_type": "photo",
                 "similarity": similarity, "memory_id": "photo_7"}]
    return search


url_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
json.dump({"photo_7": "https://s3.example/memory_7.jpg"}, url_file)
url_file.close()
retriever.PHOTO_URL_FILE = url_file.name
retriever.PHOTO_MIN_SIMILARITY = 0.6

print("\n[관련도] 기준을 넘을 때만 띄운다")
retriever.retrieve_by_category = fake_search(0.72)
got = retriever.find_related_photo("P001", "수진이 바다")
check(got is not None and got["url"] == "https://s3.example/memory_7.jpg",
      "0.72 → 최신 주소와 함께 돌려준다", f"나온 값: {got!r}")

retriever.retrieve_by_category = fake_search(0.6)
check(retriever.find_related_photo("P001", "q") is not None, "기준과 같으면(0.6) 띄운다")

retriever.retrieve_by_category = fake_search(0.45)
check(retriever.find_related_photo("P001", "q") is None,
      "0.45 → 대화엔 쓰여도(0.4 이상) 사진은 안 띄운다")

retriever.retrieve_by_category = fake_search(None)
check(retriever.find_related_photo("P001", "q") is None, "사진 기억이 없으면 None")

print("\n[주소] 주소를 모르면 띄우지 않는다")
retriever.retrieve_by_category = fake_search(0.9)
with open(url_file.name, "w", encoding="utf-8") as f:
    json.dump({"photo_8": "https://s3.example/other.jpg"}, f)
check(retriever.find_related_photo("P001", "q") is None, "그 사진의 주소가 표에 없으면 None")

os.unlink(url_file.name)
check(retriever.find_related_photo("P001", "q") is None,
      "아직 동기화 전이라 주소 표가 없으면 None")

print(f"\n{'모두 통과' if not fails else f'{fails}개 실패'}")
sys.exit(1 if fails else 0)
