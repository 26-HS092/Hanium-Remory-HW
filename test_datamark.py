"""데이터마킹 테스트. 표식이 턴마다 바뀌는지, 모델이 따라 쓴 표식이 다 지워지는지 본다.

python test_datamark.py 로 그냥 돌린다(인형에 pytest 를 깔지 않는다).
"""

import re

import datamark

OK = "\033[32m✓\033[0m"
NG = "\033[31m✗\033[0m"

fails = 0


def check(good: bool, what: str, detail: str = ""):
    global fails
    if not good:
        fails += 1
    print(f"  {OK if good else NG} {what}" + ("" if good else f"  ({detail})"))


def expect_strip(text: str, want: str):
    got = datamark.strip_marks(text)
    check(got == want, f"{text!r} → {want!r}", f"나온 값: {got!r}")


print("\n[표식] 턴마다 새로 만든다")
marks = {datamark.new_mark() for _ in range(20)}
check(all(re.fullmatch(r"§[0-9a-f]{4}", m) for m in marks), "형식은 §+16진수 4자리")
check(len(marks) > 1, "호출할 때마다 다른 값", f"{len(marks)}개만 나옴")

print("\n[제거] 모델이 표식을 따라 써도 TTS 가 읽지 않게 지운다")
expect_strip("할머니가§a3f9좋아하시던§a3f9된장찌개요", "할머니가 좋아하시던 된장찌개요")
expect_strip("산책 §a3f9 가실래요?", "산책 가실래요?")
expect_strip("산책§ a3f9가실래요?", "산책 가실래요?")
expect_strip("§0b1c오늘 날씨 좋네요§0b1c", "오늘 날씨 좋네요")
expect_strip("표식 없는 평범한 대답이에요.", "표식 없는 평범한 대답이에요.")

print("\n[안내문] 이번 턴 표식을 그대로 알려준다")
m = datamark.new_mark()
check(m in datamark.instruction(m), "안내문에 표식이 들어감")

print()
if fails:
    print(f"{NG} {fails}개 실패")
    raise SystemExit(1)
print(f"{OK} 전부 통과")
