"""안전 판별 테스트. 오탐(멀쩡한 말을 위험으로 보는 것)을 특히 신경 써서 본다.

python test_safety.py 로 그냥 돌린다(인형에 pytest 를 깔지 않는다).
"""

import safety

OK = "\033[32m✓\033[0m"
NG = "\033[31m✗\033[0m"

fails = 0


def expect(text: str, kind, why: str = ""):
    global fails
    got = safety.classify(text)
    got_kind = got.kind if got else None
    good = got_kind == kind
    if not good:
        fails += 1
    mark = OK if good else NG
    detail = "" if good else f"  (나온 값: {got_kind})"
    print(f"  {mark} {text!r} → {kind}{detail}  {why}")


print("\n[자해·자살] 잡아야 하는 것")
expect("이제 그만 죽고 싶어", safety.SELF_HARM)
expect("차라리 죽었으면 좋겠어", safety.SELF_HARM)
expect("살기 싫다 이제", safety.SELF_HARM)
expect("영감 따라 죽어야지", safety.SELF_HARM)
expect("내가 없어져야 자식들이 편하지", safety.SELF_HARM)

print("\n[자해·자살] 걸리면 안 되는 것 — '죽겠다' 는 한국어에서 흔한 강조다")
expect("아이고 힘들어 죽겠네", None, "← 매일 나올 말")
expect("배고파 죽겠다", None)
expect("좋아 죽겠어 손주 온다니까", None)
expect("허리가 아파 죽겠어", None)

print("\n[자해·자살] 보조용언 '-어버리-' 가 껴도 어미가 기준이다")
expect("죽어버리고 싶어", safety.SELF_HARM, "← ~고 싶 : 욕구")
expect("그냥 죽어버렸으면 좋겠어", safety.SELF_HARM, "← ~었으면 : 소망")
expect("확 죽어버릴까", safety.SELF_HARM)
expect("힘들어 죽어버리겠네", None, "← ~겠 : 강조 관용구")
expect("아이고 죽어버리겠다", None)

print("\n[자해·자살] 체념조도 소극적 자살 사고로 본다")
expect("이제 죽어야지", safety.SELF_HARM)
expect("빨리 죽어야지 자식들 고생시키지 말고", safety.SELF_HARM)

print("\n[우선순위] '때려치우다' 는 폭력이 아니다")
expect("다 때려치우고 죽어버리고 싶다", safety.SELF_HARM, "← 학대로 새면 알림이 안 간다")
expect("일 때려치웠어", None)

print("\n[타해] 자해는 '죽어-', 타해는 '죽여-' — 한 글자 차이다")
expect("저 놈 죽여버리고 싶어", safety.HARM_OTHERS)
expect("확 죽여버릴까", safety.HARM_OTHERS)
expect("죽이고 싶을 만큼 밉다", safety.HARM_OTHERS)
expect("때려죽이고 싶어", safety.HARM_OTHERS)
expect("죽어버리고 싶어", safety.SELF_HARM, "← 한 글자 차이지만 자해다")

print("\n[타해] '죽여준다' 는 칭찬이다")
expect("이 노래 죽여준다", None)
expect("맛이 죽여주네", None)

print("\n[의료] 잡아야 하는 것")
expect("이 약 두 알 먹어도 되나?", safety.MEDICAL)
expect("약 안 먹어도 괜찮겠지?", safety.MEDICAL)
expect("병원 안 가도 될까", safety.MEDICAL)

print("\n[의료] 걸리면 안 되는 것")
expect("아침에 약 먹었어", None, "← 복용 보고일 뿐")
expect("약이 써서 싫어", None)

print("\n[학대] 잡아야 하는 것")
expect("며느리가 나를 때렸어", safety.ABUSE)
expect("밥을 안 주더라", safety.ABUSE)
expect("방에 가둬놨어", safety.ABUSE)

print("\n[학대] 걸리면 안 되는 것 — '맞았' 은 '옳다' 로 훨씬 자주 쓴다")
expect("네 말이 맞았어", None)
expect("어제 비가 많이 왔어", None)

print("\n[거친 말]")
expect("에이 씨발 리모컨이 어디 갔어", safety.PROFANITY)
expect("이런 지랄맞은 날씨가 있나", safety.PROFANITY)

print("\n[평범한 대화] 아무것도 걸리면 안 된다")
for t in ["오늘 날씨가 좋네", "손녀가 보고 싶어", "점심에 뭐 먹을까",
          "옛날에 부산 살았지", "다리가 좀 아파", "노래 한 곡 틀어줘",
          "우리 딸 이름이 뭐였더라", "집에 가고 싶어"]:
    expect(t, None)

print("\n[우선순위] 섞여 있으면 더 급한 쪽")
expect("죽고 싶은데 며느리가 밥도 안 줘", safety.SELF_HARM, "← 학대보다 자해가 먼저")

print("\n[빈 입력]")
expect("", None)
expect("   ", None)

print("\n[LLM 이 잡은 신호] 규칙이 놓친 것을 받아 같은 대응으로 잇는다")


def expect_llm(kind, want_reply: bool, want_alert: bool):
    global fails
    r = safety.from_llm(kind)
    got = (r is not None, bool(r and r.reply), bool(r and r.alert))
    want = (kind in safety.LLM_KINDS, want_reply, want_alert)
    good = got == want
    if not good:
        fails += 1
    print(f"  {OK if good else NG} from_llm({kind!r}) → {got}" +
          ("" if good else f"  (기대 {want})"))


expect_llm("self_harm", want_reply=True, want_alert=True)
expect_llm("harm_others", want_reply=True, want_alert=False)
expect_llm("medical", want_reply=True, want_alert=False)
expect_llm("abuse", want_reply=False, want_alert=False)   # 모델이 쓴 답을 쓴다
# 모델이 아무 값이나 넣어도 흘려보내지 않는다
expect_llm("none", want_reply=False, want_alert=False)
expect_llm("위험함", want_reply=False, want_alert=False)
expect_llm(None, want_reply=False, want_alert=False)
expect_llm("profanity", want_reply=False, want_alert=False)   # LLM 판단 대상이 아니다

print("\n[학대] '안 주-' 의 활용형")
expect("며느리가 밥도 안 줘", safety.ABUSE)
expect("밥을 안 줬어", safety.ABUSE)
expect("밥을 안 주더라", safety.ABUSE)

print("\n[격상] 규칙이 약하게 봐도 LLM 이 더 급하게 보면 올린다")


def expect_escalate(text: str, llm_kind, want):
    global fails
    rule = safety.classify(text)
    up = safety.escalate(rule, llm_kind)
    final = up or rule
    got = final.kind if final else None
    good = got == want
    if not good:
        fails += 1
    print(f"  {OK if good else NG} {text!r} 규칙={rule.kind if rule else None}"
          f" + LLM={llm_kind} → {want}" + ("" if good else f"  (나온 값: {got})"))


expect_escalate("에이 씨발 그냥 자다가 안 깨어났으면 좋겠어", "self_harm", safety.SELF_HARM)
expect_escalate("약 두 알 먹어도 되나, 이제 다 끝내고 싶어", "self_harm", safety.SELF_HARM)
expect_escalate("자다가 안 깨어났으면 좋겠어", "self_harm", safety.SELF_HARM)   # 규칙 없음
# 낮추지는 않는다
expect_escalate("이제 그만 죽고 싶어", "abuse", safety.SELF_HARM)
expect_escalate("약 두 알 먹어도 되나", "abuse", safety.MEDICAL)
expect_escalate("약 두 알 먹어도 되나", "none", safety.MEDICAL)
# 거친 말 위의 학대는 올린다(거친 말은 가장 약하다)
expect_escalate("씨발 며느리가 때렸어", "abuse", safety.ABUSE)
expect_escalate("씨발 짜증나", "none", safety.PROFANITY)

print("\n[관용구 가드] 발화가 관용구로 끝날 때만 LLM 판정을 막는다")


def expect_guard(text: str, llm_kind, want):
    global fails
    rule = safety.classify(text)
    up = safety.escalate(rule, llm_kind, text, idiom_guard=True)
    final = up or rule
    got = final.kind if final else None
    good = got == want
    if not good:
        fails += 1
    print(f"  {OK if good else NG} {text!r} + LLM={llm_kind} → {want}"
          + ("" if good else f"  (나온 값: {got})"))


expect_guard("힘들어 죽어버리겠네", "self_harm", None)
expect_guard("아이고 힘들어 죽겠네!", "self_harm", None)
expect_guard("이 노래 죽여준다", "harm_others", None)
expect_guard("맛이 죽여주네", "harm_others", None)
# 관용구 뒤에 말이 이어지면 막지 않는다
expect_guard("아파 죽겠어 그냥 다 끝내고 싶다", "self_harm", safety.SELF_HARM)
expect_guard("힘들어 죽겠다 정말, 더 살아서 뭐 하나 싶어", "self_harm", safety.SELF_HARM)
# 관용구가 아닌 동사의 판정은 막지 않는다
expect_guard("허리가 아파 죽겠어", "medical", safety.MEDICAL)
# 가드는 기본으로 꺼져 있다
_r = safety.escalate(None, "self_harm", "힘들어 죽어버리겠네")
if not (_r and _r.kind == safety.SELF_HARM):
    fails += 1
print(f"  {OK if _r else NG} IDIOM_GUARD 기본값 꺼짐")

print(f"\n{'모두 통과' if fails == 0 else str(fails) + '건 실패'}")
raise SystemExit(1 if fails else 0)
