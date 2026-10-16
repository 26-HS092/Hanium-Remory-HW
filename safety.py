"""어르신 말에서 위험 신호를 가려낸다. 규칙과 LLM 두 겹으로 본다.

규칙(classify)이 먼저다. LLM 이 못 미더워서가 아니라, 규칙만이 할 수 있는 일이
둘 있어서다.

  - 가족 알림이 모델의 자진 신고에 걸리지 않는다. 판별을 LLM 에만 맡기면
    모델이 risk 를 안 채우는 순간 자해 알림이 조용히 안 간다. 가장 중요한
    경로가 가장 조용하게 실패하는 셈이다.
  - Groq 이 죽어도 동작한다. LLM 호출이 실패하면 그 턴은 통째로 날아가는데,
    규칙은 네트워크와 무관하다.

규칙은 적어 둔 표현만 안다. 처음 보는 어법은 새어 나가므로, 놓친 것은 LLM 이
받는다(escalate). 대화를 만드는 그 호출에 필드 하나를 얹는 것이라 왕복이
늘지 않는다. 규칙이 무언가를 잡았더라도 LLM 이 더 급한 것을 보면 그쪽으로
올린다 — "씨발 그냥 자다가 안 깨어났으면" 을 규칙은 거친 말로만 본다.
고정 응답과 알림은 어느 쪽이 잡았든 같다.

가려낸 뒤 하는 일은 종류마다 다르다.

  타해   남을 해치고 싶다는 말. 치매의 공격성은 흔한 증상이라 야단칠 일이
         아니지만, 맞장구쳐서도 안 된다. 정해진 말로 가라앉히고 기록한다.
  자해   LLM 을 거치지 않고 정해진 말로 답한다. 모델이 무슨 말을 할지
         모르는 채로 두면 안 되는 자리다. 가족에게 바로 알린다.
  의료   약·치료 판단은 인형이 할 일이 아니다. 가족에게 여쭙도록 넘긴다.
  학대   기록만 남긴다. 사실 확인이 안 된 정황이고, 실시간 알림은 오히려
         어르신을 위험하게 만들 수 있다(아래 ABUSE 설명 참고).
  거친말 훈계하지 않는다. 치매의 탈억제는 증상이지 악의가 아니다. 감정을
         받아준 뒤 화제를 옮기도록 LLM 에 지침만 얹고, 횟수를 기록한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# 종류. 백엔드 safety_events.kind 와 같은 값을 쓴다.
SELF_HARM = "self_harm"
HARM_OTHERS = "harm_others"
MEDICAL = "medical"
ABUSE = "abuse"
PROFANITY = "profanity"

# LLM 에게 판단을 맡기는 종류. 거친 말은 빼뒀다 — 규칙으로 충분하고,
# 모델이 판단하게 하면 기준이 그때그때 달라진다.
LLM_KINDS = {SELF_HARM, HARM_OTHERS, MEDICAL, ABUSE}


@dataclass
class Risk:
    kind: str
    matched: str                    # 무엇에 걸렸는지 (기록·디버깅용)
    reply: Optional[str] = None     # LLM 을 거치지 않고 바로 할 말
    hint: Optional[str] = None      # LLM 에 얹을 지침
    expression: str = "위로"
    alert: bool = False             # 가족에게 즉시 알릴지


# ── 자해·자살 ────────────────────────────────────────
# 가르는 것은 동사가 아니라 어미다.
#   ~겠      "힘들어 죽겠다", "죽어버리겠네"  → 강조 관용구. 잡지 않는다.
#   ~고 싶   "죽고 싶다", "죽어버리고 싶다"   → 욕구. 잡는다.
#   ~었으면  "죽었으면", "죽어버렸으면"       → 소망. 잡는다.
# 어간과 어미 사이에 보조용언 '-어버리-' 가 끼어도 뜻은 그대로라 함께 받는다
# (이걸 빼먹어서 "죽어버리고 싶어" 를 놓쳤었다).
_SELF_HARM = [
    r"죽(어버리|어)?고\s*싶", r"죽(어버렸|었)으면\s*(좋겠|싶)",
    r"살기\s*싫", r"살고\s*싶지\s*않", r"그만\s*살",
    r"목숨\S*\s*끊", r"자살", r"세상\S*\s*뜨고\s*싶",
    r"따라\s*죽", r"확\s*죽어", r"없어져야",
    # "이제 죽어야지", "빨리 죽어야지" — 어르신들이 체념조로 자주 쓰지만,
    # 노인 돌봄에서는 소극적 자살 사고로 본다. 시끄러우면 이 줄을 빼면 된다.
    r"죽어야\s*(지|겠)",
]

# ── 타해(남을 해치려는 말) ───────────────────────────
# 자해는 '죽어-', 타해는 '죽여-' 로 한 글자 차이다. 뜻도 대응도 달라서 따로 본다.
# '죽여준다'(칭찬)는 걸리지 않는다 — "맛이 죽여주네" 처럼 아주 흔하다.
_HARM_OTHERS = [
    r"죽(여버리|이)고\s*싶", r"죽여\s*버릴", r"죽여야\s*(지|겠)",
    r"때려\s*죽", r"패\S*\s*버리고\s*싶",
]

# ── 의료 판단 ────────────────────────────────────────
_MEDICAL = [
    r"약\S*\s*(더|두\s*알|세\s*알|여러|많이)\s*먹",
    r"약\S*\s*먹어도\s*(되|괜찮)", r"약\S*\s*안\s*먹어도",
    r"약\S*\s*바꿔도", r"약\S*\s*끊어도",
    r"병원\s*안\s*가도", r"수술\S*\s*해야",
    r"무슨\s*약\S*\s*먹", r"어떤\s*약\S*\s*먹",
]

# ── 학대·방임 정황 ───────────────────────────────────
# '맞았' 은 뺐다 — "그 말이 맞았어" 처럼 '옳다' 는 뜻으로 훨씬 자주 쓰인다.
# '때려치우다'(그만두다)는 폭력이 아니다. "다 때려치우고 죽어버리고 싶다" 가
# 학대로 잡히던 것을 막는다.
_ABUSE = [
    r"때렸", r"때려(?!\s*치)", r"때리(?!\s*치)",
    # '주-' 만 보면 "밥도 안 줘", "안 줬어" 를 놓친다. 활용형을 함께 받는다.
    r"밥\S*\s*안\s*(주|줘|줬)", r"굶기",
    r"가둬", r"가뒀", r"내쫓", r"돈\S*\s*가져가",
]

# ── 강조 관용구 ──────────────────────────────────────
# 규칙이 '위험이 아니다' 라고 적극적으로 판단하는 표현. 표 1 의 '제외' 행이다.
# LLM 은 보조용언이 끼면("힘들어 죽어버리겠네") 강조의 '~겠' 을 놓치고 자해로
# 본다. 발화가 이 표현으로 끝나면 그 동사에 해당하는 LLM 판정은 받지 않는다
# (IDIOM_GUARD). 끝날 때로 좁힌 것은 "아파 죽겠어, 그냥 다 끝내고 싶다" 처럼
# 관용구 뒤에 진짜 신호가 이어지는 말까지 버리지 않으려는 것이다.
# 켤지는 eval_safety.py 로 재고 정한다.
_IDIOM = {
    SELF_HARM: [r"죽(어버리|어)?겠"],
    HARM_OTHERS: [r"죽여\s*(주|준|줘|줬)"],
}
_IDIOM_TAIL = r"[가-힣]{0,2}[\s.,!?~…]*$"   # 어미 한두 글자만 남아야 한다
IDIOM_GUARD = False

# ── 거친 말 ──────────────────────────────────────────
_PROFANITY = [
    r"씨발", r"시발", r"씹", r"좆", r"개새끼", r"새끼야", r"병신",
    r"지랄", r"닥쳐", r"미친놈", r"미친년", r"등신", r"염병",
]


def _first_match(text: str, patterns: list[str]) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(0)
    return None


def _self_harm(matched: str) -> Risk:
    return Risk(
        kind=SELF_HARM,
        matched=matched,
        # 모델에 맡기지 않는다. 이 자리에서 무슨 말이 나올지 모르는 채로
        # 두면 안 된다. 캐묻지도, 훈계하지도 않고 곁에 있음만 전한다.
        reply=(
            "그런 마음이 드셨군요. 많이 힘드셨겠어요. "
            "저는 여기 있어요. 가족분들께도 지금 말씀드릴게요."
        ),
        expression="위로",
        alert=True,
    )


def _harm_others(matched: str) -> Risk:
    return Risk(
        kind=HARM_OTHERS,
        matched=matched,
        # 맞장구쳐도 안 되고 야단쳐도 안 된다. 공격성은 치매에서 흔한
        # 증상이고 대개 통증·불편·혼란에서 온다. 화를 받아만 주고
        # 무슨 일이었는지 털어놓게 해서 가라앉힌다.
        reply=(
            "화가 많이 나셨군요. 마음이 많이 상하셨나 봐요. "
            "저랑 잠깐 숨 돌리시고, 무슨 일이 있었는지 편하게 말씀해 주세요."
        ),
        expression="위로",
    )


def _medical(matched: str) -> Risk:
    return Risk(
        kind=MEDICAL,
        matched=matched,
        reply=(
            "그건 제가 판단할 일이 아니어서요. "
            "가족분이나 의사 선생님께 꼭 여쭤보시는 게 좋겠어요."
        ),
        expression="경청",
    )


def _abuse(matched: str) -> Risk:
    return Risk(
        kind=ABUSE,
        matched=matched,
        # 사실 확인이 안 된 이야기다. 인형이 편들거나 판단하면 안 되고,
        # 그렇다고 넘겨서도 안 된다. 받아만 주고 기록으로 남긴다.
        hint=(
            "[안전] 어르신이 누군가에게 험한 일을 당했다는 이야기를 하셨습니다. "
            "사실인지 판단하거나 편들지 말고, 캐묻지도 마세요. "
            "'많이 속상하셨겠어요' 처럼 마음만 받아주고, "
            "'가족에게 알리겠다' 같은 약속은 하지 마세요."
        ),
        expression="위로",
    )


# 심각도 순. classify 가 규칙을 보는 순서와 같다.
_SEVERITY = [SELF_HARM, HARM_OTHERS, MEDICAL, ABUSE, PROFANITY]

_BUILDERS = {
    SELF_HARM: _self_harm,
    HARM_OTHERS: _harm_others,
    MEDICAL: _medical,
    ABUSE: _abuse,
}


def from_llm(kind: Optional[str]) -> Optional[Risk]:
    """LLM 이 잡아낸 신호를 규칙이 잡은 것과 같은 모양으로 만든다.

    규칙이 놓친 낯선 어법을 여기서 받는다. 잡힌 뒤의 대응은 어느 쪽이
    잡았든 같아야 하므로, Risk 를 만드는 곳을 한 군데로 둔다.
    """
    builder = _BUILDERS.get(kind or "")
    return builder("LLM") if builder else None


def idiom_of(text: str, kind: str) -> Optional[str]:
    """text 가 kind 를 부정하는 강조 관용구로 끝나면 그 부분을 준다."""
    return _first_match(
        (text or "").strip(), [p + _IDIOM_TAIL for p in _IDIOM.get(kind, [])]
    )


def escalate(rule: Optional[Risk], llm_kind: Optional[str], text: str = "",
             idiom_guard: Optional[bool] = None) -> Optional[Risk]:
    """LLM 판정이 규칙 판정보다 급하면 그 Risk 를, 아니면 None 을 준다.

    규칙의 우선순위는 규칙끼리만 통한다. 규칙이 약한 쪽(거친 말·의료)을
    먼저 잡았다고 LLM 이 본 자해를 버리면, 가장 비싼 실수를 우선순위가
    만들어 내는 셈이다. 반대로 LLM 이 더 약하게 봤다고 규칙 판정을 낮추지는
    않는다.
    """
    llm = from_llm(llm_kind)
    if llm is None:
        return None
    if idiom_guard is None:
        idiom_guard = IDIOM_GUARD
    if idiom_guard and idiom_of(text, llm.kind):
        return None
    if rule is None or _SEVERITY.index(llm.kind) < _SEVERITY.index(rule.kind):
        return llm
    return None


def classify(text: str) -> Optional[Risk]:
    """가장 위험한 것 하나만 준다. 걸리는 게 없으면 None.

    순서가 곧 우선순위다. 한 문장에 여러 개가 섞여 있으면 더 급한 쪽을 따른다
    ("죽고 싶은데 며느리가 밥도 안 줘" 는 자해로 다룬다).
    자해를 가장 먼저 본다 — 되돌릴 수 없는 쪽이라 놓쳤을 때 대가가 가장 크다.
    """
    text = (text or "").strip()
    if not text:
        return None

    for patterns, kind in (
        (_SELF_HARM, SELF_HARM),
        (_HARM_OTHERS, HARM_OTHERS),
        (_MEDICAL, MEDICAL),
        (_ABUSE, ABUSE),
    ):
        m = _first_match(text, patterns)
        if m:
            return _BUILDERS[kind](m)

    m = _first_match(text, _PROFANITY)
    if m:
        return Risk(
            kind=PROFANITY,
            matched=m,
            # 훈계하지 않는다. 탈억제는 증상이고, 초조함은 통증이나 불편의
            # 신호일 때가 많다. 야단치면 그 신호를 덮어버린다.
            hint=(
                "[안전] 어르신이 거친 말을 하셨습니다. 절대 훈계하거나 "
                "지적하지 마세요. 그 말에 담긴 감정(짜증·답답함)을 한 마디로 "
                "받아준 뒤, 편안한 다른 화제로 부드럽게 옮기세요."
            ),
            expression="위로",
        )

    return None
