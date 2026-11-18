"""위험 발언 3단계 판정(규칙 → 소형 모델 → LLM)을 문장 몇 개로 보여 주는 데모.

    python demo_risk_pipeline.py                       # 기본 예시 2문장
    python demo_risk_pipeline.py "문장1" "문장2"        # 직접 넣기
    python demo_risk_pipeline.py --cache               # LLM 을 부르지 않고 평가 캐시 판정을 쓴다

판정 순서는 main.py 와 같다.
  규칙이 잡으면                          → 긴급(rule)
  LLM 이 위험이라 하고 모델도 0.15 이상  → 긴급(llm+model)
  LLM 만 위험이라 함                     → 지켜봄(llm) — 리포트에만
출력은 화면과 logs/risk_demo_<시각>.log 에 함께 남는다.
"""
from __future__ import annotations

import collections
import contextlib
import io
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import onnxruntime  # noqa: E402

onnxruntime.set_default_logger_severity(3)   # GPU 못 찾는다는 경고 숨김

import safety  # noqa: E402
from risk_model.runtime import ONNX_PATH, RiskClassifier  # noqa: E402

HERE = Path(__file__).parent
AGREE_THR = 0.15   # main.RISK_MODEL_AGREE_THR
CACHE = HERE / "eval_llm_cache_haiku_prompt2.jsonl"
EXAMPLES = [
    "요즘 자다가 죽기만을 바라며 잠이 들어.",
    "죽기 전에 딸을 보고 싶어서 매일 딸에게 전화하지만 딸이 전화를 받지 않아서 걱정돼.",
]
KO = {None: "위험 아님", "none": "위험 아님", "self_harm": "자해", "harm_others": "타해",
      "medical": "의료 판단", "abuse": "학대"}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_cache = "--cache" in sys.argv
    texts = args or EXAMPLES

    log_path = HERE / "logs" / f"risk_demo_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_path.parent.mkdir(exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")

    def log(msg: str = "") -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}" if msg else ""
        print(line, flush=True)
        log_f.write(line + "\n")

    clf = RiskClassifier()
    if use_cache:
        cached = collections.defaultdict(list)
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cached[d["text"]].append(d["risk"])
        llm_name = "LLM(평가 캐시)"
    else:
        from eval_safety import load_chat, make_client
        chat, engine, model = load_chat()
        client = make_client(engine)
        llm_name = f"LLM({model})"

    log(f"🛡️  위험 발언 3단계 판정 데모 — 규칙 / 소형 모델({ONNX_PATH.name}) / {llm_name}")
    for n, text in enumerate(texts, 1):
        log("─" * 60)
        log(f"🗣️  [{n}] 어르신: \"{text}\"")

        t0 = time.perf_counter()
        rule = safety.classify(text)
        ms = (time.perf_counter() - t0) * 1000
        log(f"  ① 키워드 규칙   {'🚨 ' + KO[rule.kind] if rule else '— 해당 없음':<14}"
            f"({ms:.1f}ms)")

        t0 = time.perf_counter()
        g = clf.classify(text)
        ms = (time.perf_counter() - t0) * 1000
        mark = "🚨" if g.risk_prob >= AGREE_THR else "—"
        log(f"  ② 소형 모델     {mark} 위험 확률 {g.risk_prob:.2f} (가장 가까운 종류: "
            f"{KO[g.top_risk]}, 강도 {g.level}, 시점 {g.frame})  ({ms:.0f}ms)")

        t0 = time.perf_counter()
        reply = ""
        if use_cache:
            runs = cached.get(text, [])
            llm = collections.Counter(runs).most_common(1)[0][0] if runs else None
            extra = f"{len(runs)}회 중 다수결" if runs else "캐시 없음"
        else:
            with contextlib.redirect_stdout(io.StringIO()):   # 토큰 사용량 줄 숨김
                out = chat(client, text, "", None)
            llm, reply = out.get("risk"), out.get("reply", "")
            extra = f"{(time.perf_counter() - t0):.1f}s"
        llm = None if llm in (None, "none") else llm
        log(f"  ③ LLM           {'🚨 ' + KO[llm] if llm else '— 위험 아님':<14}({extra})")

        urgent_kind = rule.kind if rule else (llm if llm and g.risk_prob >= AGREE_THR else None)
        if urgent_kind and safety.from_llm(urgent_kind).reply:
            reply = safety.from_llm(urgent_kind).reply   # main.py: 긴급이면 정해진 안전 답변
        if rule:
            verdict = f"🚨 긴급 (rule: {KO[rule.kind]}) — 가족에게 즉시 알림, 정해진 안전 답변"
        elif llm and g.risk_prob >= AGREE_THR:
            verdict = f"🚨 긴급 (llm+model: {KO[llm]}) — LLM·모델 모두 위험, 가족에게 즉시 알림"
        elif llm:
            verdict = (f"👀 지켜봄 (llm: {KO[llm]}) — 모델은 동의 안 함"
                       f"(<{AGREE_THR}), 리포트에만 기록")
        else:
            verdict = "✅ 위험 없음 — 평소처럼 대화"
        log(f"  ⇒ 최종 {verdict}")
        if reply:
            log(f"  🐻 모리: {reply}")
    log("─" * 60)
    log_f.close()
    print(f"\n로그 저장: {log_path}")


if __name__ == "__main__":
    main()
