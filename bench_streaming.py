"""
bench_streaming.py
------------------
스트리밍을 켰을 때와 껐을 때, 모리가 대답을 시작하기까지 얼마나 걸리는지 잰다.

  켬: LLM 답을 받는 대로 문장마다 바로 말한다(Claude 스트리밍 + Fish 실시간).
  끔: LLM 답을 끝까지 다 받은 뒤 말한다(Claude 한 번에 받기 + Fish).

같은 질문을 글로 넣어 켬/끔을 번갈아 돌린다. 마이크·음성 인식은 거치지 않는다 —
스트리밍이 바꾸는 건 'LLM → 목소리' 구간뿐이라, 그 앞은 켬/끔이 똑같다.
재는 시간은 LLM 을 부른 순간(= 실제로는 음성 인식이 끝난 순간)부터
  첫 목소리   스피커에서 모리 목소리가 나오기 시작할 때까지   ← 주 결과
  끝까지      말을 다 마칠 때까지
RAG(기억 찾기)는 실제처럼 붙이되, 켬/끔이 같으므로 시간에서 뺀다.

준비
  - main.py 는 꺼 두기 (systemctl --user stop mori). 스피커를 같이 쓴다.
  - .env 에 ANTHROPIC_API_KEY, FISH_API_KEY. 이 스크립트는 .env 의 LLM_ENGINE·TTS_ENGINE
    과 상관없이 claude + fish 로 돈다(스트리밍은 그 조합에서만 된다).
  - 답이 스피커로 나온다. 30개 × 켬/끔 × 3번이면 180번 말해서 30분 넘게 걸린다.

실행
  python bench_streaming.py              # 질문 30개 × 켬/끔 × 3번
  python bench_streaming.py -n 3 -r 1    # 시험 삼아: 앞 3개만 1번
  python bench_streaming.py -c 짧은답    # 한 분류만

결과는 logs/bench_streaming/ 에 csv. 끝나면 분류별 중앙값 · p90(10번 중 9번은 이 안) ·
최대를 켬/끔 나란히 보여준다. Ctrl+C 로 멈춰도 거기까지 결과를 낸다.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 스트리밍은 Claude + Fish 에서만 된다. main 을 불러오기 전에 정해 둔다
# (dotenv 는 이미 있는 환경변수를 덮지 않는다).
os.environ["LLM_ENGINE"] = "claude"
os.environ["TTS_ENGINE"] = "fish"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "logs" / "bench_streaming"

# 분류별 10개씩. 기억 질문은 [관련 기억] 을 찾게 하는 질문이다(답은 그 어르신 자료에 따라 다름).
QUESTIONS = {
    "짧은답": [
        "안녕 모리야",
        "고마워",
        "배불러",
        "오늘 날씨 좋다",
        "응 알았어",
        "잘 잤어",
        "심심하다",
        "졸려",
        "밥 먹었어",
        "다녀왔어",
    ],
    "긴답": [
        "옛날이야기 하나 해 줄래?",
        "된장찌개는 어떻게 끓이는 거야?",
        "잠이 잘 안 오는데 어떻게 하면 좋을까?",
        "요즘 무릎이 아픈데 집에서 할 만한 운동 좀 알려 줘",
        "봄에 피는 꽃에는 뭐가 있는지 얘기해 줘",
        "우리 어릴 때는 어떻게 놀았는지 너도 알아?",
        "추석에는 보통 뭘 하는지 얘기해 줄래?",
        "기분이 울적할 때 뭘 하면 좋을까?",
        "김장은 어떻게 하는 건지 차례대로 말해 줘",
        "재미있는 수수께끼 하나 내 줘",
    ],
    "기억": [
        "우리 딸 이름이 뭐였지?",
        "내 고향이 어디였더라?",
        "젊었을 때 내가 무슨 일 했었지?",
        "우리 손주는 몇 살이지?",
        "내가 좋아하는 음식이 뭐였지?",
        "옛날에 가족이랑 어디 놀러 갔었지?",
        "우리 영감은 어떤 사람이었어?",
        "내가 젊을 때 살던 동네 기억나?",
        "요즘 우리 아들은 뭐 하고 지낸대?",
        "내 생일이 언제였더라?",
    ],
}
MODES = ["켬", "끔"]


def load_main():
    try:
        import main as M
    except KeyError as e:
        sys.exit(f"❌ .env 에 {e} 가 없어요 (ANTHROPIC_API_KEY · FISH_API_KEY 필요)")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("❌ .env 에 ANTHROPIC_API_KEY 가 없어요")
    return M


def get_context(M, question: str):
    """실제 대화처럼 기억을 찾아 붙인다(시간은 안 잼). 실패하면 기억 없이."""
    mark = M.new_mark()
    try:
        return M.build_context_prompt(M.PATIENT_ID, question, top_k=M.RAG_TOP_K, mark=mark), mark
    except Exception as e:
        print(f"⚠️  RAG 실패 — 기억 없이: {e}")
        return "", mark


def run_once(M, client, question: str, mode: str) -> dict:
    """질문 하나를 한 방식으로 돌려 시간을 잰다. 소리는 실제로 스피커로 나간다."""
    context, mark = get_context(M, question)
    box = {"voice": None}

    def on_voice():
        if box["voice"] is None:
            box["voice"] = time.perf_counter()

    reply = ""
    streamed = False
    t0 = time.perf_counter()
    if mode == "켬":
        # main.py 와 같은 순서: LLM 에 묻는 동안 Fish 실시간 연결을 미리 연다.
        live = M.open_live_tts()
        out = M.chat_with_memory(client, question, context, mark=mark, stream=True)
        t_llm = time.perf_counter()        # reply 앞 칸(위험·표정)이 온 시각
        rs = out.get("reply_stream")
        if rs is not None:
            streamed = True
            M.tts_play_stream(rs, live, on_start=on_voice)
            reply = rs.text()
        else:                               # 스트리밍이 안 됐다(거절·형식 오류) → 다 받은 글로
            live.close()
            reply = out["reply"]
            M.tts_play(reply, on_start=on_voice)
    else:
        out = M.chat_with_memory(client, question, context, mark=mark, stream=False)
        t_llm = time.perf_counter()        # 답 전체가 온 시각
        reply = out["reply"]
        M.tts_play(reply, on_start=on_voice)
    t_end = time.perf_counter()

    return {
        "first_voice": round(box["voice"] - t0, 3) if box["voice"] else "",
        "llm": round(t_llm - t0, 3),
        "total": round(t_end - t0, 3),
        "reply_chars": len(reply),
        "streamed": int(streamed),
        "reply": reply,
    }


def stats(xs):
    xs = [x for x in xs if x != ""]
    if not xs:
        return None
    return np.median(xs), np.percentile(xs, 90), max(xs), len(xs)


def fmt(s):
    return f"{s[0]:4.2f} / {s[1]:4.2f} / {s[2]:4.2f}" if s else "     -       "


def summarize(rows):
    if not rows:
        return
    cats = [c for c in QUESTIONS if any(r["category"] == c for r in rows)] + ["전체"]

    def pick(cat, mode, key):
        return [r[key] for r in rows if r["mode"] == mode and not r["error"]
                and (cat == "전체" or r["category"] == cat)]

    for key, title in (("first_voice", "첫 목소리까지"), ("total", "말을 다 마칠 때까지")):
        print(f"\n━━ {title} (초 · 중앙값 / p90 / 최대) ━━")
        print(f"  {'분류':<6}  {'스트리밍 켬':<20}  {'끔':<20}  중앙값 차이")
        for cat in cats:
            on, off = stats(pick(cat, "켬", key)), stats(pick(cat, "끔", key))
            diff = f"{on[0] - off[0]:+.2f}" if on and off else ""
            n = f"({on[3]}번)" if on else ""
            print(f"  {cat:<6}  {fmt(on):<20}  {fmt(off):<20}  {diff}  {n}")
    fell = sum(1 for r in rows if r["mode"] == "켬" and not r["error"] and not r.get("streamed"))
    if fell:
        print(f"\n참고: 켬인데 스트리밍이 안 되고 다 받아 말한 게 {fell}번 (csv 의 streamed=0)")
    errs = sum(1 for r in rows if r["error"])
    if errs:
        print(f"\n⚠️  실패 {errs}번 (csv 의 error 칸 참고) — 위 숫자에서 뺐어요")


def main():
    ap = argparse.ArgumentParser(description="스트리밍 켬/끔 응답 시간 비교")
    ap.add_argument("-r", "--repeat", type=int, default=3, help="반복 횟수 (기본 3)")
    ap.add_argument("-n", type=int, help="분류마다 앞에서 n개만")
    ap.add_argument("-c", "--category", choices=list(QUESTIONS), help="이 분류만")
    ap.add_argument("--gap", type=float, default=1.0, help="질문 사이 쉬는 초 (기본 1)")
    a = ap.parse_args()

    M = load_main()
    client = M.anthropic.Anthropic()
    print(f"🧠 LLM: {M.CLAUDE_MODEL} · TTS: Fish {M.FISH_MODEL} ({M.FISH_LATENCY})")
    M.warmup()

    items = [(cat, q) for cat, qs in QUESTIONS.items()
             if not a.category or cat == a.category
             for q in qs[: a.n or None]]
    total = len(items) * len(MODES) * a.repeat
    print(f"질문 {len(items)}개 × 켬/끔 × {a.repeat}번 = {total}번")

    # 첫 호출은 연결·프롬프트 캐시가 데워지느라 느리다. 한 번씩 돌리고 버린다.
    print("\n🔥 몸풀기 (기록 안 함)")
    for mode in MODES:
        try:
            run_once(M, client, "안녕", mode)
        except Exception as e:
            print(f"⚠️  몸풀기 실패({mode}): {e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.csv"
    rows, k = [], 0
    try:
        for rep in range(1, a.repeat + 1):
            for i, (cat, q) in enumerate(items):
                # 켬/끔 순서를 번갈아 바꾼다. 늘 같은 쪽이 먼저면 캐시·네트워크 상태가 한쪽에 쏠린다.
                order = MODES if (i + rep) % 2 == 0 else MODES[::-1]
                for mode in order:
                    k += 1
                    time.sleep(a.gap)
                    print(f"\n[{k}/{total}] {rep}회 · {cat} · 스트리밍 {mode} · {q}")
                    row = {"run": rep, "category": cat, "question": q, "mode": mode, "error": ""}
                    try:
                        row.update(run_once(M, client, q, mode))
                        print(f"   → 첫 목소리 {row['first_voice']}s · 끝까지 {row['total']}s")
                    except Exception as e:
                        row["error"] = repr(e)[:200]
                        print(f"   ❌ 실패: {e}")
                    rows.append(row)
    except KeyboardInterrupt:
        print("\n⏹  멈춤")

    if rows:
        keys = ["run", "category", "question", "mode", "first_voice", "llm", "total",
                "reply_chars", "streamed", "reply", "error"]
        with open(out, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        summarize(rows)
        print(f"\n📄 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
