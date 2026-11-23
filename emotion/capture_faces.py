"""
capture_faces.py
----------------
인형 카메라로 표정 평가용 사진을 찍는다. 감정 번호를 누르면 그 감정 폴더에 저장한다.

━━ 찍는 방법 ━━
1. 준비
   - 파이에서 main.py 를 끈다(카메라를 같이 못 연다).
   - 인형은 원래 자리에, 고개가 사람이 앉을 쪽을 보게 맞춘다.
   - 찍히는 사람이 인형 앞에 앉는다.

2. 켜기 (--who 이름만 바꿔서)
       python emotion/capture_faces.py --who 수아
       python emotion/capture_faces.py --who 수아 --delay 3     # 혼자 찍을 때: 3초 뒤 찰칵
   켜면 얼굴이 잡혔는지 먼저 알려준다. '얼굴이 안 보여요' 면 자리·고개를 다시 맞춘다.

3. 찍기 — 표정을 지으면 번호 + Enter
       1 기쁨   2 슬픔   3 분노   4 불안   5 중립(아무 표정 없이)
       c  저장하지 않고 얼굴이 잡히는지만 보기
       u  방금 찍은 사진 지우기
       q  끝내기
   얼굴을 못 찾은 사진은 저장하지 않으니 다시 찍으면 된다.

4. 찍는 동안
   - 몇 장마다 조금씩 바꾼다: 고개를 살짝 돌리기, 앞뒤로 옮겨 앉기, 불 껐다 켜기.
     장수·조건은 정하지 않는다(다양한 각도·조명에서 찍은 한 묶음으로 평가).
   - 한 장마다 표정을 풀었다가 다시 짓는다. 똑같은 사진 여러 장은 의미가 없다.
   - 다섯 감정을 골고루. 찍을 때마다 감정별 장수가 나오니 적은 걸 더 찍는다.
   - 사람은 여럿(3~5명)일수록 좋다. 다음 사람은 --who 만 바꿔 다시 켠다.

5. 점수 보기 — 옛 모델(HSEmotion)과 새 모델(ko5)이 나란히 나온다
       cd emotion/train && python evaluate.py --data ~/인형사진/기본 --detect
   파이에서도 돈다(torch 불필요). 처음 한 번: pip install scikit-learn tqdm

저장 위치: ~/인형사진/기본/<감정>/<이름>_<번호>.jpg
  (--cond 로 '기본' 대신 다른 폴더에, --out 으로 다른 곳에 저장할 수 있다)

카메라 설정은 인형이 표정을 볼 때와 같다(yolo26_tracker/tracking_hailo.py 의
CAPTURE_SIZE · SENSOR_SIZE). 사진은 자르지 않고 화면 전체를 남긴다 — 자르기는 평가 때
--detect 가 인형과 똑같이 한다.
얼굴 사진이라 저장소에 올리지 않는다(.gitignore 의 *.jpg).
"""
import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

import config
import face_crop

EMOTIONS = {"1": "기쁨", "2": "슬픔", "3": "분노", "4": "불안", "5": "중립"}
# 인형 트래커가 카메라를 여는 설정과 같게 둔다(tracking_hailo.py).
CAPTURE_SIZE = (1280, 960)
SENSOR_SIZE = (1640, 1232)
HELP = "  1 기쁨  2 슬픔  3 분노  4 불안  5 중립  |  c 얼굴 확인  u 방금 것 지우기  q 끝"


class Camera:
    def __init__(self):
        from picamera2 import Picamera2   # 파이 전용
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(
            main={"format": "RGB888", "size": CAPTURE_SIZE},
            sensor={"output_size": SENSOR_SIZE},
            buffer_count=2,
        ))
        self.cam.start()
        time.sleep(1)        # 노출·화이트밸런스가 자리 잡을 시간

    def grab(self):
        # 쌓여 있던 이전 프레임을 버리고 지금 화면을 받는다.
        self.cam.capture_array()
        return self.cam.capture_array()   # picamera2 의 RGB888 은 BGR 순서 → cv2 그대로

    def close(self):
        try:
            self.cam.stop()
        except Exception:
            pass


def count_existing(cond_dir: Path, who: str):
    """감정별 전체 장수와 이 사람의 장수."""
    total, mine = Counter(), Counter()
    for emo in EMOTIONS.values():
        d = cond_dir / emo
        if not d.is_dir():
            continue
        files = [f for f in d.iterdir() if f.suffix.lower() == ".jpg"]
        total[emo] = len(files)
        mine[emo] = sum(f.name.startswith(f"{who}_") for f in files)
    return total, mine


def next_path(d: Path, who: str) -> Path:
    n = 1
    while (d / f"{who}_{n:02d}.jpg").exists():
        n += 1
    return d / f"{who}_{n:02d}.jpg"


def save_jpg(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("jpg 변환 실패")
    buf.tofile(str(path))     # 한글 경로도 되게


def face_note(faces) -> str:
    if not faces:
        return ""
    x, y, w, h, score = faces[0]
    note = f"얼굴 {w}x{h}px"
    if len(faces) > 1:
        note += f" · ⚠️ 얼굴이 {len(faces)}개 보여요(가장 큰 얼굴로 평가해요)"
    return note


def summary(total: Counter, mine: Counter, who: str):
    row = "  ".join(f"{e} {total[e]}" for e in EMOTIONS.values())
    me = "  ".join(f"{e} {mine[e]}" for e in EMOTIONS.values())
    print(f"  전체: {row}")
    print(f"  {who}: {me}")


def shoot(cam, detector, delay: float):
    if delay > 0:
        for i in range(int(delay), 0, -1):
            print(f"  {i}…", end=" ", flush=True)
            time.sleep(1)
        time.sleep(delay - int(delay))
        print("📸", flush=True)
    frame = cam.grab()
    return frame, face_crop.detect_faces(detector, frame)


def main():
    ap = argparse.ArgumentParser(description="인형 카메라로 표정 평가 사진 찍기")
    ap.add_argument("--who", required=True, help="찍히는 사람 이름 (파일 이름에 들어감)")
    ap.add_argument("--cond", default="기본", help="조건 폴더 이름 (기본, 어두움, 역광, 옆30도, 안경 …)")
    ap.add_argument("--out", default=str(Path.home() / "인형사진"), help="저장할 곳")
    ap.add_argument("--delay", type=float, default=0, help="번호를 누르고 이만큼 뒤에 찍기(초)")
    a = ap.parse_args()

    cond_dir = Path(a.out).expanduser() / a.cond
    if not os.path.exists(config.YUNET_PATH):
        sys.exit(f"❌ 얼굴 검출 모델이 없어요: {config.YUNET_PATH}")
    detector = face_crop.create_detector(
        config.YUNET_PATH, config.FACE_SCORE_THRESHOLD, config.FACE_NMS_THRESHOLD)
    try:
        cam = Camera()
    except Exception as e:
        print(f"❌ 카메라를 못 열었어요: {e}")
        sys.exit("   main.py 가 켜져 있으면 끄고 다시 해 주세요.")

    print(f"\n📷 {a.who} · 조건 '{a.cond}' → {cond_dir}")
    total, mine = count_existing(cond_dir, a.who)
    summary(total, mine, a.who)

    # 시작하자마자 얼굴이 잡히는지 한 번 본다(자리·고개 방향 확인).
    _, faces = shoot(cam, detector, 0)
    print(f"  지금 화면: {face_note(faces) or '⚠️ 얼굴이 안 보여요 — 자리나 인형 고개를 맞춰 주세요'}")
    print(HELP)

    last: Path | None = None
    try:
        while True:
            key = input("\n번호 ▶ ").strip().lower()
            if key == "q":
                break
            if key == "u":
                if last and last.exists():
                    last.unlink()
                    emo = last.parent.name
                    total[emo] -= 1
                    mine[emo] -= 1
                    print(f"  🗑  지움: {last.name} ({emo})")
                    last = None
                else:
                    print("  지울 사진이 없어요")
                continue
            if key == "c":
                _, faces = shoot(cam, detector, 0)
                print(f"  {face_note(faces) or '⚠️ 얼굴이 안 보여요'}")
                continue
            emo = EMOTIONS.get(key)
            if not emo:
                print(HELP)
                continue

            frame, faces = shoot(cam, detector, a.delay)
            if not faces:
                print(f"  ⚠️ 얼굴을 못 찾았어요 — 저장 안 함. 다시 찍어 주세요 ({emo})")
                continue
            path = next_path(cond_dir / emo, a.who)
            save_jpg(path, frame)
            last = path
            total[emo] += 1
            mine[emo] += 1
            print(f"  ✅ {emo} 저장 {path.name}  ({a.who} {mine[emo]}장 · {emo} 전체 {total[emo]}장)"
                  f"  {face_note(faces)}")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        cam.close()

    print("\n━━ 지금까지 ━━")
    summary(total, mine, a.who)
    print(f"평가: cd emotion/train && python evaluate.py --data {cond_dir} --detect")


if __name__ == "__main__":
    main()
