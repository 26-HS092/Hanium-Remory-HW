import time
from pathlib import Path

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO


# 파일 위치: 이 파이썬 파일과 같은 폴더
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "yolo26s.pt"
RESULT_PATH = PROJECT_DIR / "latest.jpg"
TEMP_PATH = PROJECT_DIR / "latest_tmp.jpg"

# 카메라 설정
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# YOLO 설정
INFERENCE_SIZE = 320
CONFIDENCE = 0.4
PERSON_CLASS = 0

# 결과 확인용 대기 시간
SAVE_INTERVAL = 1.0


def main():
    model = YOLO(str(MODEL_PATH))
    camera = Picamera2()
    camera_started = False

    try:
        config = camera.create_video_configuration(
            main={
                "format": "RGB888",
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
            },
            buffer_count=2,
        )
        camera.configure(config)
        camera.start()
        camera_started = True
        time.sleep(1)

        print("YOLO26s 사람 탐지를 시작합니다.")
        print("선택 기준: 가장 크게 보이는 사람 한 명")
        print(f"결과 이미지: {RESULT_PATH}")
        print("종료: 터미널에서 Ctrl+C", flush=True)

        while True:
            started = time.perf_counter()

            # 카메라 이미지 가져오기
            frame = camera.capture_array()

            # 사람만 탐지
            result = model.predict(
                source=frame,
                imgsz=INFERENCE_SIZE,
                conf=CONFIDENCE,
                classes=[PERSON_CLASS],
                device="cpu",
                verbose=False,
            )[0]

            annotated_frame = result.plot()

            # 화면 중앙 좌표
            height, width = frame.shape[:2]
            screen_x = width // 2
            screen_y = height // 2

            # 화면 중앙에 흰색 십자가 표시
            cv2.drawMarker(
                annotated_frame,
                (screen_x, screen_y),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                20,
                2,
            )

            person_count = len(result.boxes)

            if person_count > 0:
                boxes = result.boxes.xyxy.cpu().numpy()

                # 탐지 박스 면적이 가장 큰 사람 선택
                areas = (
                    (boxes[:, 2] - boxes[:, 0])
                    * (boxes[:, 3] - boxes[:, 1])
                )
                target_index = int(areas.argmax())

                x1, y1, x2, y2 = map(
                    int, boxes[target_index]
                )

                # 대상 중심 좌표
                target_x = (x1 + x2) // 2
                target_y = (y1 + y2) // 2

                # 화면 중앙과의 차이
                # X: 음수=왼쪽, 양수=오른쪽
                # Y: 음수=위쪽, 양수=아래쪽
                error_x = target_x - screen_x
                error_y = target_y - screen_y

                # 선택된 사람을 노란색 박스로 강조
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    3,
                )

                cv2.putText(
                    annotated_frame,
                    "TARGET",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                # 대상 중심에 빨간 점 표시
                cv2.circle(
                    annotated_frame,
                    (target_x, target_y),
                    6,
                    (0, 0, 255),
                    -1,
                )

                # 화면 중앙에서 대상까지 선 표시
                cv2.line(
                    annotated_frame,
                    (screen_x, screen_y),
                    (target_x, target_y),
                    (0, 255, 255),
                    2,
                )

                status = (
                    f"Target: ({target_x}, {target_y}) "
                    f"Error: ({error_x}, {error_y})"
                )

                print(
                    f"{time.strftime('%H:%M:%S')} | "
                    f"사람 {person_count}명 | "
                    f"선택 대상 중심: ({target_x}, {target_y}) | "
                    f"중앙과 차이: X={error_x}, Y={error_y}",
                    flush=True,
                )

            else:
                status = "No person detected"

                print(
                    f"{time.strftime('%H:%M:%S')} | "
                    "사람 없음 — 추적 대기",
                    flush=True,
                )

            elapsed = time.perf_counter() - started

            # 처리 시간 표시
            cv2.putText(
                annotated_frame,
                f"Detection: {elapsed * 1000:.0f} ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # 대상 좌표와 중앙 오차 표시
            cv2.putText(
                annotated_frame,
                status,
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

            # 결과 이미지 저장
            if not cv2.imwrite(
                str(TEMP_PATH), annotated_frame
            ):
                raise RuntimeError(
                    "결과 이미지 저장에 실패했습니다."
                )

            TEMP_PATH.replace(RESULT_PATH)

            time.sleep(SAVE_INTERVAL)

    except KeyboardInterrupt:
        print("\n객체 탐지를 종료합니다.")

    finally:
        if camera_started:
            camera.stop()
        camera.close()


if __name__ == "__main__":
    main()