import time
from pathlib import Path

import cv2
import serial
from picamera2 import Picamera2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "yolo26s.pt"
RESULT_PATH = PROJECT_DIR / "latest.jpg"
TEMP_PATH = PROJECT_DIR / "latest_tmp.jpg"

# 아두이노 통신
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# 측정한 중앙값과 안전 범위
PAN_CENTER = 84
PAN_MIN = 44
PAN_MAX = 124

TILT_CENTER = 108
TILT_MIN = 88
TILT_MAX = 158

# 추적 설정
DEAD_ZONE = 35       # 중앙 부근에서는 움직이지 않음
GAIN = 0.03         # 오차에 따른 이동량
MAX_STEP = 3.0      # 한 번에 이동할 최대 각도
SMOOTHING = 0.4     # 작을수록 부드럽지만 반응이 느림

# 방향이 반대라면 해당 값을 -1로 변경
PAN_DIRECTION = 1
TILT_DIRECTION = -1

# 결과 사진은 1초마다 저장
IMAGE_SAVE_INTERVAL = 1.0


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def correction(error):
    if abs(error) <= DEAD_ZONE:
        return 0.0

    remaining = abs(error) - DEAD_ZONE
    step = min(remaining * GAIN, MAX_STEP)

    return step if error > 0 else -step


def send_angles(arduino, pan, tilt):
    command = f"{round(pan)},{round(tilt)}"

    arduino.write((command + "\n").encode("ascii"))
    arduino.flush()

    reply = arduino.readline().decode(
        errors="replace"
    ).strip()

    success = reply == f"RECEIVED: {command}"
    return command, reply, success


def main():
    model = YOLO(str(MODEL_PATH))

    camera = None
    camera_started = False
    arduino = None

    pan = float(PAN_CENTER)
    tilt = float(TILT_CENTER)

    smooth_x = None
    smooth_y = None

    last_image_save = 0.0

    try:
        arduino = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=1,
            write_timeout=1,
        )

        # 아두이노 재시작 대기
        time.sleep(2)
        arduino.reset_input_buffer()

        # 시작 시 중앙 위치 명령
        command, reply, success = send_angles(
            arduino, pan, tilt
        )

        if not success:
            raise RuntimeError(
                f"중앙 명령 응답 이상: {reply!r}"
            )

        print(f"중앙 위치 설정: {command}", flush=True)
        time.sleep(1)

        camera = Picamera2()
        camera.configure(
            camera.create_video_configuration(
                main={
                    "format": "RGB888",
                    "size": (640, 480),
                },
                buffer_count=2,
            )
        )

        camera.start()
        camera_started = True
        time.sleep(1)

        print("사람 추적 시작")
        print("대상: 가장 크게 보이는 사람 한 명")
        print(f"결과 이미지: {RESULT_PATH}")
        print("종료: Ctrl+C", flush=True)

        while True:
            started = time.perf_counter()

            frame = camera.capture_array()

            result = model.predict(
                source=frame,
                imgsz=320,
                conf=0.4,
                classes=[0],
                device="cpu",
                verbose=False,
            )[0]

            height, width = frame.shape[:2]
            screen_x = width // 2
            screen_y = height // 2

            target_box = None
            target_point = None

            if len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()

                # 가장 큰 사람 선택
                areas = (
                    (boxes[:, 2] - boxes[:, 0])
                    * (boxes[:, 3] - boxes[:, 1])
                )
                target_index = int(areas.argmax())

                x1, y1, x2, y2 = map(
                    int, boxes[target_index]
                )

                target_x = (x1 + x2) // 2
                target_y = (y1 + y2) // 2

                # 탐지 중심 좌표의 흔들림 완화
                if smooth_x is None:
                    smooth_x = float(target_x)
                    smooth_y = float(target_y)
                else:
                    smooth_x += SMOOTHING * (
                        target_x - smooth_x
                    )
                    smooth_y += SMOOTHING * (
                        target_y - smooth_y
                    )

                error_x = smooth_x - screen_x
                error_y = smooth_y - screen_y

                pan = clamp(
                    pan + PAN_DIRECTION * correction(error_x),
                    PAN_MIN,
                    PAN_MAX,
                )

                tilt = clamp(
                    tilt + TILT_DIRECTION * correction(error_y),
                    TILT_MIN,
                    TILT_MAX,
                )

                target_box = (x1, y1, x2, y2)
                target_point = (
                    round(smooth_x),
                    round(smooth_y),
                )

                target_status = (
                    f"오차 X={error_x:.0f}, "
                    f"Y={error_y:.0f}"
                )

            else:
                # 사람이 없으면 마지막 각도 유지
                smooth_x = None
                smooth_y = None
                target_status = "사람 없음: 각도 유지"

            # 계산한 각도 전송
            command, reply, success = send_angles(
                arduino, pan, tilt
            )

            if not success:
                # 통신이 불확실한 상태에서는 추적 중단
                raise RuntimeError(
                    f"아두이노 응답 이상: {reply!r}"
                )

            elapsed = time.perf_counter() - started

            print(
                f"{target_status} | "
                f"전송={command} | 수신 성공 | "
                f"{elapsed * 1000:.0f}ms",
                flush=True,
            )

            # 그림 생성과 저장은 1초마다 수행
            now = time.monotonic()

            if now - last_image_save >= IMAGE_SAVE_INTERVAL:
                image = result.plot()

                cv2.drawMarker(
                    image,
                    (screen_x, screen_y),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )

                cv2.rectangle(
                    image,
                    (
                        screen_x - DEAD_ZONE,
                        screen_y - DEAD_ZONE,
                    ),
                    (
                        screen_x + DEAD_ZONE,
                        screen_y + DEAD_ZONE,
                    ),
                    (255, 255, 255),
                    1,
                )

                if target_box is not None:
                    x1, y1, x2, y2 = target_box

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 255),
                        3,
                    )

                    cv2.putText(
                        image,
                        "TARGET",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    cv2.circle(
                        image,
                        target_point,
                        6,
                        (0, 0, 255),
                        -1,
                    )

                    cv2.line(
                        image,
                        (screen_x, screen_y),
                        target_point,
                        (0, 255, 255),
                        2,
                    )

                cv2.putText(
                    image,
                    f"PAN: {round(pan)}  TILT: {round(tilt)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if not cv2.imwrite(str(TEMP_PATH), image):
                    raise RuntimeError("결과 이미지 저장 실패")

                TEMP_PATH.replace(RESULT_PATH)
                last_image_save = now

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n추적 종료")

    finally:
        if camera is not None:
            if camera_started:
                camera.stop()
            camera.close()

        if arduino is not None:
            arduino.close()

        print("서보는 마지막 위치를 유지합니다.")


if __name__ == "__main__":
    main()