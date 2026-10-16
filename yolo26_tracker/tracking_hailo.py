import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import serial
from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
from picamera2 import Picamera2


PROJECT_DIR = Path(__file__).resolve().parent
# Hailo-10H용 COCO 모델 (NMS 포함, 사람 = 클래스 0)
MODEL_PATH = Path("/usr/share/hailo-models/yolov8m_h10.hef")
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

# 탐색 설정 (사람이 없을 때 좌우로 돌며 찾기)
# 범위를 넓히려면 SCAN_MIN/SCAN_MAX와 PAN_MIN/PAN_MAX를 함께 바꿀 것
SCAN_MIN = PAN_MIN
SCAN_MAX = PAN_MAX
SCAN_SPEED = 30.0     # 탐색 회전 속도 (도/초)
CONFIRM_FRAMES = 2    # 연속 몇 프레임 보여야 사람으로 확정할지
LOST_TIMEOUT = 2.0    # 추적 중 이 시간(초) 동안 안 보이면 다시 탐색

# 탐지 설정
CONF_THRESHOLD = 0.4
PERSON_CLASS = 0

# 브라우저로 카메라 화면 보기: http://<라즈베리파이 IP>:8000
STREAM_PORT = 8000
STREAM_JPEG_QUALITY = 70

# 결과 사진은 1초마다 저장
IMAGE_SAVE_INTERVAL = 1.0


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


class HailoPersonDetector:
    """HAT(Hailo NPU)에서 사람 박스를 찾는다."""

    def __init__(self, hef_path):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        # 다른 프로그램(STT 등)과 HAT 공유
        params.group_id = "SHARED"

        self.vdevice = VDevice(params)
        self.model = self.vdevice.create_infer_model(str(hef_path))
        self.model.input().set_format_type(FormatType.UINT8)
        self.model.output().set_format_type(FormatType.FLOAT32)

        self.input_h, self.input_w = self.model.input().shape[:2]
        self.output_name = self.model.output().name

        self.config_ctx = self.model.configure()
        self.configured = self.config_ctx.__enter__()

        self.output_buffer = np.empty(
            self.model.output().shape, dtype=np.float32
        )
        self.bindings = self.configured.create_bindings(
            output_buffers={self.output_name: self.output_buffer}
        )

    def detect(self, frame_bgr):
        """사람 박스 목록을 원본 좌표 (x1, y1, x2, y2) 배열로 반환."""
        height, width = frame_bgr.shape[:2]

        # 비율 유지 리사이즈 + 여백 채우기 (letterbox)
        scale = min(self.input_w / width, self.input_h / height)
        new_w, new_h = int(width * scale), int(height * scale)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2

        resized = cv2.resize(frame_bgr, (new_w, new_h))
        canvas = np.full(
            (self.input_h, self.input_w, 3), 114, dtype=np.uint8
        )
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        self.bindings.input().set_buffer(np.ascontiguousarray(canvas))
        self.configured.run([self.bindings], 10000)

        # 클래스별 리스트, 각 행 = [ymin, xmin, ymax, xmax, score] (0~1)
        persons = self.bindings.output().get_buffer()[PERSON_CLASS]
        if len(persons) == 0:
            return np.empty((0, 4)), np.empty(0)

        persons = persons[persons[:, 4] >= CONF_THRESHOLD]

        x1 = (persons[:, 1] * self.input_w - pad_x) / scale
        y1 = (persons[:, 0] * self.input_h - pad_y) / scale
        x2 = (persons[:, 3] * self.input_w - pad_x) / scale
        y2 = (persons[:, 2] * self.input_h - pad_y) / scale

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height - 1)

        return boxes, persons[:, 4]

    def close(self):
        self.config_ctx.__exit__(None, None, None)
        self.vdevice.release()


class FrameStream:
    """최신 화면을 MJPEG로 브라우저에 보낸다."""

    def __init__(self, port):
        self.lock = threading.Condition()
        self.jpeg = None

        stream = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    body = (
                        "<html><body style='margin:0;background:#000'>"
                        "<img src='/stream' style='width:100%;height:auto'>"
                        "</body></html>"
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path != "/stream":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()

                try:
                    while True:
                        with stream.lock:
                            stream.lock.wait(timeout=1.0)
                            jpeg = stream.jpeg

                        if jpeg is None:
                            continue

                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                            + jpeg
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(
            target=self.server.serve_forever, daemon=True
        ).start()

    def update(self, image):
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY]
        )
        if not ok:
            return

        with self.lock:
            self.jpeg = encoded.tobytes()
            self.lock.notify_all()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


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
    detector = None
    stream = None

    camera = None
    camera_started = False
    arduino = None

    pan = float(PAN_CENTER)
    tilt = float(TILT_CENTER)

    smooth_x = None
    smooth_y = None

    # "scan": 좌우로 돌며 찾기, "track": 찾은 사람 추적
    mode = "scan"
    scan_direction = 1
    seen_frames = 0
    last_seen = 0.0

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

        detector = HailoPersonDetector(MODEL_PATH)
        stream = FrameStream(STREAM_PORT)

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

        print("사람 추적 시작 (Hailo HAT)")
        print("대상: 가장 크게 보이는 사람 한 명")
        print(f"탐색 범위: PAN {SCAN_MIN}~{SCAN_MAX}")
        print(f"결과 이미지: {RESULT_PATH}")
        print(f"카메라 화면: http://<라즈베리파이 IP>:{STREAM_PORT}")
        print("종료: Ctrl+C", flush=True)

        last_loop = time.monotonic()

        while True:
            started = time.perf_counter()

            now = time.monotonic()
            dt = now - last_loop
            last_loop = now

            frame = camera.capture_array()

            boxes, scores = detector.detect(frame)

            height, width = frame.shape[:2]
            screen_x = width // 2
            screen_y = height // 2

            target_box = None
            target_point = None

            if len(boxes) > 0:
                seen_frames += 1
                last_seen = now
            else:
                seen_frames = 0

            if mode == "scan" and seen_frames >= CONFIRM_FRAMES:
                mode = "track"
                smooth_x = None
                smooth_y = None
                print("사람 발견: 추적 시작", flush=True)

            elif mode == "track" and now - last_seen > LOST_TIMEOUT:
                mode = "scan"
                print("사람 놓침: 다시 탐색", flush=True)

            if mode == "track" and len(boxes) > 0:
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

            elif mode == "track":
                # 잠깐 안 보이면 마지막 각도 유지
                smooth_x = None
                smooth_y = None
                target_status = "사람 없음: 각도 유지"

            else:
                # 좌우 끝까지 돌고 방향 전환, 틸트는 중앙
                pan += scan_direction * SCAN_SPEED * dt

                if pan >= SCAN_MAX:
                    pan = SCAN_MAX
                    scan_direction = -1
                elif pan <= SCAN_MIN:
                    pan = SCAN_MIN
                    scan_direction = 1

                tilt = float(TILT_CENTER)
                target_status = "탐색 중"

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

            # 화면 그리기 (스트리밍용, 매 프레임)
            image = frame.copy()

            for box, score in zip(boxes, scores):
                bx1, by1, bx2, by2 = map(int, box)
                cv2.rectangle(
                    image, (bx1, by1), (bx2, by2), (255, 0, 0), 2
                )
                cv2.putText(
                    image,
                    f"person {score:.2f}",
                    (bx1, max(15, by1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

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
                f"{mode.upper()}  PAN: {round(pan)}  TILT: {round(tilt)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            stream.update(image)

            # 결과 이미지 저장은 1초마다 수행
            if now - last_image_save >= IMAGE_SAVE_INTERVAL:
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

        if stream is not None:
            stream.close()

        if detector is not None:
            detector.close()

        if arduino is not None:
            arduino.close()

        print("서보는 마지막 위치를 유지합니다.")


if __name__ == "__main__":
    main()