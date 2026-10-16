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
# Hailo-10H용 포즈 모델: 사람 박스 + 관절 17개(코·눈·귀 포함) 위치
MODEL_PATH = Path("/usr/share/hailo-models/yolov8m_pose_h10.hef")
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

# 카메라 (IMX219 = Pi Camera v2)
# 640x480 센서 모드는 가운데만 잘라 써서 화각이 24°밖에 안 된다.
# 전체 화면 모드(1640x1232)를 받아 줄여 써야 원래 화각 62°가 나온다.
FRAME_SIZE = (640, 480)
SENSOR_SIZE = (1640, 1232)
HFOV = 62.2         # 가로 화각 (도)
VFOV = 48.8         # 세로 화각 (도)
DEG_PER_PX_X = HFOV / FRAME_SIZE[0]
DEG_PER_PX_Y = VFOV / FRAME_SIZE[1]

# 추적 설정
DEAD_ZONE = 15       # 중앙 부근(픽셀)에서는 움직이지 않음
KP = 0.4             # 한 프레임에 남은 각도 오차의 몇 %를 줄일지
MAX_SPEED = 90.0     # 최대 회전 속도 (도/초)
SMOOTHING = 0.6      # 작을수록 부드럽지만 반응이 느림
HEAD_Y_RATIO = 0.1   # 얼굴·어깨가 안 보일 때 박스 위에서 이 비율 지점을 머리로 봄

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

# 화면 밖으로 나갈 때 따라가기
EDGE_MARGIN = 5       # 박스가 화면 끝에서 이 픽셀 안이면 잘린 것으로 봄
EDGE_SIDE_RATIO = 0.25  # 중앙에서 화면 너비 x 이 값보다 멀리서 놓치면 그쪽으로 나간 것
CHASE_SPEED = 60.0    # 놓친 쪽으로 계속 돌아가는 속도 (도/초)

# 여러 명일 때 한 사람 고정 설정
LOCK_TIMEOUT = 1.0    # 고정한 사람이 이 시간(초) 동안 안 보이면 다른 사람으로 바꿈
MATCH_GATE = 1.0      # 이전 위치에서 (박스 너비 x 이 값) 넘게 떨어진 박스는 다른 사람
HIST_WEIGHT = 0.5     # 옷 색 유사도를 얼마나 믿을지 (사람끼리 겹칠 때 구분용)
MIN_HIST_SIM = 0.3    # 여러 명일 때 옷 색이 이보다 다르면 대상이 아니라고 봄

# 탐지 설정
CONF_THRESHOLD = 0.4
NMS_IOU = 0.6
REG_MAX = 15            # 박스 회귀 분포 길이 (YOLOv8)
KEYPOINT_CONF = 0.5     # 관절이 이 확률 이상이면 보인다고 봄
FACE_KEYPOINTS = [0, 1, 2]        # 코, 왼눈, 오른눈
EAR_KEYPOINTS = [3, 4]            # 왼귀, 오른귀 (옆모습일 때만 사용)
SHOULDER_KEYPOINTS = [5, 6]

# 브라우저로 카메라 화면 보기: http://<라즈베리파이 IP>:8000
STREAM_PORT = 8000
STREAM_JPEG_QUALITY = 70

# 결과 사진은 1초마다 저장
IMAGE_SAVE_INTERVAL = 1.0


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


class HailoPoseDetector:
    """HAT(Hailo NPU)에서 사람 박스와 관절 위치를 찾는다."""

    def __init__(self, hef_path):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        # 다른 프로그램(STT 등)과 HAT 공유
        params.group_id = "SHARED"

        self.vdevice = VDevice(params)
        self.model = self.vdevice.create_infer_model(str(hef_path))
        self.model.input().set_format_type(FormatType.UINT8)
        for output in self.model.outputs:
            output.set_format_type(FormatType.FLOAT32)

        self.input_h, self.input_w = self.model.input().shape[:2]

        # 출력은 격자 크기(80/40/20)마다 박스·점수·관절 3개씩
        self.heads = {}
        for output in self.model.outputs:
            grid_h, grid_w, channels = output.shape
            kind = {(REG_MAX + 1) * 4: "box", 1: "score", 51: "kpt"}[channels]
            self.heads.setdefault((grid_h, grid_w), {})[kind] = output.name

        self.config_ctx = self.model.configure()
        self.configured = self.config_ctx.__enter__()

        self.output_buffers = {
            output.name: np.empty(output.shape, dtype=np.float32)
            for output in self.model.outputs
        }
        self.bindings = self.configured.create_bindings(
            output_buffers=self.output_buffers
        )

    def detect(self, frame_bgr):
        """원본 좌표의 사람 박스 (N, 4), 점수 (N,), 관절 (N, 17, 3=x,y,확률)."""
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

        all_boxes, all_scores, all_kpts = [], [], []

        for (grid_h, grid_w), names in self.heads.items():
            stride = self.input_h // grid_h
            scores = self.output_buffers[names["score"]].reshape(-1)
            keep = np.flatnonzero(scores >= CONF_THRESHOLD)
            if keep.size == 0:
                continue

            grid_y, grid_x = np.divmod(keep, grid_w)

            # 박스: 칸 중심에서 좌/상/우/하 거리의 분포 → 기댓값
            dist = self.output_buffers[names["box"]].reshape(
                -1, 4, REG_MAX + 1
            )[keep]
            dist = np.exp(dist - dist.max(axis=-1, keepdims=True))
            dist /= dist.sum(axis=-1, keepdims=True)
            dist = (dist @ np.arange(REG_MAX + 1)) * stride

            center_x = (grid_x + 0.5) * stride
            center_y = (grid_y + 0.5) * stride
            all_boxes.append(np.stack([
                center_x - dist[:, 0],
                center_y - dist[:, 1],
                center_x + dist[:, 2],
                center_y + dist[:, 3],
            ], axis=1))
            all_scores.append(scores[keep])

            kpts = self.output_buffers[names["kpt"]].reshape(
                -1, 17, 3
            )[keep].copy()
            kpts[..., 0] = (kpts[..., 0] * 2 + grid_x[:, None]) * stride
            kpts[..., 1] = (kpts[..., 1] * 2 + grid_y[:, None]) * stride
            kpts[..., 2] = 1 / (1 + np.exp(-kpts[..., 2]))
            all_kpts.append(kpts)

        if not all_boxes:
            return np.empty((0, 4)), np.empty(0), np.empty((0, 17, 3))

        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        kpts = np.concatenate(all_kpts)

        # 겹치는 박스 정리 (NMS)
        keep = cv2.dnn.NMSBoxes(
            np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]).tolist(),
            scores.tolist(),
            CONF_THRESHOLD,
            NMS_IOU,
        )
        keep = np.array(keep, dtype=int).reshape(-1)
        boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]

        # 모델 입력 좌표 → 원본 좌표
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / scale).clip(0, width - 1)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / scale).clip(0, height - 1)
        kpts[..., 0] = (kpts[..., 0] - pad_x) / scale
        kpts[..., 1] = (kpts[..., 1] - pad_y) / scale

        return boxes, scores, kpts

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


def correction(error_px, deg_per_px, dt):
    """픽셀 오차를 각도로 바꿔 이번 프레임에 움직일 각도를 정한다."""
    if abs(error_px) <= DEAD_ZONE:
        return 0.0

    remaining = (abs(error_px) - DEAD_ZONE) * deg_per_px
    step = min(remaining * KP, MAX_SPEED * dt)

    return step if error_px > 0 else -step


def face_point(box, keypoints):
    """얼굴 중심 (보이는 코·눈의 평균, 둘 다 안 보이면 귀).

    얼굴이 안 보이면(뒤돌았거나 화면 밖) 어깨나 박스로 머리 위치를 어림한다.
    두 번째 값은 얼굴을 실제로 찾았는지 여부.
    """
    # 귀는 한쪽만 보이면 중심이 그쪽으로 치우치므로 코·눈을 먼저 씀
    for indices in (FACE_KEYPOINTS, EAR_KEYPOINTS):
        face = keypoints[indices]
        visible = face[face[:, 2] >= KEYPOINT_CONF]
        if len(visible) > 0:
            return (visible[:, 0].mean(), visible[:, 1].mean()), True

    shoulders = keypoints[SHOULDER_KEYPOINTS]
    if (shoulders[:, 2] >= KEYPOINT_CONF).all():
        shoulder_w = abs(shoulders[0, 0] - shoulders[1, 0])
        return (
            shoulders[:, 0].mean(),
            shoulders[:, 1].mean() - shoulder_w * 0.6,
        ), False

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y1 + (y2 - y1) * HEAD_Y_RATIO), False


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - inter
    )
    return inter / union if union > 0 else 0.0


def torso_histogram(frame, box):
    """상체 가운데 부분의 색 분포 (옷 색으로 사람 구분)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    crop = frame[
        int(y1 + h * 0.2):int(y1 + h * 0.6),
        int(x1 + w * 0.25):int(x2 - w * 0.25),
    ]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


class TargetLock:
    """한 사람을 고르면, 다른 사람이 더 크게 보여도 그 사람만 계속 따라간다."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.box = None
        self.hist = None
        self.last_match = 0.0

    def update(self, frame, boxes, now):
        """이번 프레임에서 고정 대상의 인덱스를 반환 (안 보이면 None)."""
        if self.box is not None and now - self.last_match > LOCK_TIMEOUT:
            self.reset()

        if len(boxes) == 0:
            return None

        if self.box is None:
            # 새로 고를 때는 가장 크게 보이는(가까운) 사람
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            index = int(areas.argmax())
        else:
            index = self._match(frame, boxes)
            if index is None:
                return None

        # 다른 사람과 겹쳐 있으면 옷 색이 섞이므로 기억을 갱신하지 않음
        overlapped = any(
            box_iou(boxes[index], other) > 0
            for i, other in enumerate(boxes) if i != index
        )
        hist = None if overlapped else torso_histogram(frame, boxes[index])
        if hist is not None:
            if self.hist is None:
                self.hist = hist
            else:
                self.hist = 0.8 * self.hist + 0.2 * hist

        self.box = boxes[index].copy()
        self.last_match = now
        return index

    def _match(self, frame, boxes):
        px1, py1, px2, py2 = self.box
        prev_w = max(px2 - px1, 60.0)
        prev_cx, prev_cy = (px1 + px2) / 2, (py1 + py2) / 2

        best_index = None
        best_score = -np.inf

        for index, box in enumerate(boxes):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            distance = np.hypot(cx - prev_cx, cy - prev_cy) / prev_w
            # 한 명만 보이면 헷갈릴 사람이 없으므로
            # (빠르게 움직였거나 카메라가 돌며 쫓아간 경우)
            # 위치가 많이 달라도 옷 색만 맞으면 같은 사람으로 봄
            relaxed = len(boxes) == 1
            if distance > MATCH_GATE and not relaxed:
                continue

            if relaxed and self.hist is not None:
                hist = torso_histogram(frame, box)
                if hist is not None and cv2.compareHist(
                    self.hist, hist, cv2.HISTCMP_CORREL
                ) < MIN_HIST_SIM:
                    continue

            score = box_iou(self.box, box) - 0.5 * distance

            # 후보가 여럿이면 옷 색으로도 비교
            if len(boxes) > 1 and self.hist is not None:
                hist = torso_histogram(frame, box)
                if hist is not None:
                    similarity = cv2.compareHist(
                        self.hist, hist, cv2.HISTCMP_CORREL
                    )
                    if similarity < MIN_HIST_SIM:
                        continue
                    score += HIST_WEIGHT * similarity

            if score > best_score:
                best_score = score
                best_index = index

        return best_index


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
    lost_side = 0         # 대상이 마지막으로 있던 쪽 (-1 왼쪽, 1 오른쪽, 0 가운데)
    lock = TargetLock()
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

        detector = HailoPoseDetector(MODEL_PATH)
        stream = FrameStream(STREAM_PORT)

        camera = Picamera2()
        camera.configure(
            camera.create_video_configuration(
                main={
                    "format": "RGB888",
                    "size": FRAME_SIZE,
                },
                sensor={"output_size": SENSOR_SIZE},
                buffer_count=2,
            )
        )

        camera.start()
        camera_started = True
        time.sleep(1)

        print("사람 추적 시작 (Hailo HAT)")
        print("대상: 가장 크게 보이는 사람 한 명의 얼굴 (고르면 고정)")
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

            boxes, scores, keypoints = detector.detect(frame)

            height, width = frame.shape[:2]
            screen_x = width // 2
            screen_y = height // 2

            target_box = None
            target_point = None
            target_face = None

            if len(boxes) > 0:
                seen_frames += 1
                last_seen = now
            else:
                seen_frames = 0

            if mode == "scan" and seen_frames >= CONFIRM_FRAMES:
                mode = "track"
                smooth_x = None
                smooth_y = None
                lock.reset()
                print("사람 발견: 추적 시작", flush=True)

            elif mode == "track" and now - last_seen > LOST_TIMEOUT:
                mode = "scan"
                # 나간 쪽부터 찾아봄
                if lost_side != 0:
                    scan_direction = PAN_DIRECTION * lost_side
                print("사람 놓침: 다시 탐색", flush=True)

            target_index = None
            if mode == "track":
                target_index = lock.update(frame, boxes, now)

            if target_index is not None:
                x1, y1, x2, y2 = map(
                    int, boxes[target_index]
                )

                (target_x, target_y), face_found = face_point(
                    boxes[target_index], keypoints[target_index]
                )

                if face_found:
                    target_face = keypoints[target_index][
                        FACE_KEYPOINTS + EAR_KEYPOINTS
                    ]
                else:
                    # 얼굴이 안 보이는데 박스가 화면 끝에 걸려 잘렸으면
                    # 머리가 그 바깥에 있으므로 잘린 쪽 끝을 조준해서 더 빨리 돌아감
                    at_left = x1 <= EDGE_MARGIN
                    at_right = x2 >= width - 1 - EDGE_MARGIN
                    if at_left and not at_right:
                        target_x = x1
                    elif at_right and not at_left:
                        target_x = x2
                    if y1 <= EDGE_MARGIN:
                        target_y = y1

                if abs(target_x - screen_x) > width * EDGE_SIDE_RATIO:
                    lost_side = 1 if target_x > screen_x else -1
                else:
                    lost_side = 0

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
                    pan + PAN_DIRECTION * correction(
                        error_x, DEG_PER_PX_X, dt
                    ),
                    PAN_MIN,
                    PAN_MAX,
                )

                tilt = clamp(
                    tilt + TILT_DIRECTION * correction(
                        error_y, DEG_PER_PX_Y, dt
                    ),
                    TILT_MIN,
                    TILT_MAX,
                )

                target_box = (x1, y1, x2, y2)
                target_point = (
                    round(smooth_x),
                    round(smooth_y),
                )

                target_status = (
                    f"{'얼굴' if face_found else '얼굴 추정'} "
                    f"오차 X={error_x:.0f}, "
                    f"Y={error_y:.0f}"
                )

            elif mode == "track":
                # 잠깐 안 보이면 마지막 각도 유지
                # (다른 사람만 보여도 LOCK_TIMEOUT 동안은 넘어가지 않음)
                smooth_x = None
                smooth_y = None

                if lost_side != 0:
                    # 화면 밖으로 나갔으면 나간 쪽으로 계속 돌아감
                    pan = clamp(
                        pan + PAN_DIRECTION * lost_side * CHASE_SPEED * dt,
                        PAN_MIN,
                        PAN_MAX,
                    )
                    side = "오른쪽" if lost_side > 0 else "왼쪽"
                    target_status = f"대상 놓침: {side}으로 따라감"
                else:
                    target_status = "대상 안 보임: 각도 유지"

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

                if target_face is not None:
                    for fx, fy, conf in target_face:
                        if conf >= KEYPOINT_CONF:
                            cv2.circle(
                                image,
                                (round(fx), round(fy)),
                                3,
                                (0, 255, 0),
                                -1,
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

            # capture_array()가 다음 프레임까지 기다리므로 따로 쉬지 않음

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