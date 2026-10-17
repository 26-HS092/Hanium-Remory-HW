import signal
import threading
import time
from collections import deque
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
# 감정 인식이 얼굴을 선명하게 자르도록 CAPTURE_SIZE로 찍고,
# 추적은 FRAME_SIZE로 줄인 화면으로 한다.
CAPTURE_SIZE = (1280, 960)
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
# (AUTO_CALIBRATE가 켜져 있으면 시작할 때 직접 재서 덮어씀)
PAN_DIRECTION = 1
TILT_DIRECTION = -1

# 시작할 때 서보를 살짝 돌려 화면이 밀리는 방향·거리로
# 방향과 "서보 1도 = 화면 몇 픽셀"을 직접 잰다.
# 카메라 앞이 무늬 없는 벽·천장이면 측정이 안 돼서 위 설정값을 그대로 씀
AUTO_CALIBRATE = True
CALIBRATE_STEP = 8          # 시험으로 돌려볼 각도
CALIBRATE_MIN_INLIERS = 40  # 이만큼 특징점이 같은 방향으로 움직여야 믿음

# 탐색 설정 (사람이 없을 때 좌우로 돌며 찾기)
# 범위를 넓히려면 SCAN_MIN/SCAN_MAX와 PAN_MIN/PAN_MAX를 함께 바꿀 것
SCAN_MIN = PAN_MIN
SCAN_MAX = PAN_MAX
SCAN_SPEED = 30.0     # 탐색 회전 속도 (도/초)
CONFIRM_FRAMES = 2    # 연속 몇 프레임 보여야 사람으로 확정할지
LOST_TIMEOUT = 2.0    # 추적 중 이 시간(초) 동안 안 보이면 다시 탐색
SCAN_GIVEUP = 10.0    # 이 시간(초) 동안 찾아도 없으면 정면에서 쉰다 (보이면 바로 다시 추적)

# 놓쳤을 때 사라진 쪽으로 돌아보기
EDGE_MARGIN = 5       # 박스가 화면 끝에서 이 픽셀 안이면 잘린 것으로 봄
HISTORY_TIME = 0.5    # 최근 이 시간(초) 동안의 얼굴 위치로 움직이던 속도를 구함
LEAD_TIME = 0.7       # 사라진 뒤 그 속도로 이 시간(초)만큼 더 간 곳을 찾아봄
MAX_LEAD = 25.0       # 속도로 예측해서 더 가 보는 최대 각도
CHASE_SPEED = 60.0    # 그곳으로 돌아보는 속도 (도/초)

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


def measure_shift(before, after):
    """두 화면 사이의 평행 이동(픽셀)과 그걸 지지하는 특징점 수."""
    orb = cv2.ORB_create(1500)
    gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in (before, after)]
    kp1, des1 = orb.detectAndCompute(gray[0], None)
    kp2, des2 = orb.detectAndCompute(gray[1], None)
    if des1 is None or des2 is None:
        return None, 0

    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
    if len(matches) < CALIBRATE_MIN_INLIERS:
        return None, 0

    src = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst = np.float32([kp2[m.trainIdx].pt for m in matches])
    affine, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0
    )
    if affine is None:
        return None, 0

    return affine[:, 2], int(inliers.sum())


def calibrate(grab, arduino):
    """서보를 살짝 돌려 보고 방향과 서보 1도당 픽셀 수를 잰다.

    반환: (pan_dir, tilt_dir, pan_deg_per_px, tilt_deg_per_px)
    측정이 안 된 축은 설정값(PAN_DIRECTION 등)을 그대로 쓴다.
    """
    result = [PAN_DIRECTION, TILT_DIRECTION, DEG_PER_PX_X, DEG_PER_PX_Y]

    def shot(pan, tilt):
        command, reply, success = send_angles(arduino, pan, tilt)
        if not success:
            raise RuntimeError(f"보정 중 아두이노 응답 이상: {reply!r}")
        time.sleep(0.6)
        for _ in range(3):  # 서보가 멈춘 뒤 프레임을 받도록 버퍼 비우기
            frame = grab()
        return frame

    tests = [
        # (축 이름, 결과 위치, 움직인 각도, 화면 축)
        ("PAN", 0, (PAN_CENTER + CALIBRATE_STEP, TILT_CENTER), 0),
        ("TILT", 1, (PAN_CENTER, TILT_CENTER + CALIBRATE_STEP), 1),
    ]

    for name, index, moved, axis in tests:
        before = shot(PAN_CENTER, TILT_CENTER)
        after = shot(*moved)
        shift, inliers = measure_shift(before, after)

        if shift is None or inliers < CALIBRATE_MIN_INLIERS:
            print(f"{name} 보정 실패 (특징점 부족): 설정값 사용", flush=True)
            continue

        px = shift[axis]
        px_per_deg = abs(px) / CALIBRATE_STEP
        if abs(shift[1 - axis]) > abs(px):
            print(
                f"{name} 보정 실패 (화면이 엉뚱한 방향으로 밀림 "
                f"x={shift[0]:.0f}, y={shift[1]:.0f}px): "
                "카메라가 돌아가 달렸는지 확인. 설정값 사용",
                flush=True,
            )
            continue
        if px_per_deg < 3:
            print(
                f"{name} 보정 실패 (화면이 거의 안 움직임 {px:.0f}px): "
                "카메라가 거치대에 달려 있는지 확인. 설정값 사용",
                flush=True,
            )
            continue

        # 서보 각도를 +로 바꿨을 때 화면이 +쪽(오른쪽/아래)으로 밀리면
        # 대상이 오른쪽/아래에 있을 때 각도를 -로 바꿔야 가운데로 온다
        result[index] = -1 if px > 0 else 1
        result[index + 2] = 1 / px_per_deg
        print(
            f"{name} 보정: 방향 {result[index]:+d}, "
            f"서보 1도 = {px_per_deg:.1f}px (특징점 {inliers}개)",
            flush=True,
        )

    shot(PAN_CENTER, TILT_CENTER)
    return tuple(result)


def predict_lost_goal(history):
    """최근 얼굴 각도 기록으로 사라진 사람이 지금쯤 있을 (pan, tilt)를 구한다."""
    t0, pan0, tilt0 = history[0]
    t1, pan1, tilt1 = history[-1]

    lead_pan = lead_tilt = 0.0
    if t1 - t0 >= 0.1:
        lead_pan = (pan1 - pan0) / (t1 - t0) * LEAD_TIME
        lead_tilt = (tilt1 - tilt0) / (t1 - t0) * LEAD_TIME

    return (
        clamp(pan1 + clamp(lead_pan, -MAX_LEAD, MAX_LEAD), PAN_MIN, PAN_MAX),
        clamp(tilt1 + clamp(lead_tilt, -MAX_LEAD, MAX_LEAD), TILT_MIN, TILT_MAX),
    )


def step_toward(current, goal, max_step):
    return current + clamp(goal - current, -max_step, max_step)


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


class HailoFaceTracker:
    """카메라 → 포즈 검출 → 팬틸트 제어를 한 줄로 돌리는 얼굴 추적기.

    단독 실행(main)에서는 그대로 앞에서 돌리고, 메인 앱에서는 start()로
    백그라운드 스레드에서 돌린다. 감정 인식은 snapshot()으로 최신 화면과
    지금 따라가는 사람의 얼굴 위치를 받아 간다.
    """

    def __init__(
        self,
        stream_port=STREAM_PORT,
        save_image=True,
        verbose=True,
        auto_calibrate=None,
    ):
        self.stream_port = stream_port      # None이면 브라우저 화면 끔
        self.save_image = save_image        # latest.jpg 저장 여부
        self.verbose = verbose              # 프레임마다 상태 출력 여부
        self.auto_calibrate = auto_calibrate

        self.detector = None
        self.stream = None
        self.camera = None
        self.camera_started = False
        self.arduino = None

        self.pan = float(PAN_CENTER)
        self.tilt = float(TILT_CENTER)
        self.pan_dir, self.tilt_dir = PAN_DIRECTION, TILT_DIRECTION
        self.pan_deg_per_px, self.tilt_deg_per_px = DEG_PER_PX_X, DEG_PER_PX_Y

        self.smooth_x = None
        self.smooth_y = None

        # "scan": 좌우로 돌며 찾기, "track": 찾은 사람 추적
        self.mode = "scan"
        self.scan_direction = 1
        self.scan_started = time.monotonic()   # 이번 탐색을 시작한 때 (SCAN_GIVEUP 용)
        self.scan_resting = False
        # 최근 얼굴이 있던 서보 각도 기록 (시각, pan, tilt): 놓쳤을 때 갈 곳 예측용
        self.face_history = deque()
        self.lost_goal = None      # 놓친 뒤 돌아볼 (pan, tilt)
        self.lock = TargetLock()
        self.seen_frames = 0
        self.last_seen = 0.0

        self.last_image_save = 0.0
        self.last_loop = None

        # 다른 스레드(감정 인식)에 넘겨 줄 최신 화면과 대상 얼굴 위치
        self._snapshot_lock = threading.Lock()
        self._snapshot = (None, None)
        self._stop = threading.Event()
        self._thread = None
        # 꺼져 있으면(pause) 추적을 쉬고 정면에 서 있는다. 사람이 없을 때 쓴다.
        self._active = threading.Event()
        self._active.set()
        self._parked = False

    # ── 장치 열기 ─────────────────────────────────────────────
    def open(self):
        self.arduino = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=1,
            write_timeout=1,
        )

        # 아두이노 재시작 대기
        time.sleep(2)
        self.arduino.reset_input_buffer()

        # 시작 시 중앙 위치 명령
        command, reply, success = send_angles(
            self.arduino, self.pan, self.tilt
        )

        if not success:
            raise RuntimeError(
                f"중앙 명령 응답 이상: {reply!r}"
            )

        print(f"중앙 위치 설정: {command}", flush=True)
        time.sleep(1)

        self.detector = HailoPoseDetector(MODEL_PATH)
        if self.stream_port:
            self.stream = FrameStream(self.stream_port)

        self.camera = Picamera2()
        self.camera.configure(
            self.camera.create_video_configuration(
                main={
                    "format": "RGB888",
                    "size": CAPTURE_SIZE,
                },
                sensor={"output_size": SENSOR_SIZE},
                buffer_count=2,
            )
        )

        self.camera.start()
        self.camera_started = True
        time.sleep(1)

        auto = AUTO_CALIBRATE if self.auto_calibrate is None else self.auto_calibrate
        if auto:
            (
                self.pan_dir,
                self.tilt_dir,
                self.pan_deg_per_px,
                self.tilt_deg_per_px,
            ) = calibrate(lambda: self._grab()[1], self.arduino)

        print("사람 추적 시작 (Hailo HAT)")
        print("대상: 가장 크게 보이는 사람 한 명의 얼굴 (고르면 고정)")
        print(f"탐색 범위: PAN {SCAN_MIN}~{SCAN_MAX}")
        if self.save_image:
            print(f"결과 이미지: {RESULT_PATH}")
        if self.stream:
            print(f"카메라 화면: http://<라즈베리파이 IP>:{self.stream_port}")
        print("종료: Ctrl+C", flush=True)

        self.last_loop = time.monotonic()
        self._start_scan()

    def _grab(self):
        """(감정용 원본 화면, 추적용으로 줄인 화면)"""
        full = self.camera.capture_array()
        height, width = full.shape[:2]
        if (width, height) == FRAME_SIZE:
            return full, full
        small = cv2.resize(full, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        return full, small

    # ── 감정 인식 등 다른 스레드용 ─────────────────────────────
    def snapshot(self):
        """(최신 원본 화면, 따라가는 사람 얼굴의 원본 좌표 (x, y) 또는 None)"""
        with self._snapshot_lock:
            return self._snapshot

    # ── 한 프레임 처리 ────────────────────────────────────────
    def step(self):
        started = time.perf_counter()

        now = time.monotonic()
        dt = now - self.last_loop
        self.last_loop = now

        full, frame = self._grab()

        boxes, scores, keypoints = self.detector.detect(frame)

        height, width = frame.shape[:2]
        screen_x = width // 2
        screen_y = height // 2

        target_box = None
        target_point = None
        target_face = None
        face_full = None

        if len(boxes) > 0:
            self.seen_frames += 1
            self.last_seen = now
        else:
            self.seen_frames = 0

        if self.mode == "scan" and self.seen_frames >= CONFIRM_FRAMES:
            self.mode = "track"
            self.smooth_x = None
            self.smooth_y = None
            self.lock.reset()
            print("사람 발견: 추적 시작", flush=True)

        elif self.mode == "track" and now - self.last_seen > LOST_TIMEOUT:
            self.mode = "scan"
            # 사라진 쪽으로 계속 이어서 찾아봄
            if self.lost_goal is not None and self.face_history:
                moved = self.lost_goal[0] - self.face_history[0][1]
                if abs(moved) >= 1:
                    self.scan_direction = 1 if moved > 0 else -1
            self.face_history.clear()
            self.lost_goal = None
            self._start_scan()
            print("사람 놓침: 다시 탐색", flush=True)

        target_index = None
        if self.mode == "track":
            target_index = self.lock.update(frame, boxes, now)

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
                # 감정 인식에 넘길 얼굴 위치 (원본 화면 좌표)
                full_h, full_w = full.shape[:2]
                face_full = (
                    target_x * full_w / width,
                    target_y * full_h / height,
                )
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

            # 얼굴이 있는 곳을 서보 각도로 기록 (이 프레임을 찍을 때의 각도 기준)
            self.face_history.append((
                now,
                self.pan + self.pan_dir * (target_x - screen_x) * self.pan_deg_per_px,
                self.tilt + self.tilt_dir * (target_y - screen_y) * self.tilt_deg_per_px,
            ))
            while now - self.face_history[0][0] > HISTORY_TIME:
                self.face_history.popleft()
            self.lost_goal = None

            # 탐지 중심 좌표의 흔들림 완화
            if self.smooth_x is None:
                self.smooth_x = float(target_x)
                self.smooth_y = float(target_y)
            else:
                self.smooth_x += SMOOTHING * (
                    target_x - self.smooth_x
                )
                self.smooth_y += SMOOTHING * (
                    target_y - self.smooth_y
                )

            error_x = self.smooth_x - screen_x
            error_y = self.smooth_y - screen_y

            wanted_pan = self.pan + self.pan_dir * correction(
                error_x, self.pan_deg_per_px, dt
            )
            wanted_tilt = self.tilt + self.tilt_dir * correction(
                error_y, self.tilt_deg_per_px, dt
            )
            self.pan = clamp(wanted_pan, PAN_MIN, PAN_MAX)
            self.tilt = clamp(wanted_tilt, TILT_MIN, TILT_MAX)

            # 더 돌아야 하는데 안전 범위 끝에 막힌 경우 (사람이 범위 밖)
            limits = []
            if wanted_pan != self.pan:
                limits.append(f"PAN {round(self.pan)}")
            if wanted_tilt != self.tilt:
                limits.append(f"TILT {round(self.tilt)}")

            target_box = (x1, y1, x2, y2)
            target_point = (
                round(self.smooth_x),
                round(self.smooth_y),
            )

            target_status = (
                f"{'얼굴' if face_found else '얼굴 추정'} "
                f"오차 X={error_x:.0f}, "
                f"Y={error_y:.0f}"
            )
            if limits:
                target_status += (
                    f" | 각도 한계({', '.join(limits)}): "
                    "사람이 회전 범위 밖"
                )

        elif self.mode == "track":
            # 잠깐 안 보이면 마지막 각도 유지
            # (다른 사람만 보여도 LOCK_TIMEOUT 동안은 넘어가지 않음)
            self.smooth_x = None
            self.smooth_y = None

            if self.lost_goal is None and self.face_history:
                # 마지막 위치 + 움직이던 방향으로 지금쯤 있을 곳
                self.lost_goal = predict_lost_goal(self.face_history)
                print(
                    f"대상 놓침: 사라진 쪽 PAN {self.lost_goal[0]:.0f}, "
                    f"TILT {self.lost_goal[1]:.0f}으로 돌아봄",
                    flush=True,
                )

            if self.lost_goal is not None:
                self.pan = step_toward(self.pan, self.lost_goal[0], CHASE_SPEED * dt)
                self.tilt = step_toward(self.tilt, self.lost_goal[1], CHASE_SPEED * dt)
                target_status = (
                    f"대상 놓침: PAN {self.lost_goal[0]:.0f}, "
                    f"TILT {self.lost_goal[1]:.0f} 쪽을 보는 중"
                )
            else:
                target_status = "대상 안 보임: 각도 유지"

        elif now - self.scan_started > SCAN_GIVEUP:
            # 한참 찾아도 없으면 정면으로 돌아와 쉰다. 검출은 계속해서 보이면 바로 추적
            if not self.scan_resting:
                self.scan_resting = True
                print("아무도 안 보임: 정면에서 대기", flush=True)
            self.pan = step_toward(self.pan, float(PAN_CENTER), SCAN_SPEED * dt)
            self.tilt = float(TILT_CENTER)
            target_status = "탐색 쉼"

        else:
            # 좌우 끝까지 돌고 방향 전환, 틸트는 중앙
            self.pan += self.scan_direction * SCAN_SPEED * dt

            if self.pan >= SCAN_MAX:
                self.pan = SCAN_MAX
                self.scan_direction = -1
            elif self.pan <= SCAN_MIN:
                self.pan = SCAN_MIN
                self.scan_direction = 1

            self.tilt = float(TILT_CENTER)
            target_status = "탐색 중"

        with self._snapshot_lock:
            self._snapshot = (full, face_full)

        # 계산한 각도 전송
        command, reply, success = send_angles(
            self.arduino, self.pan, self.tilt
        )

        if not success:
            # 통신이 불확실한 상태에서는 추적 중단
            raise RuntimeError(
                f"아두이노 응답 이상: {reply!r}"
            )

        elapsed = time.perf_counter() - started

        if self.verbose:
            print(
                f"{target_status} | "
                f"전송={command} | 수신 성공 | "
                f"{elapsed * 1000:.0f}ms",
                flush=True,
            )

        # 보는 사람이 없으면 화면을 그리지 않음
        save_now = (
            self.save_image
            and now - self.last_image_save >= IMAGE_SAVE_INTERVAL
        )
        if self.stream is None and not save_now:
            return

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
            f"{self.mode.upper()}  PAN: {round(self.pan)}  TILT: {round(self.tilt)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if self.stream is not None:
            self.stream.update(image)

        # 결과 이미지 저장은 1초마다 수행
        if save_now:
            if not cv2.imwrite(str(TEMP_PATH), image):
                raise RuntimeError("결과 이미지 저장 실패")

            TEMP_PATH.replace(RESULT_PATH)
            self.last_image_save = now

    # ── 앞에서 계속 돌리기 (단독 실행) ─────────────────────────
    def run(self):
        while not self._stop.is_set():
            self.step()
            # capture_array()가 다음 프레임까지 기다리므로 따로 쉬지 않음

    # ── 백그라운드로 돌리기 (메인 앱) ──────────────────────────
    def start(self):
        """장치를 열고 추적 스레드를 시작한다. 장치를 못 열면 예외."""
        try:
            self.open()
        except Exception:
            self.close()
            raise
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self):
        """추적을 쉰다. 정면으로 돌아가 서 있고, 검출도 하지 않는다."""
        self._active.clear()

    def resume(self):
        """추적을 다시 한다. 처음부터 사람을 찾는다."""
        self._active.set()

    @property
    def paused(self):
        return not self._active.is_set()

    def has_target(self, within=3.0):
        """지금(또는 within 초 안에) 따라가는 사람이 있는지."""
        return (
            not self.paused
            and self.mode == "track"
            and time.monotonic() - self.last_seen <= within
        )

    def _park(self):
        """정면으로 돌려 놓고 탐색 상태를 비운다. 추적 스레드에서만 부른다."""
        self.pan, self.tilt = float(PAN_CENTER), float(TILT_CENTER)
        send_angles(self.arduino, self.pan, self.tilt)
        self.mode = "scan"
        self._start_scan()
        self.seen_frames = 0
        self.face_history.clear()
        self.lost_goal = None
        self.lock.reset()
        with self._snapshot_lock:
            self._snapshot = (None, None)
        self._parked = True
        print("추적 쉼: 정면에서 대기", flush=True)

    def _start_scan(self):
        self.scan_started = time.monotonic()
        self.scan_resting = False

    def _loop(self):
        errors = 0
        while not self._stop.is_set():
            try:
                if not self._active.is_set():
                    if not self._parked:
                        self._park()
                    self._active.wait(0.5)
                    continue
                if self._parked:
                    self._parked = False
                    self.last_loop = time.monotonic()   # 쉰 시간만큼 한 번에 돌지 않게
                    self._start_scan()
                    print("추적 다시 시작: 사람 찾는 중", flush=True)
                self.step()
                errors = 0
            except Exception as error:
                # USB 순간 끊김 등 — 추적만 잠깐 쉬고 대화는 계속
                errors += 1
                if errors in (1, 50):
                    print(f"⚠️  얼굴 추적 오류: {error}", flush=True)
                self._stop.wait(0.5)

    @property
    def shared_camera(self):
        """감정 인식이 쓰는 카메라 자리에 끼울 수 있는 객체."""
        return SharedCamera(self)

    # ── 정리 ────────────────────────────────────────────────
    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
            self._thread = None

        if self.camera is not None:
            if self.camera_started:
                self.camera.stop()
                self.camera_started = False
            self.camera.close()
            self.camera = None

        if self.stream is not None:
            self.stream.close()
            self.stream = None

        if self.detector is not None:
            self.detector.close()
            self.detector = None

        if self.arduino is not None:
            # 꺼질 때는 정면으로 돌려 놓음
            try:
                self.arduino.reset_input_buffer()
                command, reply, success = send_angles(
                    self.arduino, PAN_CENTER, TILT_CENTER
                )
                if success:
                    print(f"정면으로 복귀: {command}")
                else:
                    print(f"정면 복귀 응답 이상: {reply!r}")
            except Exception as error:
                print(f"정면 복귀 실패: {error}")

            self.arduino.close()
            self.arduino = None


class SharedCamera:
    """추적기가 찍은 화면을 감정 인식이 같이 쓰게 하는 카메라 흉내.

    카메라는 추적기가 열고 닫는다. 그래서 close()는 아무것도 안 한다.
    """

    def __init__(self, tracker, timeout=2.0):
        self._tracker = tracker
        self._timeout = timeout

    def snapshot(self):
        """(원본 화면, 따라가는 사람 얼굴 위치 또는 None). 첫 화면이 올 때까지 잠깐 기다림."""
        deadline = time.monotonic() + self._timeout
        while True:
            frame, face = self._tracker.snapshot()
            if frame is not None or time.monotonic() > deadline:
                return frame, face
            time.sleep(0.05)

    def capture(self):
        return self.snapshot()[0]

    def close(self):
        pass


def stop_on_signal(signum, frame):
    # kill·터미널 닫힘도 Ctrl+C처럼 처리해서 정리(정면 복귀)를 거치게 함
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGHUP, stop_on_signal)

    tracker = HailoFaceTracker()
    try:
        tracker.open()
        tracker.run()

    except KeyboardInterrupt:
        print("\n추적 종료")

    finally:
        tracker.close()


if __name__ == "__main__":
    main()