"""
얼굴 트래킹 — Pixy2 팬틸트 모터로 어르신 얼굴을 화면 가운데에 둔다.

  Pi 카메라 ──▶ YuNet 얼굴 검출 ──▶ 얼굴 중심 오차 ──▶ PD 제어 ──▶ Pixy2 set_servos
     ▲                                                              │
     └──────────── 같은 틸트 플레이트 위라 모터와 함께 돈다 ◀──────────┘

Pixy2 카메라는 색 덩어리만 알아서 '사람'을 직접 못 잡는다. 그래서 기본값(face)은
사람을 알아보는 일은 Pi 카메라 + YuNet 이 하고, Pixy2 는 모터를 돌린다.
TRACK_SOURCE=pixy 로 두면 Pixy2 가 학습한 색(피부색·색 마커)을 따라간다.

얼굴이 가운데 있으면 감정 샘플(emotion_service.py)도 얼굴이 잘린 채 찍히지 않는다.
카메라는 EmotionService 가 연 것을 같이 쓴다(디바이스 중복 오픈 금지).
프레임은 검출에만 쓰고 바로 버린다 — 저장하지 않는다.
"""
import threading
import time

import cv2

import config
import face_crop
from pixy_servo import RCS_MAX, RCS_MIN


class AxisPD:
    """서보 한 축. 서보는 '위치' 장치라 PD 출력을 위치에 누적한다(Pixy2 데모와 같은 방식)."""

    def __init__(self, kp, kd, direction, home,
                 deadband=config.TRACK_DEADBAND, max_step=config.TRACK_MAX_STEP):
        self.kp, self.kd = kp, kd
        self.direction = 1 if direction >= 0 else -1
        self.deadband = deadband
        self.max_step = max_step
        self.home = home
        self.pos = float(home)
        self._prev = None

    def update(self, err):
        """err: -1 ~ 1 (화면 반폭 기준). 반환: 새 서보 위치."""
        e = 0.0 if abs(err) < self.deadband else err
        d = 0.0 if self._prev is None else e - self._prev
        self._prev = e
        step = self.kp * e + self.kd * d
        step = max(-self.max_step, min(self.max_step, step))
        self.pos = min(RCS_MAX, max(RCS_MIN, self.pos + self.direction * step))
        return self.pos

    def drift_home(self, rate):
        """대상을 놓쳤을 때 홈 자세로 천천히 복귀."""
        self._prev = None
        diff = self.home - self.pos
        self.pos += max(-rate, min(rate, diff))
        return self.pos


class FaceTracker:
    """백그라운드 스레드에서 얼굴을 찾아 팬틸트를 돌린다."""

    def __init__(self, servo, camera=None, source=config.TRACK_SOURCE):
        if source == "face" and camera is None:
            raise ValueError("TRACK_SOURCE=face 는 카메라가 필요합니다")
        self._servo = servo
        self._cam = camera
        self._source = source
        self._detector = None
        if source == "face":
            # 감정 분석기와 같은 YuNet 이지만 인스턴스는 따로 (스레드 간 공유 금지).
            self._detector = face_crop.create_detector(
                config.YUNET_PATH, config.FACE_SCORE_THRESHOLD, config.FACE_NMS_THRESHOLD,
            )
        home_pan, home_tilt = config.TRACK_HOME
        self.pan = AxisPD(config.PAN_KP, config.PAN_KD, config.PAN_DIR, home_pan)
        self.tilt = AxisPD(config.TILT_KP, config.TILT_KD, config.TILT_DIR, home_tilt)
        self._period = 1.0 / config.TRACK_FPS
        self._stop = threading.Event()
        self._thread = None
        self.locked = False            # 지금 얼굴을 잡고 있는지 (표시·디버그용)
        self._last_seen = 0.0

    # ── 대상 위치: 화면 가운데 기준 정규화 오차 ─────────────
    def _locate(self):
        """(pan_err, tilt_err) 또는 None. Pixy2 데모와 같은 부호:
        pan_err  = (가운데 - x) / 반폭   → 대상이 왼쪽이면 +
        tilt_err = (y - 가운데) / 반높이 → 대상이 아래면 +
        """
        if self._source == "pixy":
            blk = self._servo.largest_block(config.PIXY_SIGNATURES)
            if blk is None:
                return None
            cx, cy = blk[0], blk[1]
            W, H = self._servo.frame_size
        else:
            frame = self._cam.capture()
            H0, W0 = frame.shape[:2]
            scale = config.TRACK_DETECT_WIDTH / float(W0)
            small = cv2.resize(frame, (config.TRACK_DETECT_WIDTH, int(H0 * scale)))
            det = face_crop.detect_largest_face(self._detector, small)
            del frame
            if det is None:
                return None
            x, y, w, h = det[:4]
            cx, cy = x + w / 2.0, y + h / 2.0
            H, W = small.shape[:2]
        return (W / 2.0 - cx) / (W / 2.0), (cy - H / 2.0) / (H / 2.0)

    # ── 제어 한 번 ──────────────────────────────────────────
    def step(self, now=None):
        now = time.monotonic() if now is None else now
        loc = self._locate()
        if loc is not None:
            if not self.locked:
                print("🎯 얼굴 트래킹: 잡음")
            self.locked = True
            self._last_seen = now
            pan_pos = self.pan.update(loc[0])
            tilt_pos = self.tilt.update(loc[1])
        else:
            if self.locked and now - self._last_seen > config.TRACK_LOST_HOLD_SEC:
                print("🎯 얼굴 트래킹: 놓침 → 정면으로 복귀")
                self.locked = False
            if self.locked:
                return loc       # 잠깐 가려진 것일 수 있으니 그 자리에서 기다림
            pan_pos = self.pan.drift_home(config.TRACK_HOME_STEP)
            tilt_pos = self.tilt.drift_home(config.TRACK_HOME_STEP)
        self._servo.set(pan_pos, tilt_pos)
        return loc

    def _loop(self):
        self._servo.set(self.pan.pos, self.tilt.pos)
        errors = 0
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.step(t0)
                errors = 0
            except Exception as e:      # USB 순간 끊김 등 — 트래킹만 쉬고 대화는 계속
                errors += 1
                if errors in (1, 50):
                    print(f"⚠️  얼굴 트래킹 오류: {e}")
                self._stop.wait(0.5)
                continue
            self._stop.wait(max(0.0, self._period - (time.monotonic() - t0)))

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._servo.close()


def create_tracker(camera=None):
    """설정대로 트래커를 만든다. 꺼져 있거나 Pixy2 가 없으면 None (대화는 계속)."""
    if not config.TRACK_ENABLED:
        return None
    try:
        from pixy_servo import PixyServo
        tracker = FaceTracker(PixyServo(), camera=camera, source=config.TRACK_SOURCE)
        tracker.start()
        print(f"🎯 얼굴 트래킹 활성화 (source={config.TRACK_SOURCE})")
        return tracker
    except Exception as e:
        print(f"⚠️  얼굴 트래킹 비활성화: {e}")
        return None


# ── 단독 실행: 모터 방향·게인 맞추기용 ─────────────────────
if __name__ == "__main__":
    from vision import Camera
    cam = Camera() if config.TRACK_SOURCE == "face" else None
    from pixy_servo import PixyServo
    trk = FaceTracker(PixyServo(), camera=cam)
    trk.start()
    print("Ctrl+C 로 종료. 얼굴이 화면 밖으로 도망가면 PAN_DIR / TILT_DIR 을 -1 로.")
    try:
        while True:
            time.sleep(0.5)
            print(f"locked={trk.locked} pan={trk.pan.pos:.0f} tilt={trk.tilt.pos:.0f}")
    except KeyboardInterrupt:
        pass
    finally:
        trk.close()
        if cam:
            cam.close()
