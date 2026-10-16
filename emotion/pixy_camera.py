"""
Pixy2 카메라 영상 — Pi 카메라 대신 Pixy2 센서 프레임을 받아 쓴다.

  Pixy2 센서 ──USB──▶ Bayer 원본 316x208 ──디모자이크──▶ BGR ──▶ YuNet / 감정 분류

Pixy2 는 색 블록만 보내주는 칩이지만, libpixyusb2 의 get_raw_frame 처럼 Pixy 프로그램을
멈추면(video_stop) 센서 원본 프레임을 통째로 받을 수 있다. 바인딩에 이 함수가 있어야 하므로
README 의 'Pixy2 파이썬 모듈 빌드' 대로 패치한 pixy.py/_pixy*.so 를 써야 한다.

Pixy 프로그램이 멈춰 있는 동안 색 블록(TRACK_SOURCE=pixy)은 안 나온다. 서보는 그대로 움직인다.
vision.Camera 와 같은 capture()/close() 를 가져서 EmotionService·FaceTracker 에 그대로 끼운다.
"""
import cv2
import numpy as np

import config
from pixy_servo import LOCK, open_pixy

FRAME_TYPE = 0x21                 # (해상도 2 << 4) | 모드 1 → 316x208 전체 화면
WIDTH, HEIGHT = 316, 208


class PixyCamera:
    """Pixy2 센서 프레임을 BGR ndarray 로. 스레드 여러 개가 같이 불러도 된다."""

    def __init__(self):
        self._pixy = open_pixy()
        if not hasattr(self._pixy, "video_get_frame"):
            raise RuntimeError("pixy 바인딩에 video_get_frame 이 없음 — 패치한 모듈로 다시 빌드")
        with LOCK:
            res = self._pixy.video_stop()
        if res < 0:
            raise RuntimeError(f"Pixy2 프로그램 정지 실패 ({res})")
        self._buf = bytearray(WIDTH * HEIGHT)

    def capture(self):
        """현재 프레임을 BGR ndarray 로 반환. (316x208)"""
        with LOCK:   # 버퍼를 같이 쓰니 변환까지 락 안에서
            n = self._pixy.video_get_frame(FRAME_TYPE, 0, 0, WIDTH, HEIGHT, self._buf)
            if n < 0:
                raise RuntimeError(f"Pixy2 프레임 받기 실패 ({n})")
            bayer = np.frombuffer(self._buf, np.uint8).reshape(HEIGHT, WIDTH)
            # Pixy2 는 BGGR. OpenCV 이름은 한 칸 밀려 있어서 BGGR → BayerRG 다.
            frame = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2BGR)
        # 뒤집기는 디모자이크 뒤에 (Bayer 를 뒤집으면 색 배열이 바뀐다)
        if config.CAM_HFLIP and config.CAM_VFLIP:
            frame = cv2.flip(frame, -1)
        elif config.CAM_HFLIP:
            frame = cv2.flip(frame, 1)
        elif config.CAM_VFLIP:
            frame = cv2.flip(frame, 0)
        return frame

    def close(self):
        try:
            with LOCK:
                self._pixy.video_resume()   # 안 하면 Pixy2 가 멈춘 채로 남는다
        except Exception:
            pass


# ── 단독 실행: 한 장 받아서 파일로 확인 ─────────────────────
if __name__ == "__main__":
    import sys
    import time
    cam = PixyCamera()
    try:
        cam.capture()                     # 첫 장은 버림 (노출 안정화)
        t0 = time.monotonic()
        n = 10
        for _ in range(n):
            frame = cam.capture()
        print(f"{n / (time.monotonic() - t0):.1f} fps, {frame.shape}")
        out = sys.argv[1] if len(sys.argv) > 1 else "pixy_frame.png"
        cv2.imwrite(out, frame)
        print(f"저장: {out}")
    finally:
        cam.close()
