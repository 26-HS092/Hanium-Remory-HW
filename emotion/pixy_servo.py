"""
Pixy2 팬틸트 래퍼 — 모터(서보) 제어 + (선택) Pixy2 색 블록 읽기.

Pixy2 는 USB 로 파이에 꽂고, 공식 libpixyusb2 의 파이썬(SWIG) 바인딩 `pixy` 를 쓴다.
  - 서보: pixy.set_servos(pan, tilt), 위치 범위 0 ~ 1000 (500 = 가운데)
  - 블록: pixy.ccc_get_blocks(n, BlockArray) — 색 시그니처로 학습한 블롭 목록

`pixy` 모듈은 pip 패키지가 아니라 직접 빌드해야 한다. (README '얼굴 트래킹' 참고)
모듈이 없거나 Pixy2 가 안 꽂혀 있으면 생성자에서 예외 → 호출 쪽이 트래킹 없이 계속 간다.
"""
import threading

RCS_MIN = 0
RCS_MAX = 1000
RCS_CENTER = 500


class PixyServo:
    """Pixy2 팬틸트. 서보 0 = pan(좌우), 서보 1 = tilt(상하)."""

    def __init__(self):
        import pixy  # libpixyusb2 SWIG 바인딩. 파이에서 빌드한 pixy.py + _pixy*.so
        self._pixy = pixy
        if pixy.init() < 0:
            raise RuntimeError("Pixy2 연결 실패 (USB 연결 / udev 권한 확인)")
        # 블록 모드로 둔다. 서보 제어는 어떤 프로그램에서든 되고,
        # TRACK_SOURCE=pixy 일 때는 이 모드의 블록을 그대로 쓴다.
        pixy.change_prog("color_connected_components")
        self._blocks = pixy.BlockArray(20)
        self._lock = threading.Lock()   # USB 링크는 한 번에 한 명령만
        self.frame_size = (pixy.get_frame_width(), pixy.get_frame_height())

    def set(self, pan, tilt):
        pan = int(min(RCS_MAX, max(RCS_MIN, pan)))
        tilt = int(min(RCS_MAX, max(RCS_MIN, tilt)))
        with self._lock:
            self._pixy.set_servos(pan, tilt)

    def largest_block(self, signatures=None):
        """가장 큰 색 블록의 (cx, cy, w, h). signatures 로 거르고, 없으면 None."""
        with self._lock:
            count = self._pixy.ccc_get_blocks(20, self._blocks)
        best = None
        for i in range(max(0, count)):
            b = self._blocks[i]
            if signatures and b.m_signature not in signatures:
                continue
            if best is None or b.m_width * b.m_height > best[2] * best[3]:
                best = (b.m_x, b.m_y, b.m_width, b.m_height)
        return best

    def close(self):
        try:
            self.set(RCS_CENTER, RCS_CENTER)
        except Exception:
            pass
