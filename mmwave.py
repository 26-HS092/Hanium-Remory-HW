"""
mmwave.py
---------
mmWave 레이더(Seeed MR24HPC1 계열, UART 115200)로 사람이 있는지 본다.

라즈베리파이 GPIO14(TX)·GPIO15(RX)에 붙어 있다. Pi 5 에서는 /dev/serial0 이
디버그 커넥터(ttyAMA10)라서 GPIO UART 인 /dev/ttyAMA0 을 직접 연다.
/boot/firmware/config.txt 에 dtparam=uart0=on 이 있어야 이 장치가 생긴다.

프레임: 53 59 | 제어 | 명령 | 길이(2) | 데이터 | 체크섬 | 54 43
  사람 있음/없음: 제어 0x80, 명령 0x01(센서가 알려 줌) 또는 0x81(물어본 답), 데이터 1=있음

센서가 없거나 답이 끊기면 '있음'으로 본다. 레이더 하나 때문에 인형이
아예 말을 안 듣게 되면 안 된다.
"""
from __future__ import annotations

import os
import threading
import time

import serial

MMWAVE_PORT = os.getenv("MMWAVE_PORT", "/dev/ttyAMA0")
MMWAVE_BAUD = 115200
MMWAVE_ENABLED = os.getenv("MMWAVE_ENABLED", "true").lower() == "true"

QUERY_SEC = 2.0        # 이만큼마다 사람 있는지 물어본다
SILENT_SEC = 15.0      # 이만큼 답이 없으면 센서가 끊긴 걸로 보고 '있음'으로 둔다

_QUERY = bytes([0x53, 0x59, 0x80, 0x81, 0x00, 0x01, 0x0F, 0xBD, 0x54, 0x43])


class MmWavePresence:
    """레이더를 백그라운드로 읽는다. present 로 지금 상태를, on_change 로 바뀔 때를 받는다."""

    def __init__(self, port: str = MMWAVE_PORT, baud: int = MMWAVE_BAUD):
        self._ser = serial.Serial(port, baud, timeout=0.1)
        self._present = True
        self._known = threading.Event()        # 첫 답을 받았는지
        self._here = threading.Event()         # 사람 있음
        self._away = threading.Event()         # 사람 없음 (웨이크워드 대기를 끊는 데 쓴다)
        self._here.set()
        self._callbacks = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ── 밖에서 쓰는 것 ──
    @property
    def present(self) -> bool:
        return self._present

    @property
    def away_event(self) -> threading.Event:
        return self._away

    def on_change(self, fn) -> None:
        """fn(present: bool). 레이더 스레드에서 불리니 오래 걸리는 일은 따로 돌린다."""
        self._callbacks.append(fn)

    def wait_known(self, timeout: float) -> bool:
        """첫 답이 올 때까지 기다린다."""
        return self._known.wait(timeout)

    def wait_present(self, stop_event=None) -> bool:
        """사람이 올 때까지 기다린다. stop_event 가 켜지면 False."""
        while not self._here.wait(0.5):
            if stop_event is not None and stop_event.is_set():
                return False
        return True

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._ser.close()

    # ── 안쪽 ──
    def _set(self, present: bool) -> None:
        self._known.set()
        if present == self._present:
            return
        self._present = present
        if present:
            self._away.clear()
            self._here.set()
        else:
            self._here.clear()
            self._away.set()
        print(f"📡 mmWave: {'사람 있음' if present else '사람 없음'}", flush=True)
        for fn in self._callbacks:
            try:
                fn(present)
            except Exception as e:
                print(f"⚠️  mmWave 콜백 오류: {e}")

    def _run(self) -> None:
        buf = b""
        last_query = 0.0
        last_rx = time.monotonic()
        warned = False
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                if now - last_query > QUERY_SEC:
                    self._ser.write(_QUERY)
                    last_query = now
                buf += self._ser.read(64)
            except serial.SerialException as e:
                if not warned:
                    print(f"⚠️  mmWave 읽기 실패: {e}")
                    warned = True
                self._stop.wait(1.0)
                continue

            while True:
                i = buf.find(b"\x53\x59")
                if i == -1:
                    buf = buf[-1:]          # 머리 첫 바이트만 걸쳐 있을 수 있다
                    break
                if len(buf) < i + 7:
                    buf = buf[i:]
                    break
                n = (buf[i + 4] << 8) | buf[i + 5]
                end = i + 9 + n
                if len(buf) < end:
                    buf = buf[i:]
                    break
                frame = buf[i:end]
                buf = buf[end:]
                if frame[-2:] != b"\x54\x43" or sum(frame[:-3]) & 0xFF != frame[-3]:
                    continue                # 깨진 프레임
                last_rx = time.monotonic()
                warned = False
                if frame[2] == 0x80 and frame[3] in (0x01, 0x81) and n >= 1:
                    self._set(frame[6] == 1)

            if time.monotonic() - last_rx > SILENT_SEC:
                if not warned:
                    print("⚠️  mmWave 응답 없음 → 사람 있음으로 둔다")
                    warned = True
                self._set(True)
                last_rx = time.monotonic()


def create_presence() -> MmWavePresence | None:
    """레이더를 연다. 꺼져 있거나 못 열면 None (그러면 늘 사람이 있는 것처럼 동작)."""
    if not MMWAVE_ENABLED:
        return None
    if not os.path.exists(MMWAVE_PORT):
        print(f"⚠️  mmWave 비활성화: {MMWAVE_PORT} 없음 (config.txt 에 dtparam=uart0=on 필요)")
        return None
    try:
        presence = MmWavePresence()
    except Exception as e:
        print(f"⚠️  mmWave 비활성화: {e}")
        return None
    if presence.wait_known(timeout=5.0):
        print(f"📡 mmWave 활성화: 지금 {'사람 있음' if presence.present else '사람 없음'}")
    else:
        print("📡 mmWave 활성화 (아직 답 없음 → 사람 있음으로 시작)")
    return presence


if __name__ == "__main__":
    p = create_presence()
    if p is None:
        raise SystemExit(1)
    p.on_change(lambda here: None)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        p.close()
