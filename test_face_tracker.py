"""
얼굴 트래킹 제어 루프 시뮬레이션 테스트 — 파이·Pixy2·카메라 없이 돌아간다.

가짜 팬틸트(서보 속도·화각 반영)와 가짜 얼굴 위치로 FaceTracker.step() 을 돌려서
  - 화면 가장자리 얼굴이 가운데로 수렴하는지
  - 움직이는 얼굴을 놓치지 않고 따라가는지
  - 방향 설정이 틀리면 실제로 도망가는지(부호 검증)
  - 놓쳤을 때 잠깐 기다렸다가 정면으로 돌아오는지
를 본다.

    python test_face_tracker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "emotion"))

import config                    # noqa: E402
from face_tracker import FaceTracker  # noqa: E402

DEG_PER_UNIT = 180.0 / 1000      # 서보 0~1000 ≈ 180°
HFOV, VFOV = 66.0, 41.0          # Pi Camera Module 3 (16:9) 대략값
SERVO_DEG_PER_SEC = 400.0        # SG90 급 서보 속도 (0.15s/60°)
DT = 1.0 / config.TRACK_FPS


class FakeRig:
    """서보 명령 → 실제 각도(속도 제한) → 카메라 화면 속 얼굴 위치."""

    def __init__(self):
        self.cmd = [500.0, 500.0]
        self.angle = [0.0, 0.0]          # 실제 카메라 방향(°), 0 = 정면
        self.frame_size = (316, 208)

    # PixyServo 인터페이스
    def set(self, pan, tilt):
        self.cmd = [pan, tilt]

    def close(self):
        pass

    def advance(self, dt):
        for i in range(2):
            target = (self.cmd[i] - 500) * DEG_PER_UNIT
            diff = target - self.angle[i]
            lim = SERVO_DEG_PER_SEC * dt
            self.angle[i] += max(-lim, min(lim, diff))

    def observe(self, face_az, face_el):
        """얼굴 방위(°)를 화면 오차로. Pixy2 데모 부호:
        pan_err 는 얼굴이 왼쪽이면 +, tilt_err 는 얼굴이 아래면 +.
        서보 pan 위치가 커지면 왼쪽(+az), tilt 위치가 커지면 아래(-el)를 본다고 둔다."""
        rel_az = face_az - self.angle[0]
        rel_el = face_el + self.angle[1]
        if abs(rel_az) > HFOV / 2 or abs(rel_el) > VFOV / 2:
            return None
        return rel_az / (HFOV / 2), -rel_el / (VFOV / 2)


class SimTracker(FaceTracker):
    def __init__(self, rig, pan_dir=1, tilt_dir=1):
        super().__init__(rig, camera=None, source="pixy")   # pixy 경로: 검출기 안 만듦
        self.pan.direction = pan_dir
        self.tilt.direction = tilt_dir
        self.rig = rig
        self.face = (20.0, 10.0)     # (방위°, 고도°)
        self.visible = True
        self._pending = None         # 카메라 한 프레임 지연

    def _locate(self):
        loc, self._pending = self._pending, (
            self.rig.observe(*self.face) if self.visible else None)
        return loc


def run(trk, seconds, t0=0.0, face_fn=None):
    t = t0
    errs = []
    for _ in range(int(seconds / DT)):
        if face_fn:
            trk.face = face_fn(t)
        loc = trk.step(now=t)
        trk.rig.advance(DT)
        t += DT
        errs.append(loc)
    return t, errs


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return ok


def main():
    results = []

    # 1) 화면 가장자리 정지 얼굴 → 2초 안에 데드밴드 안으로
    rig = FakeRig()
    trk = SimTracker(rig)
    trk.face = (28.0, 15.0)
    _, errs = run(trk, 2.0)
    last = errs[-1]
    ok = last is not None and abs(last[0]) < config.TRACK_DEADBAND * 1.5 \
        and abs(last[1]) < config.TRACK_DEADBAND * 1.5
    results.append(check("정지 얼굴 수렴", ok, f"최종 오차={last}"))
    first = next(i for i, e in enumerate(errs)
                 if e and abs(e[0]) < 0.15 and abs(e[1]) < 0.15)
    results.append(check("수렴 속도 < 1초", first * DT < 1.0, f"{first * DT:.2f}s"))
    # 오버슈트: 부호가 크게 뒤집히면 떨림
    overs = max((-e[0] for e in errs if e), default=0)
    results.append(check("오버슈트 작음", overs < 0.2, f"반대쪽 최대 {overs:.2f}"))

    # 2) 좌우로 천천히 걷는 얼굴(±30°, 8초 주기) → 화면 밖으로 안 나감
    import math
    rig = FakeRig()
    trk = SimTracker(rig)
    _, errs = run(trk, 10.0, face_fn=lambda t: (30 * math.sin(2 * math.pi * t / 8), 5.0))
    lost = sum(e is None for e in errs[3:])
    worst = max(abs(e[0]) for e in errs[3:] if e)
    results.append(check("움직이는 얼굴 추종", lost == 0 and worst < 0.5,
                         f"놓친 프레임={lost}, 최대 오차={worst:.2f}"))

    # 3) 방향이 틀리면 얼굴이 화면 밖으로 도망가야 한다 (부호 검증)
    rig = FakeRig()
    trk = SimTracker(rig, pan_dir=-1)
    trk.face = (15.0, 0.0)
    _, errs = run(trk, 2.0)
    results.append(check("PAN_DIR 반대면 놓침", errs[-1] is None))

    # 4) 놓침 → TRACK_LOST_HOLD_SEC 동안 유지 → 이후 홈 복귀
    rig = FakeRig()
    trk = SimTracker(rig)
    trk.face = (25.0, 0.0)
    t, _ = run(trk, 2.0)
    held = trk.pan.pos
    trk.visible = False
    t, _ = run(trk, config.TRACK_LOST_HOLD_SEC * 0.8, t0=t)
    results.append(check("가려져도 잠깐 유지", abs(trk.pan.pos - held) < 1,
                         f"{held:.0f} → {trk.pan.pos:.0f}"))
    run(trk, config.TRACK_LOST_HOLD_SEC + 12.0, t0=t)
    results.append(check("놓치면 정면 복귀", abs(trk.pan.pos - config.TRACK_HOME[0]) < 1,
                         f"pan={trk.pan.pos:.0f}"))

    print(f"\n{sum(results)}/{len(results)} 통과")
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
