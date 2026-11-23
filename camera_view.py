"""
Pi 카메라 화면을 브라우저로 실시간 보기 (MJPEG).

    python camera_view.py              # http://<파이 IP>:8080
    python camera_view.py --faces      # YuNet 얼굴 박스도 그려서
    python camera_view.py --port 9000 --size 640x480

※ 카메라는 한 프로그램만 열 수 있다. mori.service 가 카메라를 쓰고 있으면
   먼저 `sudo systemctl stop mori` 하고 실행.
   (본 앱 안에서 트래킹 화면을 보려면 .env 에 TRACK_STREAM_PORT=8080 을 넣고 재시작)
"""
import argparse
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "emotion"))

PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mori camera</title></head>
<body style="margin:0;background:#000;display:flex;justify-content:center">
<img src="/stream" style="max-width:100%;max-height:100vh">
</body></html>"""


class Latest:
    """가장 최근 JPEG 한 장. 접속한 브라우저들이 이걸 받아간다."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None

    def put(self, jpeg):
        with self.cond:
            self.jpeg = jpeg
            self.cond.notify_all()

    def wait(self):
        with self.cond:
            self.cond.wait(timeout=1.0)
            return self.jpeg


def make_server(port, latest):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    jpeg = latest.wait()
                    if jpeg is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        + jpeg + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    return server


def lan_ips():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        return [s.getsockname()[0]]
    except OSError:
        return ["<파이 IP>"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--size", default="1280x720", help="캡처 해상도 WxH")
    ap.add_argument("--faces", action="store_true", help="YuNet 얼굴 박스 그리기")
    ap.add_argument("--quality", type=int, default=75, help="JPEG 품질")
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))

    import config
    from picamera2 import Picamera2
    from libcamera import Transform

    if not Picamera2.global_camera_info():
        sys.exit("❌ 카메라가 안 잡힘 (rpicam-hello --list-cameras 로 확인: 케이블/재부팅)")

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": size, "format": "RGB888"},
        transform=Transform(hflip=config.CAM_HFLIP, vflip=config.CAM_VFLIP),
    ))
    cam.start()

    detector = None
    if args.faces:
        import face_crop
        detector = face_crop.create_detector(config.YUNET_PATH, config.FACE_SCORE_THRESHOLD)

    latest = Latest()
    server = make_server(args.port, latest)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for ip in lan_ips():
        print(f"📷 카메라 화면: http://{ip}:{args.port}  (Ctrl+C 로 종료)")

    frames, t0, fps = 0, time.time(), 0.0
    try:
        while True:
            frame = cam.capture_array()   # "RGB888" = BGR 순서 → OpenCV 그대로
            if detector is not None:
                for x, y, w, h, score in face_crop.detect_faces(detector, frame):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{score:.2f}", (x, max(0, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            frames += 1
            if time.time() - t0 >= 1.0:
                fps, frames, t0 = frames / (time.time() - t0), 0, time.time()
            cv2.putText(frame, f"{fps:.1f} fps", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if ok:
                latest.put(enc.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
