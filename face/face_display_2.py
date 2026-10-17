"""
face_display_2.py  (브라우저 얼굴 버전 — 기존 pygame FaceDisplay 대체)
---------------------------------------------------------------
브라우저(mori_face.html)에 WebSocket으로 표정/말하기/사진 명령을 보낸다.
main.py 는 set_expression / listen_start / listen_stop / think_start / think_stop /
listen_level / nod / start_speaking / stop_speaking /
show_notice / hide_notice / show_photo / hide_photo / has_screen / ask_show / ask_wait /
close 만 쓴다. 화면에서 누른 답(예/아니오)과 얼굴 터치(on_answer / on_touch)는
같은 WebSocket 으로 거꾸로 받는다.

필요: pip install websockets
브라우저는 chromium --kiosk 로 mori_face.html 을 띄워두면 자동 연결된다.
"""
from __future__ import annotations

import asyncio, itertools, json, queue, threading, time
import websockets


class FaceDisplay2:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self._host, self._port = host, port
        self._clients = set()
        self._cur_expr = "평온"
        self._ready_app = False      # 앱 준비가 끝났는지. 끝나기 전엔 화면이 로딩을 띄운다.
        self._listening = False
        self._thinking = False
        self._notice = None          # (제목, 부제) — 안내가 떠 있는 동안만
        # 화면에서 묻는 중인 질문 (id, 제목, 부제) 과 눌린 답. 답은 브라우저가
        # 보내오고 ask_wait 가 꺼내 간다.
        self._ask = None
        self._ask_ids = itertools.count(1)
        self._answers: "queue.Queue[tuple[int, str]]" = queue.Queue()
        # 화면이 눌린 그 순간 부를 함수(답을 받는다). 누른 소리를 바로 내는 데 쓴다.
        # WebSocket 스레드에서 불리니 오래 걸리는 일은 따로 돌려야 한다.
        self.on_answer = None
        # 얼굴을 건드리셨을 때 부를 함수(kind: "poke" 톡 건드림 | "pet" 쓰다듬기).
        self.on_touch = None
        self._last_level = 0.0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._boot_error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)   # 서버가 listen 시작할 때까지 대기
        if self._boot_error:          # 포트가 이미 쓰이는 중 등 — 부른 쪽이 알게 한다
            raise self._boot_error

    def _run(self):
        asyncio.set_event_loop(self._loop)

        async def handler(ws, *_):   # websockets v10/v11 시그니처 모두 호환
            self._clients.add(ws)
            try:
                # 새로 연결되면 현재 표정을 즉시 동기화
                await ws.send(json.dumps({"cmd": "emotion", "name": self._cur_expr},
                                         ensure_ascii=False))
                if self._ready_app:
                    await ws.send(json.dumps({"cmd": "ready"}))
                if self._listening:
                    await ws.send(json.dumps({"cmd": "listen_start"}))
                if self._thinking:
                    await ws.send(json.dumps({"cmd": "think_start"}))
                if self._notice:
                    title, sub, icon = self._notice
                    await ws.send(json.dumps(
                        {"cmd": "notice", "title": title, "sub": sub, "icon": icon},
                        ensure_ascii=False))
                if self._ask:
                    await ws.send(json.dumps(self._ask_msg(), ensure_ascii=False))
                async for raw in ws:
                    self._on_message(raw)
            finally:
                self._clients.discard(ws)

        async def boot():
            await websockets.serve(handler, self._host, self._port)
            self._ready.set()

        try:
            self._loop.run_until_complete(boot())
        except OSError as e:
            self._boot_error = e
            self._ready.set()
            return
        self._loop.run_forever()

    def _on_message(self, raw):
        """화면에서 온 소식. 지금은 질문에 누른 답 하나뿐이다."""
        try:
            m = json.loads(raw)
        except (TypeError, ValueError):
            return
        if m.get("evt") == "ask_answer" and m.get("answer") in ("yes", "no"):
            self._answers.put((m.get("id"), m["answer"]))
            if self.on_answer:
                try:
                    self.on_answer(m["answer"])
                except Exception:
                    pass
        elif m.get("evt") == "touch" and m.get("kind") in ("poke", "pet"):
            if self.on_touch:
                try:
                    self.on_touch(m["kind"])
                except Exception:
                    pass

    def _ask_msg(self) -> dict:
        ask_id, title, sub = self._ask
        return {"cmd": "ask", "id": ask_id, "title": title, "sub": sub}

    def _send(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False)

        async def _do():
            if self._clients:
                await asyncio.gather(*[c.send(data) for c in self._clients],
                                     return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_do(), self._loop)
        except Exception:
            pass

    # ── main.py 가 쓰는 인터페이스 ──
    def set_expression(self, label: str):
        """label: 기쁨 | 슬픔 | 위로 | 경청 | 놀람 | 평온  (+수면)"""
        self._cur_expr = label
        self._send({"cmd": "emotion", "name": label})

    def set_ready(self):
        """앱 준비가 끝났다고 알린다. 화면이 로딩을 걷고 얼굴을 보인다."""
        self._ready_app = True
        self._send({"cmd": "ready"})

    def listen_start(self):
        """어르신 말을 듣기 시작했다 — 화면에 불을 켠다.

        꺼져 있다가 켜지는 순간 한 번 확 번쩍이고, 듣는 동안은 은은하게
        숨쉰다. 웨이크워드를 알아들었는지 멀리서도 바로 보이게 하려는 것이다.
        """
        self._listening = True
        self._send({"cmd": "listen_start"})

    def listen_stop(self):
        """다 들었다 — 불을 끈다(이제 생각하거나 말할 차례다)."""
        self._listening = False
        self._send({"cmd": "listen_stop"})

    def listen_level(self, v: float):
        """듣는 중 목소리 크기(0~1). 전구가 목소리에 맞춰 밝아진다.

        녹음 루프에서 0.1초마다 불린다. 거의 안 변했으면 보내지 않는다.
        """
        if abs(v - self._last_level) < 0.04 and not (v == 0 and self._last_level):
            return
        self._last_level = v
        self._send({"cmd": "level", "v": round(v, 2)})

    def nod(self):
        """말씀 중간에 쉬실 때 한 번 끄덕인다. 너무 잦지 않게 화면이 걸러 준다."""
        self._send({"cmd": "nod"})

    def think_start(self):
        """다 듣고 답을 만드는 중이다 — 생각하는 얼굴을 짓는다.

        답이 나오기까지 몇 초 걸리는데 얼굴이 가만히 있으면 못 알아들은 줄
        아신다. 눈을 위로 굴리고 고개를 살짝 갸웃하며 점 세 개를 띄운다.
        말을 시작하면(speak_start) 화면이 알아서 그친다.
        """
        self._thinking = True
        self._send({"cmd": "think_start"})

    def think_stop(self):
        self._thinking = False
        self._send({"cmd": "think_stop"})

    def start_speaking(self):
        self._thinking = False
        self._send({"cmd": "speak_start"})

    def stop_speaking(self):
        self._send({"cmd": "speak_stop"})

    def show_notice(self, title: str, sub: str = "", icon: str = "pill"):
        """약 알림·가족 메시지 같은 안내를 화면에 띄운다.

        말하는 동안만 얼굴을 덮는다. 소리를 놓쳤거나 잘 안 들리는 어르신도
        무슨 일인지 읽을 수 있게 하려는 것이다.
        icon: pill(알약) | message(말풍선)
        """
        self._notice = (title, sub, icon)
        self._send({"cmd": "notice", "title": title, "sub": sub, "icon": icon})

    def hide_notice(self):
        """안내를 내리고 모리 얼굴로 돌아온다."""
        self._notice = None
        self._send({"cmd": "notice_hide"})

    def has_screen(self) -> bool:
        """화면이 붙어 있는지. 안 붙어 있으면 터치로 물을 수 없다."""
        return bool(self._clients)

    def ask_show(self, title: str, sub: str = "") -> int:
        """화면에 예/아니오를 띄운다. ask_wait 에 넘길 번호를 돌려준다.

        사진을 보고 계시던 중이어도 사진이 비쳐 보이게 반투명으로 덮는다.
        """
        while True:                      # 지난 질문에 늦게 눌린 답은 버린다
            try:
                self._answers.get_nowait()
            except queue.Empty:
                break
        ask_id = next(self._ask_ids)
        self._ask = (ask_id, title, sub)
        self._send(self._ask_msg())
        return ask_id

    def ask_wait(self, ask_id: int, timeout: float) -> str | None:
        """눌린 답("yes"/"no")을 기다린다. 아무것도 안 누르시면 None.

        끝나면 질문을 내린다.
        """
        deadline = time.monotonic() + timeout
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                try:
                    got_id, answer = self._answers.get(timeout=left)
                except queue.Empty:
                    return None
                if got_id == ask_id:
                    return answer
        finally:
            self._ask = None
            self._send({"cmd": "ask_hide"})

    def show_photo(self, url: str):
        """가족이 보낸 사진을 화면 전체에 띄운다."""
        self._send({"cmd": "photo", "url": url})

    def hide_photo(self):
        """사진을 내리고 얼굴로 복귀한다."""
        self._send({"cmd": "photo_hide"})

    def close(self):
        self._listening = False
        self._thinking = False
        self._notice = None
        self._ask = None
        self._send({"cmd": "ask_hide"})
        self._send({"cmd": "listen_stop"})
        self._send({"cmd": "think_stop"})
        self._send({"cmd": "notice_hide"})
        self._send({"cmd": "emotion", "name": "평온"})
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
