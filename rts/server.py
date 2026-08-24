"""Spectator server: runs matches on a background thread and streams them.

Stdlib only, so it runs anywhere Python does. The sim thread paces itself to
wall clock; each browser gets a Server-Sent Events feed of snapshots. If an
LLM commander is slow, the sim carries on without it -- pacing is never tied
to how fast an agent thinks.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config as cfg
from .commanders import make_commander
from .match import Match, record_path_for

STATIC = Path(__file__).parent / "static"
SNAPSHOT_HZ = 10


class MatchRunner:
    """Owns the current match and the thread that advances it."""

    def __init__(self, spec_a: str, spec_b: str, seed: int = 0, speed: float = 1.0,
                 think_interval: int = cfg.THINK_INTERVAL_TICKS,
                 model: str = "", personas: tuple[str, str] = ("", ""),
                 max_calls: int | None = None, autorestart: bool = True,
                 record_dir: str = ""):
        self.spec = (spec_a, spec_b)
        self.personas = personas
        self.model = model
        self.think_interval = think_interval
        self.max_calls = max_calls
        self.autorestart = autorestart
        self.record_dir = record_dir
        self.speed = speed
        self.paused = False
        self.seed = seed
        self.series: list[dict] = []          # results of finished matches
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.match = self._new_match()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _new_match(self) -> Match:
        kwargs = {"model": self.model} if self.model else {}
        commanders = [
            make_commander(self.spec[0], "RED", self.seed,
                           persona=self.personas[0] or None,
                           max_calls=self.max_calls, **kwargs),
            make_commander(self.spec[1], "BLUE", self.seed + 1,
                           persona=self.personas[1] or None,
                           max_calls=self.max_calls, **kwargs),
        ]
        return Match(commanders, seed=self.seed, think_interval=self.think_interval,
                     record_path=record_path_for(self.record_dir) if self.record_dir else None)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def restart(self, seed: int | None = None) -> None:
        with self._lock:
            self.seed = self.seed + 1 if seed is None else seed
            self.match.close()
            self.match = self._new_match()

    def snapshot(self) -> dict:
        with self._lock:
            snap = self.match.snapshot()
        snap["speed"] = self.speed
        snap["paused"] = self.paused
        snap["series"] = self.series[-12:]
        return snap

    def _loop(self) -> None:
        finished_at: float | None = None
        while not self._stop.is_set():
            started = time.time()
            with self._lock:
                match = self.match
                if not self.paused and not match.world.finished:
                    for _ in range(max(1, int(self.speed))):
                        match.step()
                        if match.world.finished:
                            break
                just_finished = match.world.finished

            if just_finished:
                if finished_at is None:
                    finished_at = time.time()
                    world = match.world
                    self.series.append({
                        "winner": None if world.winner is None else world.teams[world.winner].name,
                        "reason": world.finish_reason,
                        "clock": f"{world.tick // cfg.TICKS_PER_SECOND // 60}:"
                                 f"{world.tick // cfg.TICKS_PER_SECOND % 60:02d}",
                    })
                elif self.autorestart and time.time() - finished_at > 12:
                    finished_at = None
                    self.restart()
            else:
                finished_at = None

            # Fractional speeds slow the tick rate; integer speeds ran extra
            # ticks above, so the sleep stays one tick long either way.
            interval = 1.0 / (cfg.TICKS_PER_SECOND * min(self.speed, 1.0))
            time.sleep(max(0.0, interval - (time.time() - started)))


RUNNER: MatchRunner | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter console; the game is the output
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        assert RUNNER is not None
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/state":
            self._send(200, json.dumps(RUNNER.snapshot()).encode(), "application/json")
        elif path == "/stream":
            self._stream()
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        assert RUNNER is not None
        if self.path.split("?")[0] != "/control":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"bad json"}', "application/json")
            return
        action = body.get("action")
        if action == "pause":
            RUNNER.paused = True
        elif action == "resume":
            RUNNER.paused = False
        elif action == "restart":
            RUNNER.restart()
        elif action == "speed":
            RUNNER.speed = max(0.25, min(float(body.get("value", 1)), 20))
        else:
            self._send(400, b'{"error":"unknown action"}', "application/json")
            return
        self._send(200, json.dumps({"ok": True, "speed": RUNNER.speed,
                                    "paused": RUNNER.paused}).encode(),
                   "application/json")

    def _stream(self) -> None:
        assert RUNNER is not None
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                payload = json.dumps(RUNNER.snapshot())
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
                time.sleep(1.0 / SNAPSHOT_HZ)
        except (BrokenPipeError, ConnectionResetError):
            pass  # spectator closed the tab


def serve(runner: MatchRunner, host: str = "127.0.0.1", port: int = 8765) -> None:
    global RUNNER
    RUNNER = runner
    runner.start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"  watch it at  http://{host}:{port}")
    print(f"  {runner.match.commanders[0].name} ({runner.spec[0]})  vs  "
          f"{runner.match.commanders[1].name} ({runner.spec[1]})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        runner.stop()
        httpd.server_close()
