"""The background service behind the Claude Code hook, one per project.

The hook starts it the first time it's needed and it exits after sitting idle,
so nobody manages it. It keeps what can't survive between hook calls: the
checks, the verdict cache and the judge's waiting processes.

It listens on 127.0.0.1 only, on a free port, and answers only callers that
send the token from a file in the user's own folder.

Tool calls in enforce mode are answered once judged. Everything else is
answered at once; shadow checks then run in the background, so the agent
never waits for them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import socketserver
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ...checks import checks_from_yaml
from ...screener import Screener
from ...sinks import JsonlSink
from .hook import CONFIG, read_runtime, runtime_path

logger = logging.getLogger("taster_ai")

IDLE_EXIT_S = 1800.0
MAX_EVENT_BYTES = 8_000_000
#: Shadow checks waiting beyond this are dropped and logged, not queued
#: forever: a burst must not use up the user's plan.
MAX_PENDING = 64


def deny(reason: str) -> str:
    """Claude Code's JSON for blocking a tool call. Taster never says allow:
    that would skip the user's own permission prompts."""
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    )


class Service:
    """Decides what to do with one tool call."""

    def __init__(self, project: str, *, background_workers: int = 4):
        self.project = project
        self.config = os.path.join(project, CONFIG)
        self.logs = JsonlSink(os.path.join(project, ".taster", "logs"))
        self.background = ThreadPoolExecutor(max_workers=background_workers)
        self.screener: Screener | None = None
        self.rules = ""
        self.pending = 0
        self.last_used = time.monotonic()
        self._lock = threading.Lock()
        self._loaded: tuple = ()
        self.load()
        self.log({"event": "service-start", "pid": os.getpid()})

    def _files(self) -> list[str]:
        folder = os.path.dirname(self.config)
        return [os.path.join(folder, n) for n in sorted(os.listdir(folder)) if n.endswith((".yaml", ".md"))]

    def _stamp(self) -> tuple:
        return tuple((f, os.path.getmtime(f)) for f in self._files())

    def load(self) -> None:
        """Read the config, again whenever a file in it changes. A broken
        edit keeps the last good checks and is logged."""
        stamp = self._stamp()
        if stamp == self._loaded:
            return
        digest = hashlib.sha256()
        for path in self._files():
            with open(path, "rb") as f:
                digest.update(f.read())
        try:
            checks = checks_from_yaml(self.config)
            screener = Screener(checks, sinks=[self.log])
        except Exception as e:
            self._loaded = stamp
            self.log({"event": "config-error", "error": f"{type(e).__name__}: {e}"})
            if self.screener is None:
                raise
            return
        self.close_judges()
        self.screener, self._loaded, self.rules = screener, stamp, digest.hexdigest()[:12]
        self.warm()

    def judges(self) -> list[Any]:
        if self.screener is None:
            return []
        found = {id(j): j for c in self.screener.checks for j in (c.judge, *c.detectors.values())}
        return list(found.values())

    def warm(self) -> None:
        for judge in self.judges():
            if hasattr(judge, "warm"):
                judge.warm()

    def close_judges(self) -> None:
        for judge in self.judges():
            if hasattr(judge, "close"):
                judge.close()

    def log(self, record: dict[str, Any]) -> None:
        self.logs({**record, "rules": self.rules})

    def targets(self, tool: str, args: dict[str, Any]) -> list[Any]:
        """The target each check would use for this call, leaving out "pass"."""
        found = []
        for check in self.screener.checks if self.screener else []:
            targets = check.scope.for_stage("tool_call")
            target = targets.target_for(tool, args) if targets is not None else None
            if target is not None and target.mode != "pass":
                found.append(target)
        return found

    def handle(self, event: dict[str, Any]) -> str:
        """What the hook prints: "" for no opinion, or a deny."""
        self.last_used = time.monotonic()
        self.load()
        tool = str(event.get("tool_name", ""))
        args = event.get("tool_input")
        args = args if isinstance(args, dict) else {}
        call_id = event.get("tool_use_id")
        targets = self.targets(tool, args)
        if not targets:
            return ""
        if any(t.mode == "enforce" for t in targets):
            outcome = self.screener.screen_call(tool, args, call_id)
            return deny(outcome.message()) if outcome is not None and outcome.blocked else ""
        with self._lock:
            if self.pending >= MAX_PENDING:
                self.log({"event": "dropped", "tool": tool})
                return ""
            self.pending += 1
        self.background.submit(self._shadow, self.screener, tool, args, call_id)
        return ""

    def _shadow(self, screener: Screener, tool: str, args: dict[str, Any], call_id: Any) -> None:
        try:
            screener.screen_call(tool, args, call_id)
        finally:
            with self._lock:
                self.pending -= 1
                self.last_used = time.monotonic()

    def idle_for(self) -> float:
        with self._lock:
            return 0.0 if self.pending else time.monotonic() - self.last_used

    def close(self) -> None:
        self.log({"event": "service-stop"})
        self.close_judges()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        token = self.rfile.readline(200).strip().decode(errors="replace")
        if not secrets.compare_digest(token, self.server.token):
            return
        line = self.rfile.readline(MAX_EVENT_BYTES)
        try:
            reply = f"ok {self.server.service.handle(json.loads(line))}"
        except Exception as e:  # the hook reports it; onFailure decides
            logger.exception("taster: could not handle a call")
            reply = f"error {type(e).__name__}: {e}".replace("\n", " ")
        self.wfile.write(reply.encode() + b"\n")


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False


def write_runtime(project: str, port: int, token: str) -> None:
    """Publish where to find the service: atomically, readable only by the
    user. The lock from the hook goes once the file is in place."""
    path = runtime_path(project)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}"
    descriptor = os.open(temporary, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as f:
        f.write(f"{port} {token} {os.getpid()}")
    os.replace(temporary, path)
    try:
        os.remove(path + ".lock")
    except OSError:
        pass


def remove_runtime(project: str) -> None:
    """Remove the file, if it's still ours."""
    runtime = read_runtime(project)
    if runtime and runtime["pid"] == os.getpid():
        try:
            os.remove(runtime_path(project))
        except OSError:
            pass


def run(project: str, idle_exit_s: float = IDLE_EXIT_S) -> None:
    service = Service(project)
    server = _Server(("127.0.0.1", 0), _Handler)
    server.service = service  # type: ignore[attr-defined]
    server.token = secrets.token_hex(16)  # type: ignore[attr-defined]
    write_runtime(project, server.server_address[1], server.token)  # type: ignore[attr-defined]

    def watch() -> None:
        while service.idle_for() < idle_exit_s:
            time.sleep(min(5.0, idle_exit_s))
        server.shutdown()

    threading.Thread(target=watch, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        remove_runtime(project)
        service.close()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m taster_ai.adapters.claude_code.serve <project>", file=sys.stderr)
        return 2
    run(sys.argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
