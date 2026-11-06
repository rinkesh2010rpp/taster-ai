"""The Claude Code hook: a tiny client that starts on every tool call.

Claude Code runs this command before each tool call and writes the call to
its stdin as JSON. It passes the call to the project's background service
(`serve.py`), starting the service if nothing is running, and prints the
service's answer: nothing to stay out of the way, or a deny.

It stays small on purpose: a few standard modules and nothing else from this
package (not even json, which costs more to import than everything else
here). If it can't get an answer it exits 1 with the reason on stderr,
which Claude Code treats as a hook error: the call goes ahead unless the hook
setting says `"onFailure": "block"`.
"""

from __future__ import annotations

import hashlib
import os
import socket
import sys
import time

CONFIG = os.path.join(".taster", "taster.yaml")
#: How long to wait for the service to answer. Below the hook's own timeout,
#: so a stuck service is reported here rather than as a cancelled hook.
REPLY_TIMEOUT = 50.0
START_TIMEOUT = 20.0
STALE_LOCK_S = 30.0


def state_folder() -> str:
    """Per-user folder for what the client and the service share."""
    return os.path.join(os.path.expanduser("~"), ".taster", "run")


def runtime_path(project: str) -> str:
    """The file naming one project's service: its port and token."""
    key = hashlib.sha256(os.path.normcase(os.path.abspath(project)).encode()).hexdigest()[:16]
    return os.path.join(state_folder(), f"{key}.json")


def read_runtime(project: str) -> dict | None:
    """The service's port and token, from a file holding "port token pid"."""
    try:
        with open(runtime_path(project), encoding="utf-8") as f:
            port, token, pid = f.read().split()
        return {"port": int(port), "token": token, "pid": int(pid)}
    except (OSError, ValueError):
        return None


class NotRunning(Exception):
    """Nothing answered at the service's address."""


def connect(port: int) -> socket.socket:
    conn = socket.socket()
    conn.settimeout(0.5)
    try:
        # Bytes, not str: a str host loads the idna codec, which costs more
        # than the whole connection.
        conn.connect((b"127.0.0.1", port))
    except OSError:
        conn.close()
        raise NotRunning from None
    return conn


def exchange(runtime: dict, event: bytes) -> str:
    """Send one call to the service; its answer, "" for no opinion. Raises
    NotRunning if nothing is there, OSError for any other failure."""
    with connect(runtime["port"]) as conn:
        conn.settimeout(REPLY_TIMEOUT)
        conn.sendall(runtime["token"].encode() + b"\n" + event.replace(b"\n", b" ") + b"\n")
        with conn.makefile("rb") as replies:
            line = replies.readline()
    if not line:
        raise OSError("the service closed the connection")
    status, _, body = line.decode().rstrip("\n").partition(" ")
    if status != "ok":
        raise OSError(body)
    return body


def start_service(project: str) -> None:
    """Start the service detached, so it outlives this process. A lock file
    stops two calls arriving together from starting two services."""
    import subprocess

    os.makedirs(state_folder(), exist_ok=True)
    lock = runtime_path(project) + ".lock"
    try:
        if time.time() - os.path.getmtime(lock) > STALE_LOCK_S:
            os.remove(lock)
    except OSError:
        pass
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return  # another call is starting it
    command = [sys.executable, "-m", "taster_ai.adapters.claude_code.serve", project]
    options: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "cwd": project,
        "close_fds": True,
    }
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        try:  # leave any job object Claude Code runs hooks in
            subprocess.Popen(command, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **options)
        except OSError:
            subprocess.Popen(command, creationflags=flags, **options)
    else:
        subprocess.Popen(command, start_new_session=True, **options)


def ask(project: str, event: bytes) -> str:
    """The service's answer, starting the service if it isn't running."""
    runtime = read_runtime(project)
    if runtime:
        try:
            return exchange(runtime, event)
        except NotRunning:
            pass
    start_service(project)
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        fresh = read_runtime(project)
        if fresh and fresh != runtime:
            try:
                return exchange(fresh, event)
            except NotRunning:
                pass
        time.sleep(0.05)
    raise OSError("the service did not start; see .taster/logs")


def main() -> int:
    raw = sys.stdin.buffer.read()
    project = os.environ.get("CLAUDE_PROJECT_DIR")
    if not project:
        import json  # only when Claude Code didn't say where the project is

        try:
            project = json.loads(raw).get("cwd")
        except (ValueError, AttributeError):
            project = None
    project = project or os.getcwd()
    if not os.path.exists(os.path.join(project, CONFIG)):
        return 0  # not set up here: stay out of the way
    try:
        output = ask(project, raw)
    except (OSError, ValueError) as e:
        print(f"taster: {e}", file=sys.stderr)
        return 1
    if output:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
