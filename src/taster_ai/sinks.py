"""Where screening records go. A sink is any callable taking one dict.

Sinks run in line, before the model sees the result. Fast ones (the built-in
JSONL, log and print sinks) cost microseconds; wrap a slow one (a network
call to Slack, Datadog, a database) in BackgroundSink so the agent never
waits on it.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

Sink = Callable[[dict[str, Any]], None]
logger = logging.getLogger("taster_ai")


class JsonlSink:
    """Appends one JSON line per screened call to <directory>/<date>-<id>.jsonl.

    Each process writes its own file, so several workers sharing a directory
    (or a network volume) never write to the same file.
    """

    def __init__(self, directory: str):
        self.directory = directory
        self.process_id = os.urandom(4).hex()

    def __call__(self, record: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"{now:%Y-%m-%d}-{self.process_id}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now.isoformat(timespec="seconds"), **record}, default=str) + "\n")


def log_sink(record: dict[str, Any]) -> None:
    """One summary line per screened call on the `taster_ai` logger (INFO)."""
    logger.info(
        "%s mode=%s label=%s conf=%s action=%s would=%s err=%s hash=%s",
        record["tool"], record["mode"], record["label"], record["confidence"],
        record["action"], record["would_action"], record["error"], record["input_hash"],
    )


def print_sink(record: dict[str, Any]) -> None:
    """Like log_sink, but printed, for hosts that capture stdout (e.g. Modal)."""
    print(
        f"[taster] {record['tool']} mode={record['mode']} label={record['label']} "
        f"conf={record['confidence']} action={record['action']} would={record['would_action']} "
        f"err={record['error']} hash={record['input_hash']}"
    )


class BackgroundSink:
    """Runs a slow sink on a background thread so the agent never waits on it.

        sinks=[print_sink, BackgroundSink(send_to_slack)]

    Records queue up to `max_queue`; beyond that new ones are dropped (and
    counted in `dropped`) rather than slowing the agent down. Queued records
    are flushed for up to `flush_on_exit` seconds when the process exits;
    call `flush()` yourself before a short-lived process ends.
    """

    def __init__(self, sink: Sink, *, max_queue: int = 1000, flush_on_exit: float = 2.0):
        self.sink = sink
        self.dropped = 0
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(max_queue)
        self._thread = threading.Thread(target=self._run, name="taster-sink", daemon=True)
        self._thread.start()
        if flush_on_exit:
            atexit.register(self.flush, flush_on_exit)

    def __call__(self, record: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 100 == 0:
                logger.warning("taster: background sink %r is behind; %d records dropped", self.sink, self.dropped)

    def _run(self) -> None:
        while True:
            record = self._queue.get()
            try:
                self.sink(record)
            except Exception:  # a failing sink must not kill the thread
                logger.exception("taster: background sink %r failed", self.sink)
            finally:
                self._queue.task_done()

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until queued records are handled; True if the queue emptied in time."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True
