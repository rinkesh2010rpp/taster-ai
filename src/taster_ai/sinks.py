"""Where screening records go. A sink is any callable taking one dict."""

from __future__ import annotations

import json
import logging
import os
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
    """One summary line per screened call on the `taster_ai` logger."""
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
