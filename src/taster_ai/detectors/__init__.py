"""Detectors judge whether text is a prompt injection. Add your own by
subclassing `Detector` and implementing `detect`."""

from __future__ import annotations

from typing import Any, Mapping

from .base import Detector, FallbackDetector
from .heuristic import HeuristicDetector
from .jev import JevDetector

__all__ = ["Detector", "FallbackDetector", "HeuristicDetector", "JevDetector", "build_detector"]

BUILTIN: dict[str, type[Detector]] = {"jev": JevDetector, "heuristic": HeuristicDetector}


def build_detector(spec: Any, named: Mapping[str, Detector] | None = None) -> Detector:
    """Build a detector from config.

    `spec` is a Detector, a name ("jev", "heuristic", or a key of `named`),
    or a dict: {"type": "jev", "timeout": 3} or
    {"type": "fallback", "primary": ..., "fallback": ...}.
    """
    if isinstance(spec, Detector):
        return spec
    if isinstance(spec, str):
        if named and spec in named:
            return named[spec]
        if spec in BUILTIN:
            return BUILTIN[spec]()
        raise ValueError(f"unknown detector {spec!r}")
    if isinstance(spec, Mapping):
        options = dict(spec)
        kind = options.pop("type", None)
        if kind == "fallback":
            return FallbackDetector(
                build_detector(options.pop("primary"), named), build_detector(options.pop("fallback"), named)
            )
        if kind in BUILTIN:
            return BUILTIN[kind](**options)
        raise ValueError(f"unknown detector type {kind!r}")
    raise ValueError(f"can't build a detector from {spec!r}")
