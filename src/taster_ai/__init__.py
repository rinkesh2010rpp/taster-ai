"""Taster: checks on what your AI agent does and what it takes in.

Every tool call is checked before it runs, and every tool result before your
agent swallows it, by the checks you choose (Injection, Constitution, your
own Check). The core has no dependencies; framework adapters live in
`taster_ai.adapters`.

Names load on first use, so a program that needs one small part of the
package (the Claude Code hook starts on every tool call) doesn't pay for the
rest.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.3.0.dev0"

_NAMES = {
    ".actions": ("Messages",),
    ".checks": (
        "Check",
        "Constitution",
        "Injection",
        "checks_from_dict",
        "checks_from_env",
        "checks_from_json",
        "checks_from_yaml",
    ),
    ".detectors": (
        "ClaudeCliDetector",
        "Detector",
        "FallbackDetector",
        "InjectionPatterns",
        "JevDetector",
        "LLMDetector",
    ),
    ".scope": ("Scope", "Target"),
    ".screener": ("Decision", "Outcome", "Screener"),
    ".sinks": ("BackgroundSink", "JsonlSink", "log_sink", "print_sink"),
    ".standards": ("INJECTION_STANDARD",),
    ".verdict": ("TOO_LARGE", "TOO_LONG", "Verdict"),
}
_HOME = {name: module for module, names in _NAMES.items() for name in names}

__all__ = sorted(_HOME)


def __getattr__(name: str):
    if name in _HOME:
        value = getattr(import_module(_HOME[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
