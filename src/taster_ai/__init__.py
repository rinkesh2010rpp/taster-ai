"""Taster: antivirus for your AI agent.

Every tool result is tasted before your agent swallows it. The core has no
dependencies; framework adapters live in `taster_ai.adapters`.
"""

from .actions import Messages
from .detectors import Detector, FallbackDetector, HeuristicDetector, JevDetector
from .policy import Policy, Rule
from .screener import Decision, Screener
from .sinks import JsonlSink, log_sink, print_sink
from .verdict import ToolContext, Verdict

__version__ = "0.1.0.dev0"

__all__ = [
    "Decision",
    "Detector",
    "FallbackDetector",
    "HeuristicDetector",
    "JevDetector",
    "JsonlSink",
    "Messages",
    "Policy",
    "Rule",
    "Screener",
    "ToolContext",
    "Verdict",
    "log_sink",
    "print_sink",
]
