"""Judges: does this text follow the standard? Add your own by subclassing
`Detector` and implementing `detect(text, standard)`."""

from __future__ import annotations

from typing import Any

from .base import Detector, FallbackDetector
from .heuristic import HeuristicDetector
from .jev import JevDetector

__all__ = [
    "Detector",
    "FallbackDetector",
    "HeuristicDetector",
    "JevDetector",
    "LLMDetector",
    "as_detector",
]


def __getattr__(name: str) -> Any:
    # LLMDetector needs langchain-core; import it only when asked for.
    if name == "LLMDetector":
        from .llm import LLMDetector

        return LLMDetector
    raise AttributeError(name)


def as_detector(obj: Any) -> Detector:
    """A Detector as is, or a LangChain chat model wrapped in LLMDetector."""
    if isinstance(obj, Detector):
        return obj
    try:
        from langchain_core.language_models import BaseChatModel
    except ImportError:
        BaseChatModel = None  # type: ignore[assignment,misc]
    if BaseChatModel is not None and isinstance(obj, BaseChatModel):
        from .llm import LLMDetector

        return LLMDetector(obj)
    raise TypeError(
        f"expected a Detector or a LangChain chat model, got {type(obj).__name__}; "
        "for your own judge, subclass Detector and implement detect(text, standard)"
    )
