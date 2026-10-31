"""Scope: when a check runs, on which tools, and how strictly.

    Scope(
        tool_call=[Target("execute", mode="enforce"), Target("*", mode="shadow")],
        tool_result=[Target("search_web", mode="enforce")],
    )

A scope holds one ordered list of targets per stage: `tool_call` (before the
tool runs) and `tool_result` (after it runs, before the model sees the
result). Inside a stage the first matching target wins; a stage left out is
not checked, and a tool no target matches is not checked there.

A scope is plain configuration: no judges, no keys. That makes it safe to
keep in git and to load from YAML or JSON. Keep it somewhere the agent
cannot write (your code, a config file outside the agent's workspace, or an
environment variable). An agent that can edit its own scope can be talked
into switching its checks off.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

Mode = Literal["pass", "shadow", "enforce"]
MODES = ("pass", "shadow", "enforce")
ON_ERROR = ("label", "block")
UNCLEAR = ("wrap", "withhold", "pass")
STAGES = ("tool_call", "tool_result")


@dataclass
class Target:
    """How strictly a check treats the tools whose name matches `tool`.

    tool       fnmatch pattern on the tool name ("search_web", "mcp_*", "*")
    mode       "pass"    — not checked
               "shadow"  — checked and logged, nothing changed
               "enforce" — checked, logged, and acted on
    threshold  minimum confidence for a "breaks" verdict to block the call
               or withhold the result; below it a result is wrapped as
               untrusted and a call runs
    on_error   when no verdict can be had: "label" lets it through (a
               result marked unscreened), "block" blocks or withholds it. A
               call too large to check is blocked in enforce mode either way.
    unclear    what an "unclear" verdict does to a result: "wrap",
               "withhold" or "pass"; for a call, "withhold" blocks it and
               the others let it run
    when_args  {arg_name: regex}; the target matches only if every named
               argument matches (re.search). A call that doesn't match falls
               through to later targets.
    detector   name of a judge given to the check (`detectors={"name":
               ...}`); None uses the check's judge
    """

    tool: str
    mode: Mode = "shadow"
    threshold: float = 0.85
    on_error: Literal["label", "block"] = "label"
    unclear: Literal["wrap", "withhold", "pass"] = "wrap"
    when_args: Mapping[str, str] = field(default_factory=dict)
    detector: str | None = None

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"target {self.tool!r}: mode must be one of {MODES}, got {self.mode!r}")
        if self.on_error not in ON_ERROR:
            raise ValueError(f"target {self.tool!r}: on_error must be one of {ON_ERROR}")
        if self.unclear not in UNCLEAR:
            raise ValueError(f"target {self.tool!r}: unclear must be one of {UNCLEAR}")
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError(f"target {self.tool!r}: threshold must be between 0 and 1")
        if self.detector is not None and not isinstance(self.detector, str):
            raise ValueError(f"target {self.tool!r}: detector must be a name; give the object to the check")
        self._arg_patterns = {arg: re.compile(pattern) for arg, pattern in (self.when_args or {}).items()}

    def matches(self, tool: str, args: Mapping[str, Any]) -> bool:
        if not fnmatch.fnmatchcase(tool, self.tool):
            return False
        return all(p.search(str((args or {}).get(arg, ""))) for arg, p in self._arg_patterns.items())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Target:
        """Unknown keys raise, so a typo can't silently switch a check off."""
        bad = set(raw) - _TARGET_KEYS
        if bad:
            raise ValueError(f"target {raw.get('tool')!r}: unknown keys {sorted(bad)}")
        return cls(**raw)


PASS = Target("*", mode="pass")
_TARGET_KEYS = {"tool", "mode", "threshold", "on_error", "unclear", "when_args", "detector"}


class Targets:
    """One stage's ordered targets; the first match wins, and a tool no
    target matches is not checked."""

    def __init__(self, targets: Iterable[Target | Mapping[str, Any]] = ()):
        self.targets = [t if isinstance(t, Target) else Target.from_dict(t) for t in targets]

    def target_for(self, tool: str, args: Mapping[str, Any] | None = None) -> Target:
        for target in self.targets:
            if target.matches(tool, args or {}):
                return target
        return PASS

    def __iter__(self):
        return iter(self.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> Target:
        return self.targets[i]

    def __repr__(self) -> str:
        return f"Targets({self.targets!r})"


@dataclass
class Scope:
    """When a check runs, on which tools, and how strictly: an ordered list
    of targets per stage. At least one stage is needed."""

    tool_call: Targets | list | None = None
    tool_result: Targets | list | None = None

    def __post_init__(self):
        for stage in STAGES:
            value = getattr(self, stage)
            if value is not None and not isinstance(value, Targets):
                setattr(self, stage, Targets(value))
        if not self.stages:
            raise ValueError(f"a scope needs at least one stage: {STAGES}")

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(s for s in STAGES if getattr(self, s) is not None)

    def for_stage(self, stage: str) -> Targets | None:
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        return getattr(self, stage)

    @property
    def targets(self) -> list[Target]:
        return [t for s in self.stages for t in self.for_stage(s)]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Scope:
        """{"tool_call": [{"tool": "*", "mode": "shadow"}], "tool_result": [...]}"""
        if not isinstance(data, Mapping):
            raise ValueError(f"a scope maps stages {STAGES} to lists of targets, got {type(data).__name__}")
        unknown = set(data) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown scope stages: {sorted(unknown)}; stages are {STAGES}")
        return cls(**{s: [Target.from_dict(t) for t in data[s] or []] for s in data})


def load_yaml(source: str) -> tuple[Any, str | None]:
    """Parse YAML from a path or from text; also returns the file's folder
    (None for text), for paths inside it."""
    import yaml  # optional dependency: pip install taster-ai[yaml]

    folder = None
    if "\n" not in source and os.path.exists(source):
        folder = os.path.dirname(os.path.abspath(source))
        with open(source, encoding="utf-8") as f:
            source = f.read()
    return yaml.safe_load(source), folder
