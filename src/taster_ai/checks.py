"""Checks: what a tool call or result is held to, and who judges it.

Every check has the same parts:

    name      shown in logs and to the agent
    standard  what it is held to, in plain words
    scope     when it runs (tool_call, tool_result), on which tools, how strictly
    judge     who decides: Jev by default, or a Detector, or a chat model

    Check(
        name="secrets_in_urls",
        standard="Never put a secret or token in a URL.",
        scope=Scope(tool_call=[Target("execute", mode="enforce")]),
    )

Two checks come ready made, set up for what they are for:

    Injection()                     tool results, shadow, every tool
    Constitution(standard=text)     tool calls, shadow, every tool

Whenever a check's scope matches, its judge gets two things: the text (the
tool call, or what the tool returned) and the check's standard, and answers
follows, breaks or unclear. A judge keeps no standard of its own, so one
judge can serve every check.

Telling the agent the standard is up to whoever builds the agent (put it in
the system prompt). Taster only checks. Keep standards where the agent
cannot edit them.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from .detectors import Detector, JevDetector, as_detector
from .policy import STAGES, Scope, Target, load_yaml
from .standards import INJECTION_STANDARD


class Check:
    """A standard, the scope it applies in, and the judge that applies it.

    name         shown in logs and in what the agent is told
    standard     plain words: what calls or results must follow
    scope        a Scope, or its dict form
    judge        a Detector, a LangChain chat model (wrapped in
                 LLMDetector), or None for Jev with `jev_options`
                 (timeout, title, referer, ...)
    detectors    extra judges by name, for targets that say `detector: <name>`
    """

    #: The reason given to the agent when the judge gave none.
    default_reason = "breaks the standard"

    def __init__(
        self,
        name: str,
        standard: str,
        scope: Scope | Mapping[str, Any],
        judge: Any = None,
        *,
        detectors: Mapping[str, Any] | None = None,
        **jev_options: Any,
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("a check needs a name")
        if not isinstance(standard, str) or not standard.strip():
            raise ValueError(f"check {name!r}: the standard must be non-empty text")
        self.name = name.strip()
        self.standard = standard.strip()
        self.scope = scope if isinstance(scope, Scope) else Scope.from_dict(scope)
        if judge is not None and jev_options:
            raise ValueError(f"check {self.name!r}: Jev options {sorted(jev_options)} apply only to the default judge")
        self.judge: Detector = JevDetector(**jev_options) if judge is None else as_detector(judge)
        self.detectors = {n: as_detector(d) for n, d in (detectors or {}).items()}
        missing = sorted(
            {t.detector for t in self.scope.targets if t.mode != "pass" and t.detector} - set(self.detectors)
        )
        if missing:
            raise ValueError(
                f"check {self.name!r}: targets name detectors that weren't given: {missing}; "
                "pass detectors={name: ...}"
            )

    def judge_for(self, target: Target) -> Detector:
        return self.detectors[target.detector] if target.detector else self.judge

    def reason(self, verdict: Any) -> str:
        """Why it failed, in the words the agent is told: the judge's reason
        if it gave one."""
        return verdict.evidence[0] if verdict.evidence else self.default_reason

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, stages={self.scope.stages})"


class Injection(Check):
    """Prompt injection in what tools return.

    Defaults: every tool's result, in shadow mode, against INJECTION_STANDARD,
    judged by Jev. Give a `standard` to hold results to your own words.
    """

    default_reason = "flagged as a possible prompt injection"

    def __init__(
        self,
        scope: Scope | Mapping[str, Any] | None = None,
        judge: Any = None,
        *,
        standard: str | None = None,
        detectors: Mapping[str, Any] | None = None,
        **jev_options: Any,
    ):
        super().__init__(
            "injection",
            standard or INJECTION_STANDARD,
            scope if scope is not None else Scope(tool_result=[Target("*", mode="shadow")]),
            judge,
            detectors=detectors,
            **jev_options,
        )

    def reason(self, verdict: Any) -> str:
        # A judge's reason can quote the attack; the agent is not shown it.
        return self.default_reason


class Constitution(Check):
    """The owner's rules, checked against tool calls before they run.

    Defaults: every tool call, in shadow mode, judged by Jev. The standard
    has no default: it is your constitution.
    """

    default_reason = "breaks the constitution"

    def __init__(
        self,
        standard: str,
        scope: Scope | Mapping[str, Any] | None = None,
        judge: Any = None,
        *,
        detectors: Mapping[str, Any] | None = None,
        **jev_options: Any,
    ):
        super().__init__(
            "constitution",
            standard,
            scope if scope is not None else Scope(tool_call=[Target("*", mode="shadow")]),
            judge,
            detectors=detectors,
            **jev_options,
        )

    @classmethod
    def from_file(cls, path: str, **options: Any) -> Constitution:
        with open(path, encoding="utf-8") as f:
            return cls(f.read(), **options)


def call_pieces(tool: str, args: Mapping[str, Any], size: int, overlap: int = 200) -> list[str]:
    """The text a judge reads for one call, in pieces of at most `size`
    chars. A call that fits is one piece. Otherwise the longest arguments
    are moved out of the header until it takes at most half a piece, and each
    piece is that header (the tool and every short argument) plus one
    overlapping window of one long argument. Nothing is cut out: every
    character of every argument is in some piece, next to the tool's name."""
    whole = f"ACTION: {tool}\nARGS: {_dumps(dict(args))}"
    if len(whole) <= size:
        return [whole]
    values = {k: v if isinstance(v, str) else _dumps(v) for k, v in args.items()}
    long: list[str] = []
    for key in sorted(values, key=lambda k: len(values[k]), reverse=True):
        if len(_header(tool, args, long)) <= size // 2:
            break
        long.append(key)
    header = _header(tool, args, long)
    pieces = []
    for key in long:
        value = values[key]
        label = f"\n--- {key}, chars {len(value)}-{len(value)} of {len(value)} ---\n"
        window = max(100, size - len(header) - len(label))
        step = max(1, window - min(overlap, window // 2))
        for start in range(0, len(value), step):
            end = min(len(value), start + window)
            pieces.append(f"{header}\n--- {key}, chars {start}-{end} of {len(value)} ---\n{value[start:end]}")
            if end >= len(value):
                break
    return pieces


def _header(tool: str, args: Mapping[str, Any], long: list[str]) -> str:
    text = f"ACTION: {tool}\nARGS: {_dumps({k: v for k, v in args.items() if k not in long})}"
    if long:
        text += f"\nLONG ARGS, shown in parts below: {', '.join(long)}"
    return text


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


_BUILT_IN = {"injection": Injection, "constitution": Constitution}


def checks_from_dict(data: Mapping[str, Any], folder: str | None = None) -> list[Check]:
    """Checks from plain data (what a YAML or JSON file holds):

        {"checks": [
            {"check": "injection",
             "scope": {"tool_result": [{"tool": "search_web", "mode": "enforce"}]}},
            {"check": "constitution", "standard_file": "constitution.md"},
            {"name": "secrets_in_urls", "standard": "Never put a secret in a URL.",
             "scope": {"tool_call": [{"tool": "*", "mode": "shadow"}]}},
        ]}

    `check` picks a built-in check; leave it out for your own (then `name`,
    `standard` and `scope` are needed). `standard_file` is read relative to
    `folder` (the YAML file's folder). Every check is judged by Jev; build
    checks in code to use another judge. Unknown keys raise.
    """
    unknown = set(data) - {"checks"}
    if unknown:
        raise ValueError(f"unknown keys: {sorted(unknown)} (the file holds checks only)")
    checks = []
    for raw in data.get("checks") or []:
        raw = dict(raw)
        kind = raw.pop("check", None)
        if "standard_file" in raw:
            if "standard" in raw:
                raise ValueError("give a check standard or standard_file, not both")
            path = raw.pop("standard_file")
            with open(os.path.join(folder or "", path), encoding="utf-8") as f:
                raw["standard"] = f.read()
        if "scope" in raw:
            raw["scope"] = Scope.from_dict(raw["scope"])
        allowed = {"name", "standard", "scope"} if kind is None else {"standard", "scope"}
        bad = set(raw) - allowed
        if bad:
            raise ValueError(f"check {kind or raw.get('name')!r}: unknown keys {sorted(bad)}")
        if kind is None:
            missing = {"name", "standard", "scope"} - set(raw)
            if missing:
                raise ValueError(f"check {raw.get('name')!r}: missing {sorted(missing)}")
            checks.append(Check(**raw))
        elif kind in _BUILT_IN:
            checks.append(_BUILT_IN[kind](**raw))
        else:
            raise ValueError(f"unknown check {kind!r}; built-in checks are {sorted(_BUILT_IN)}")
    return checks


def checks_from_yaml(source: str) -> list[Check]:
    """`source` is a path to a YAML file, or YAML text."""
    data, folder = load_yaml(source)
    return checks_from_dict(data or {}, folder)


__all__ = ["INJECTION_STANDARD", "STAGES", "Check", "Constitution", "Injection", "call_pieces"]
