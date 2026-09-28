"""Rules: which tool results get screened, and what happens next.

A policy is plain configuration: rules only, no detectors, no keys. That
makes it safe to keep in git and to load from YAML, JSON or an environment
variable. The detectors that do the judging are given to the Screener (or
middleware) separately; a rule can name one of them.

Keep the policy somewhere the agent cannot write (your code, a config file
outside the agent's workspace, or an environment variable). An agent that
can edit its own policy can be talked into switching its screen off.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Mode = Literal["pass", "shadow", "enforce"]
MODES = ("pass", "shadow", "enforce")
ON_ERROR = ("label", "block")
UNCLEAR = ("wrap", "withhold", "pass")


@dataclass
class Rule:
    """How to treat results from tools whose name matches `tool`.

    tool       fnmatch pattern on the tool name ("search_web", "mcp_*", "*")
    mode       "pass"    — not screened
               "shadow"  — screened and logged, result unchanged
               "enforce" — screened, logged, and acted on
    threshold  minimum confidence for an "injection" verdict to withhold
               the result; below it the result is wrapped as untrusted
    on_error   when no verdict can be had: "label" passes the result through
               marked unscreened, "block" withholds it
    unclear    what an "unclear" verdict does: "wrap", "withhold" or "pass"
    when_args  {arg_name: regex}; the rule matches only if every named
               argument matches (re.search). A call that doesn't match falls
               through to later rules.
    detector   name of a detector registered with the screener
               (`detectors={"name": ...}`); None uses the default detector
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
            raise ValueError(f"rule {self.tool!r}: mode must be one of {MODES}, got {self.mode!r}")
        if self.on_error not in ON_ERROR:
            raise ValueError(f"rule {self.tool!r}: on_error must be one of {ON_ERROR}")
        if self.unclear not in UNCLEAR:
            raise ValueError(f"rule {self.tool!r}: unclear must be one of {UNCLEAR}")
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError(f"rule {self.tool!r}: threshold must be between 0 and 1")
        if self.detector is not None and not isinstance(self.detector, str):
            raise ValueError(f"rule {self.tool!r}: detector must be a name; register the object with the screener")
        self._arg_patterns = {arg: re.compile(pattern) for arg, pattern in (self.when_args or {}).items()}

    def matches(self, tool: str, args: Mapping[str, Any]) -> bool:
        if not fnmatch.fnmatchcase(tool, self.tool):
            return False
        return all(p.search(str((args or {}).get(arg, ""))) for arg, p in self._arg_patterns.items())


PASS = Rule("*", mode="pass")
_RULE_KEYS = {"tool", "mode", "threshold", "on_error", "unclear", "when_args", "detector"}


@dataclass
class Policy:
    """An ordered list of rules; the first match wins.

    A tool call that matches no rule is passed unscreened; end the list with
    a `Rule("*", ...)` to choose otherwise.
    """

    rules: list[Rule] = field(default_factory=list)

    def rule_for(self, tool: str, args: Mapping[str, Any] | None = None) -> Rule:
        for rule in self.rules:
            if rule.matches(tool, args or {}):
                return rule
        return PASS

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Policy:
        """Build from plain data (what a YAML or JSON file holds):

            {"rules": [{"tool": "search_web", "mode": "enforce"},
                       {"tool": "mcp_*", "mode": "enforce", "detector": "jev"},
                       {"tool": "*", "mode": "pass"}]}

        Unknown keys raise, so a typo can't silently switch screening off.
        """
        unknown = set(data) - {"rules"}
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)} (a policy holds rules only)")
        rules = []
        for raw in data.get("rules") or []:
            bad = set(raw) - _RULE_KEYS
            if bad:
                raise ValueError(f"rule {raw.get('tool')!r}: unknown keys {sorted(bad)}")
            rules.append(Rule(**raw))
        return cls(rules=rules)

    @classmethod
    def from_json(cls, text: str) -> Policy:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, source: str) -> Policy:
        """`source` is a path to a YAML file, or YAML text."""
        import yaml  # optional dependency: pip install taster-ai[yaml]

        if "\n" not in source and os.path.exists(source):
            with open(source, encoding="utf-8") as f:
                source = f.read()
        return cls.from_dict(yaml.safe_load(source) or {})

    @classmethod
    def from_env(cls, var: str = "TASTER_POLICY", default: Policy | None = None) -> Policy:
        """Load JSON from an environment variable, or return `default` if it's unset."""
        raw = os.environ.get(var, "").strip()
        if not raw:
            if default is None:
                raise ValueError(f"{var} is not set and no default policy was given")
            return default
        return cls.from_json(raw)
