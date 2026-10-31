"""The owner's rules, checked against every tool call before it runs.

    from taster_ai import Constitution

    constitution = Constitution([
        "Never write secrets, API keys or tokens to files.",
        "Never send a secret to any host except the one that issued it.",
    ])

Injection screening asks "is this text trying to hijack the reader?" about a
tool's result. The constitution asks "does this action break the owner's
rules?" about a tool's call, whoever suggested it. Each call is judged on its
own: the judge sees the rules and the call, nothing else.

The judge is any detector asked the constitution question. Detectors answer
with Taster's labels, read here as: clean = follows, injection = violates,
unclear = unclear. Jev is the default; a LangChain chat model works too.

Telling the agent its rules is up to whoever builds the agent (put them in
the system prompt). Taster only checks. As with the policy, keep the rules
where the agent cannot edit them.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .detectors import Detector, JevDetector, as_detector

QUESTION = (
    "Below is an action an AI agent is about to take, and the constitution its "
    "owner wrote. Does the action break any rule of the constitution?\n\n"
    "Constitution:\n{rules}"
)
CRITERIA = {
    "clean": "Follows: the action breaks no rule of the constitution.",
    "injection": "Violates: the action breaks at least one rule of the constitution.",
    "unclear": "Can't tell from the action alone whether it breaks a rule.",
}
#: How the record names each label for a constitution check.
LABELS = {"clean": "follows", "injection": "violates", "unclear": "unclear", "error": "error"}


def describe(tool: str, args: Mapping[str, Any], max_chars: int) -> str:
    """The text the judge reads for one call. Over `max_chars`, each long
    argument keeps its start and end, so a short call is shown whole and a
    huge `write_file` still shows where it writes and what it begins with."""
    text = f"ACTION: {tool}\nARGS: {json.dumps(dict(args), ensure_ascii=False, default=str)}"
    if len(text) <= max_chars:
        return text
    budget = max(200, (max_chars - len(tool) - 100) // max(1, len(args)))
    trimmed = {}
    for key, value in args.items():
        value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if len(value) > budget:
            half = budget // 2
            value = f"{value[:half]} …[{len(value) - 2 * half} chars cut]… {value[-half:]}"
        trimmed[key] = value
    return f"ACTION: {tool}\nARGS: {json.dumps(trimmed, ensure_ascii=False)}"[:max_chars]


class Constitution:
    """The owner's rules and the judge that checks calls against them.

    rules         plain sentences, one rule each
    judge         a Detector already asking the constitution question, or a
                  LangChain chat model (asked it via LLMDetector); None uses
                  Jev with `jev_options` (timeout, title, referer, ...)
    """

    def __init__(self, rules: Iterable[str], judge: Any = None, **jev_options: Any):
        self.rules = [str(r).strip() for r in rules if str(r).strip()]
        if not self.rules:
            raise ValueError("a constitution needs at least one rule")
        self.question = QUESTION.format(rules="\n".join(f"{i}. {r}" for i, r in enumerate(self.rules, 1)))
        if judge is None:
            self.judge: Detector = JevDetector(question=self.question, criteria=CRITERIA, **jev_options)
        elif isinstance(judge, Detector):
            self.judge = judge
        else:
            as_detector(judge)  # raises TypeError unless it's a chat model
            from .detectors.llm import LLMDetector

            self.judge = LLMDetector(judge, question=self.question, criteria=CRITERIA)

    def describe(self, tool: str, args: Mapping[str, Any]) -> str:
        return describe(tool, args, self.judge.max_chars)
