"""LangChain (v1) agent middleware; works with deepagents too.

    from taster_ai import Constitution, Injection, Scope, Target
    from taster_ai.adapters.langchain import TasterMiddleware

    taster = TasterMiddleware(checks=[
        Injection(scope=Scope(tool_result=[Target("search_web", mode="enforce")])),
        Constitution(standard=open("constitution.md").read()),
    ])
    agent = create_agent(model, tools, middleware=[taster])

Before each tool runs, every check with a tool_call stage looks at the call;
if one in enforce mode blocks it, the tool is not run and the agent is told
which checks it failed. After it runs, every check with a tool_result stage
looks at the result before the model does.

The middleware checks the agent it is attached to. To check a subagent,
add it to that subagent's middleware too.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Mapping

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from ..actions import text_of
from ..checks import Check, checks_from_yaml
from ..policy import Policy
from ..screener import Screener
from ..sinks import Sink


class TasterMiddleware(AgentMiddleware):
    """Checks every tool call before it runs and every tool result before
    the model sees it, as the checks' scopes say."""

    def __init__(
        self,
        policy: Policy | None = None,
        detector: Any = None,
        *,
        checks: Iterable[Check] | None = None,
        detectors: Mapping[str, Any] | None = None,
        sinks: Iterable[Sink] | None = None,
        screener: Screener | None = None,
        **screener_options: Any,
    ):
        """checks (or the 0.1 policy, detector, detectors), sinks and any
        other options are passed to Screener (see there); or pass a ready
        `screener`."""
        super().__init__()
        if screener is None:
            screener = Screener(
                policy, detector, checks=checks, detectors=detectors, sinks=sinks, **screener_options
            )
        self.screener = screener

    @classmethod
    def from_yaml(cls, source: str, **options: Any) -> TasterMiddleware:
        """Checks from a YAML file (or YAML text); see checks_from_dict."""
        return cls(checks=checks_from_yaml(source), **options)

    def wrap_tool_call(self, request, handler):
        blocked = self._check(request.tool_call)
        if blocked is not None:
            return blocked
        return self._apply(request.tool_call, handler(request))

    async def awrap_tool_call(self, request, handler):
        blocked = await asyncio.to_thread(self._check, request.tool_call)
        if blocked is not None:
            return blocked
        result = await handler(request)
        return await asyncio.to_thread(self._apply, request.tool_call, result)

    def _check(self, tool_call: dict) -> ToolMessage | None:
        """The message that replaces a blocked call's result, or None to run it."""
        name = tool_call.get("name", "")
        outcome = self.screener.screen_call(name, tool_call.get("args") or {}, tool_call.get("id"))
        if outcome is None or not outcome.blocked:
            return None
        return ToolMessage(
            content=outcome.message(self.screener.messages),
            tool_call_id=tool_call.get("id") or "",
            name=name,
            status="error",
        )

    def _apply(self, tool_call: dict, result: Any) -> Any:
        name = tool_call.get("name", "")
        if not isinstance(result, ToolMessage):
            return result  # a Command (state update) carries no fetched text
        outcome = self.screener.screen_result(
            name, tool_call.get("args") or {}, text_of(result.content), tool_call.get("id")
        )
        if outcome is None:
            return result
        content = outcome.render(result.content, self.screener.messages)
        return result if content is None else result.model_copy(update={"content": content})
