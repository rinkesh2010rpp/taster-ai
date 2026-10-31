"""LangChain (v1) agent middleware; works with deepagents too.

    from taster_ai import Policy, JevDetector, print_sink
    from taster_ai.adapters.langchain import TasterMiddleware

    taster = TasterMiddleware(
        policy=Policy.from_yaml("taster.yaml"),   # what to screen
        detector=JevDetector(),                   # who judges (or a chat model)
        sinks=[print_sink],                       # where to log
    )
    agent = create_agent(model, tools, middleware=[taster])

The middleware screens the agent it is attached to. To screen a subagent,
add it to that subagent's middleware too.

With `constitution=[...]` and rules that say checks=["constitution"], each
matching call is also checked against the owner's rules before it runs; in
enforce mode a violating call is not run and the agent is told why.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Mapping

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from ..actions import text_of
from ..policy import Policy
from ..screener import Screener
from ..sinks import Sink


class TasterMiddleware(AgentMiddleware):
    """Checks every tool call before it runs and screens every tool result
    before the model sees it, as the policy says."""

    def __init__(
        self,
        policy: Policy | None = None,
        detector: Any = None,
        *,
        detectors: Mapping[str, Any] | None = None,
        sinks: Iterable[Sink] | None = None,
        constitution: Any = None,
        screener: Screener | None = None,
        **screener_options: Any,
    ):
        """policy, detector, detectors, sinks, constitution and any other
        options are passed to Screener (see there); or pass a ready
        `screener`."""
        super().__init__()
        if screener is None:
            if policy is None:
                raise ValueError("pass a policy (or a ready screener)")
            screener = Screener(
                policy, detector, detectors=detectors, sinks=sinks, constitution=constitution, **screener_options
            )
        self.screener = screener

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
        decision = self.screener.check(name, tool_call.get("args") or {}, tool_call.get("id"))
        if decision is None or not decision.blocked:
            return None
        return ToolMessage(
            content=decision.message(self.screener.messages),
            tool_call_id=tool_call.get("id") or "",
            name=name,
            status="error",
        )

    def _apply(self, tool_call: dict, result: Any) -> Any:
        name = tool_call.get("name", "")
        if not isinstance(result, ToolMessage):
            return result  # a Command (state update) carries no fetched text
        decision = self.screener.screen(name, tool_call.get("args") or {}, text_of(result.content), tool_call.get("id"))
        if decision is None:
            return result
        content = decision.render(result.content, self.screener.messages)
        return result if content is None else result.model_copy(update={"content": content})

