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
    """Screens every tool result before the model sees it."""

    def __init__(
        self,
        policy: Policy | None = None,
        detector: Any = None,
        *,
        detectors: Mapping[str, Any] | None = None,
        sinks: Iterable[Sink] | None = None,
        screener: Screener | None = None,
        **screener_options: Any,
    ):
        """policy, detector, detectors, sinks and any other options are
        passed to Screener (see there); or pass a ready `screener`."""
        super().__init__()
        if screener is None:
            if policy is None:
                raise ValueError("pass a policy (or a ready screener)")
            screener = Screener(policy, detector, detectors=detectors, sinks=sinks, **screener_options)
        self.screener = screener

    def wrap_tool_call(self, request, handler):
        return self._apply(request.tool_call, handler(request))

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        return await asyncio.to_thread(self._apply, request.tool_call, result)

    def _apply(self, tool_call: dict, result: Any) -> Any:
        name = tool_call.get("name", "")
        if not isinstance(result, ToolMessage):
            return result  # a Command (state update) carries no fetched text
        decision = self.screener.screen(name, tool_call.get("args") or {}, text_of(result.content), tool_call.get("id"))
        if decision is None:
            return result
        content = decision.render(result.content, self.screener.messages)
        return result if content is None else result.model_copy(update={"content": content})

