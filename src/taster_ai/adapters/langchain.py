"""LangChain (v1) agent middleware, which also covers deepagents.

    from taster_ai.adapters.langchain import TasterMiddleware

    taster = TasterMiddleware(policy)
    agent = create_agent(model, tools, middleware=[taster])

deepagents does not hand new middleware to its built-in "general-purpose"
subagent, and subagents call tools too. Pass `taster.subagents()` so it is
covered (and list any other middleware it should inherit):

    agent = create_deep_agent(
        model=llm, tools=tools,
        middleware=[summarization, taster],
        subagents=taster.subagents(inherit=[summarization]),
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from ..actions import text_of
from ..policy import Policy
from ..screener import Screener
from ..sinks import Sink

logger = logging.getLogger("taster_ai")
_SUBAGENT_TOOL = "task"  # deepagents' tool for handing work to a subagent


class TasterMiddleware(AgentMiddleware):
    """Screens every tool result before the model sees it."""

    def __init__(self, policy: Policy | None = None, *, screener: Screener | None = None, sinks: Iterable[Sink] = (), **screener_options: Any):
        super().__init__()
        if screener is None:
            if policy is None:
                raise ValueError("pass a policy or a screener")
            screener = Screener(policy, sinks=sinks, **screener_options)
        self.screener = screener
        self._covers_subagents = False
        self._warned = False

    def wrap_tool_call(self, request, handler):
        return self._apply(request.tool_call, handler(request))

    async def awrap_tool_call(self, request, handler):
        result = await handler(request)
        return await asyncio.to_thread(self._apply, request.tool_call, result)

    def _apply(self, tool_call: dict, result: Any) -> Any:
        name = tool_call.get("name", "")
        if name == _SUBAGENT_TOOL and not self._covers_subagents and not self._warned:
            self._warned = True
            logger.warning(
                "taster: the agent handed work to a subagent, but subagents are not screened. "
                "With deepagents pass subagents=taster.subagents() (or taster.protect(your_subagents))."
            )
        if not isinstance(result, ToolMessage):
            return result  # a Command (state update) carries no fetched text
        decision = self.screener.screen(name, tool_call.get("args") or {}, text_of(result.content), tool_call.get("id"))
        if decision is None:
            return result
        content = decision.render(result.content, self.screener.messages)
        return result if content is None else result.model_copy(update={"content": content})

    # --- deepagents helpers ---------------------------------------------

    def subagents(self, inherit: Iterable[AgentMiddleware] = ()) -> list[dict]:
        """deepagents' stock general-purpose subagent, screened.

        An explicitly declared subagent inherits the main agent's model and
        tools but NOT its middleware, so list in `inherit` any middleware it
        should keep (e.g. a customised SummarizationMiddleware).
        """
        from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

        self._covers_subagents = True
        return [{**GENERAL_PURPOSE_SUBAGENT, "middleware": [*inherit, self]}]

    def protect(self, subagents: Iterable[dict], inherit: Iterable[AgentMiddleware] = ()) -> list[dict]:
        """Add the screen to your own subagent specs, and include a screened
        general-purpose subagent unless you declared one."""
        inherit = list(inherit)
        protected = []
        for spec in subagents:
            if "runnable" in spec:  # a pre-compiled subagent: we can't reach inside
                logger.warning("taster: subagent %r is pre-compiled; add TasterMiddleware to it yourself", spec.get("name"))
                protected.append(spec)
                continue
            protected.append({**spec, "middleware": [*inherit, *spec.get("middleware", []), self]})
        if not any(spec.get("name") == "general-purpose" for spec in protected):
            protected = self.subagents(inherit) + protected
        self._covers_subagents = True
        return protected
