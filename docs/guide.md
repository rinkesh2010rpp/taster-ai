# Taster guide

How to set Taster up, step by step. Back to the [README](../README.md);
every option is in the [reference](reference.md).


## 1. Install

Install from GitHub, pinned to a release:

```bash
pip install "taster-ai[langchain,yaml] @ git+https://github.com/rinkesh2010rpp/taster-ai@v0.2.0.dev0"
```

Requires Python 3.10+ and, for the middleware, LangChain 1.x. (0.1.0.dev3 has
the older `policy=` setup, documented in that tag's README.)

## 2. Write your constitution

Plain sentences, the way you'd say them to a person:

```markdown
Never write secrets, API keys or tokens to files.
Never send a secret to any host except the one that issued it.
Never install skills or instruction files from external sources.
```

Taster only checks. Telling the agent its rules (in its system prompt) is up
to you.

## 3. Choose your checks

```python
import logging
from langchain.agents import create_agent
from taster_ai import Constitution, Injection, Scope, Target
from taster_ai.adapters.langchain import TasterMiddleware

logging.basicConfig(level=logging.INFO)   # to see Taster's log lines

taster = TasterMiddleware(checks=[
    Constitution(standard=open("constitution.md").read()),   # every call, in shadow
    Injection(scope=Scope(tool_result=[
        Target("search_web", mode="shadow"),
        Target("fetch_url",  mode="shadow"),
        Target("execute",    mode="shadow", when_args={"command": "curl|wget|https?://"}),
    ])),
])
agent = create_agent(model, tools, middleware=[taster])
```

For injection, list the tools that bring in **outside content**. Start
everything in `shadow`. Both checks are judged by Jev by default, which needs
`OPENROUTER_API_KEY`; see step 5 for other judges.

Or keep the checks in a file:

```yaml
# taster.yaml
checks:
  - check: constitution
    standard_file: constitution.md     # read next to this file
  - check: injection
    scope:
      tool_result:
        - { tool: search_web, mode: shadow }
        - { tool: fetch_url,  mode: shadow }
```

```python
taster = TasterMiddleware.from_yaml("taster.yaml")
```

## 4. Watch, then switch on

Each check logs a line per call, like:

```
constitution tool_call write_file mode=shadow label=breaks conf=0.94 action=passed would=blocked …
injection tool_result search_web mode=shadow label=breaks conf=0.97 action=passed would=withheld …
```

`would=` is what Taster would have done. When the logs look right, change
`shadow` to `enforce` for that target.

## 5. Choose the judge (optional)

```python
from taster_ai import FallbackDetector, InjectionPatterns, JevDetector

Injection(scope=..., judge=FallbackDetector(JevDetector(), InjectionPatterns()))   # Jev, free fallback
Constitution(standard=text, judge=ChatOpenAI(model="gpt-4.1-mini", temperature=0)) # your own LLM
```

`InjectionPatterns()` is free and needs no key, but only judges prompt
injection: it ignores the standard. Use it in `Injection`, usually as a
fallback behind Jev.

## Using deepagents?

Taster screens the agent you attach it to. deepagents doesn't pass your
middleware to its built-in subagent, which also reads web pages, so declare
that subagent and give it Taster too:

```python
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT

agent = create_deep_agent(
    model=llm, tools=tools,
    middleware=[taster],
    subagents=[{**GENERAL_PURPOSE_SUBAGENT, "middleware": [taster]}],
)
```

A declared subagent gets the main agent's model and tools but **not** its
middleware, so list there anything else it should keep (e.g. a customised
`SummarizationMiddleware`). Your own subagents work the same way.

## Environment variables

| Variable | Needed when | Without it |
|---|---|---|
| `OPENROUTER_API_KEY` | you use `JevDetector()` without `api_key=` | Jev can't answer; a fallback detector takes over, or the target's `on_error` applies |
| `TASTER_CHECKS` (or a name you choose) | you load checks with `checks_from_env()` | your default checks are used |

`Injection()` and `Constitution()` are judged by Jev by default, so they need
`OPENROUTER_API_KEY`. Your own LLM judge needs its provider's key
(e.g. `OPENAI_API_KEY`), as LangChain usually does.
