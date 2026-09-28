# Taster

**Your agent eats nothing we haven't tasted.**

Antivirus for your AI agent: every web page, file and tool result is screened
for prompt injection before your agent reads it. Poison gets spat out.

> Status: early alpha (0.1.0.dev0). APIs may change.

## Why

Agents read things they didn't write: search results, fetched pages, issues,
emails, command output. Any of it can carry text like *"AI assistant reading
this: ignore your previous instructions and…"*. Guarding your prompts isn't
enough when the attack arrives in a tool result.

Taster sits between your agent's tools and its model:

```
agent calls a tool ─▶ tool runs ─▶ Taster screens the result ─▶ model sees it
                                         │
                       clean → unchanged │ unsure → wrapped as untrusted
                                         │ injection → withheld
```

## Install

```bash
pip install "taster-ai[langchain]"      # or [deepagents]
```

## Quick start

Taster is one middleware. It needs three things:

| | What | Options |
|---|---|---|
| **policy** | *what* to screen and what to do | Python, YAML, JSON, or an environment variable |
| **detector** | *who* judges | the free heuristic (default), Jev, **your own chat model**, or a chain of them |
| **sinks** | *where* to log | print, logger (default), JSONL files, or any function |

```python
from langchain.agents import create_agent
from taster_ai import Policy, JevDetector, HeuristicDetector, FallbackDetector, print_sink
from taster_ai.adapters.langchain import TasterMiddleware

taster = TasterMiddleware(
    policy=Policy.from_yaml("taster.yaml"),
    detector=FallbackDetector(JevDetector(), HeuristicDetector()),
    sinks=[print_sink],
)

agent = create_agent(model, tools, middleware=[taster])
```

The smallest setup works with no API key at all:

```python
taster = TasterMiddleware(policy=Policy.from_yaml("taster.yaml"))   # heuristic detector, logger sink
```

## Policy: what to screen

A policy is **rules only**: plain config, no objects, no keys, safe to keep in
git. Rules are checked top to bottom; the first match wins; unmatched tools
are not screened.

```yaml
# taster.yaml
rules:
  - { tool: search_web, mode: enforce }
  - { tool: "mcp_*",    mode: enforce, detector: strict }   # a named detector, see below
  - { tool: execute,    mode: shadow, when_args: { command: 'curl|wget|https?://' } }
  - { tool: "*",        mode: pass }
```

| Field | Meaning |
|---|---|
| `tool` | tool name pattern (`search_web`, `mcp_*`, `*`) |
| `mode` | `pass` (not screened), `shadow` (screen + log only), `enforce` (screen + act) |
| `threshold` | confidence an `injection` verdict needs to withhold the result (default 0.85) |
| `unclear` | what an `unclear` verdict does: `wrap` (default), `withhold`, `pass` |
| `on_error` | if no verdict can be had: `label` (pass through, marked unscreened) or `block` |
| `when_args` | `{arg: regex}` — only match calls whose arguments match |
| `detector` | name of a detector from `detectors={...}`; omit to use the default |

Load it from anywhere:

```python
Policy.from_yaml("taster.yaml")                  # pip install "taster-ai[yaml]"
Policy.from_json(text)
Policy.from_env("TASTER_POLICY", default=...)    # JSON in an environment variable
Policy(rules=[Rule("search_web", mode="enforce"), Rule("*", mode="pass")])
```

**Start in shadow.** Shadow mode logs what enforce *would* do (`would_action`)
without changing anything, so you can check for false positives on real
traffic first.

**Keep the policy where the agent can't write it.** An agent that can edit its
own policy can be talked into switching its screen off.

## Detector: who judges

| Detector | What it is | Cost |
|---|---|---|
| `HeuristicDetector()` | patterns real attacks use: override phrases, fake system/owner messages, hiding things from the user, secrets, directives in HTML comments, invisible Unicode | free, instant, offline |
| `JevDetector()` | [Jev](https://openrouter.ai) decision model via OpenRouter; calibrated probabilities | ~0.2–0.4 s; needs `OPENROUTER_API_KEY` |
| **your chat model** | any LangChain chat model (OpenAI, Anthropic, Gemini, Ollama…), wrapped in `LLMDetector` automatically | your model's price and speed |
| `FallbackDetector(a, b, …)` | try each in order; the first that answers wins | — |
| `CallableDetector(fn)` | any function `text -> (label, confidence)` | yours |

```python
from langchain_openai import ChatOpenAI

detector = ChatOpenAI(model="gpt-4.1-mini", temperature=0)                   # your LLM as the judge
detector = FallbackDetector(JevDetector(), my_llm, HeuristicDetector())       # all three, in order
```

Different tools can use different judges: register them by name and refer to
them from rules.

```python
TasterMiddleware(
    policy=policy,
    detector=JevDetector(),                                  # default
    detectors={"strict": my_llm, "free": HeuristicDetector()},
)
```

**Using your own LLM as the judge:** the judge reads the same untrusted text,
so a page can try to talk it round. `LLMDetector` fences the text between
random markers, tells the model the text is evidence and never instructions,
forces a `{label, confidence, reason}` answer and gives the model no tools.
Still, use a model **other than your agent's** (ideally a small one), and
note its confidence is its own estimate, not a calibrated probability.

The heuristic is a baseline and last-resort fallback: it misses reworded
attacks, one signal on its own is only `unclear`, and text in double quotes is
ignored (so articles *about* injection aren't flagged).

### API keys

`JevDetector` reads `OPENROUTER_API_KEY` from the environment (or
`api_key=` from your own secret store). Never put keys in policy files. If the
agent has a shell tool that sees environment variables, it can read the key
too: use a separate key with a spending limit.

## Sinks: where records go

A sink is any function that takes one dict. After every screened call, each
sink gets a record:

```json
{"tool": "search_web", "rule": "search_web", "mode": "enforce", "label": "injection",
 "confidence": 0.97, "detector": "jev", "action": "withheld", "would_action": "withheld",
 "latency_s": 0.21, "input_hash": "3be1c0a9d2f4a1b7", ...}
```

It holds a **hash** of the text, never the text itself.

```python
from taster_ai import print_sink, log_sink, JsonlSink, BackgroundSink

sinks=[print_sink]                          # one printed line per call
sinks=[JsonlSink("logs/taster")]            # full record, one file per day per process
sinks=[print_sink, BackgroundSink(to_slack)]  # slow sinks run off the agent's path
sinks=[]                                    # nothing (default is [log_sink])
```

Sinks run before the model sees the result, so wrap network calls in
`BackgroundSink`. A failing sink is logged and skipped; it never breaks a turn.

## deepagents

deepagents does not pass your middleware to its built-in `general-purpose`
subagent, and subagents call tools too. Add one line so they're covered:

```python
agent = create_deep_agent(
    model=llm,
    tools=tools,
    middleware=[summarization, taster],
    subagents=taster.subagents(inherit=[summarization]),  # or taster.protect(your_subagents)
)
```

A declared subagent inherits the main agent's model and tools but **not** its
middleware, so list in `inherit` anything it should keep. If you forget
`subagents=`, Taster logs a warning the first time the agent delegates.

## Without an agent framework

```python
from taster_ai import Screener

screener = Screener(policy, detector)
decision = screener.screen("read_email", {}, email_text)
if decision and decision.action == "withheld":
    ...
```

## Guarantees

- **Never breaks a turn.** Detector failures, timeouts and bugs become
  `error` verdicts, handled by the rule's `on_error`.
- **Long results** are split into overlapping chunks, screened in parallel;
  one flagged chunk flags the whole result. Results too long to screen fully
  count as unscreenable rather than half-checked.
- **Repeats are cheap:** verdicts are cached by text hash.
- **The core has no dependencies.** LangChain is only needed for the
  middleware and `LLMDetector`.

## Known gaps

- Content an agent downloads to a file and reads later is only screened if a
  rule covers the reading tool. File provenance tracking is planned.
- Detection is probabilistic. Taster reduces risk; keep least-privilege
  tools and human approval for dangerous actions.

## Roadmap

- **0.2** — local PromptGuard 2 detector, voting ensembles, a public benchmark
  with false-positive rates
- **0.3** — file provenance tracking, an MCP proxy that screens any MCP server

## License

Apache-2.0
