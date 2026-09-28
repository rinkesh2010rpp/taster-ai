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

## Quick start (LangChain)

```python
from langchain.agents import create_agent
from taster_ai import HeuristicDetector, JevDetector, FallbackDetector, Policy, Rule
from taster_ai.adapters.langchain import TasterMiddleware

policy = Policy(
    detector=FallbackDetector(JevDetector(), HeuristicDetector()),
    rules=[
        Rule("search_web", mode="enforce"),
        Rule("execute", mode="shadow", when_args={"command": r"\bcurl\b|\bwget\b|https?://"}),
        Rule("*", mode="pass"),
    ],
)

agent = create_agent(model, tools, middleware=[TasterMiddleware(policy)])
```

## deepagents

deepagents does not pass your middleware to its built-in `general-purpose`
subagent, and subagents call tools too. Add one line so they're covered:

```python
taster = TasterMiddleware(policy)

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

## Rules

Rules are checked top to bottom; the first match wins. Unmatched tools pass.

| Field | Meaning |
|---|---|
| `tool` | tool name pattern (`search_web`, `mcp_*`, `*`) |
| `mode` | `pass` (not screened), `shadow` (screen + log only), `enforce` (screen + act) |
| `threshold` | confidence an `injection` verdict needs to withhold the result (default 0.85) |
| `unclear` | what an `unclear` verdict does: `wrap` (default), `withhold`, `pass` |
| `on_error` | if no verdict can be had: `label` (pass through, marked unscreened) or `block` |
| `when_args` | `{arg: regex}` — only match calls whose arguments match |
| `detector` | a different detector for this rule |

**Start in shadow.** Shadow mode logs what enforce *would* do (`would_action`)
without changing anything, so you can check for false positives on real
traffic first.

Policies can also come from YAML or JSON:

```yaml
detector: { type: fallback, primary: jev, fallback: heuristic }
rules:
  - { tool: search_web, mode: enforce }
  - { tool: execute, mode: shadow, when_args: { command: "curl|wget|https?://" } }
  - { tool: "*", mode: pass }
```

```python
policy = Policy.from_yaml("taster.yaml")      # pip install "taster-ai[yaml]"
policy = Policy.from_env("TASTER_POLICY")      # JSON in an environment variable
```

**Keep the policy where the agent can't write it.** An agent that can edit its
own policy can be talked into switching its screen off.

## Detectors

| Detector | What it is | Cost |
|---|---|---|
| `JevDetector` | [Jev](https://openrouter.ai) decision model via OpenRouter; asks *"is this an attempt to hijack the reader?"* | ~0.2–0.4 s, fractions of a cent; needs `OPENROUTER_API_KEY` |
| `HeuristicDetector` | patterns real attacks use: override phrases, fake system/owner messages, hiding things from the user, secrets, directives in HTML comments, invisible Unicode | free, instant, offline |
| `FallbackDetector(a, b)` | use `a`; if it fails, use `b` | — |

Write your own by subclassing `Detector` and implementing `detect(text, context) -> Verdict`.

The heuristic is a baseline and fallback, not a full detector: it misses
reworded attacks, one signal on its own is only `unclear`, and text in double
quotes is ignored (so articles *about* injection aren't flagged).

## Logging

```python
from taster_ai import JsonlSink, print_sink
TasterMiddleware(policy, sinks=[JsonlSink("logs/taster"), print_sink])
```

Each screened call produces a record with the tool, rule, mode, label,
confidence, scores, action, `would_action`, latency and a **hash** of the
text (never the text itself).

## Guarantees

- **Never breaks a turn.** Detector failures, timeouts and bugs become
  `error` verdicts, handled by the rule's `on_error`.
- **Long results** are split into overlapping chunks, screened in parallel;
  one flagged chunk flags the whole result. Results too long to screen fully
  count as unscreenable rather than half-checked.
- **Repeats are cheap:** verdicts are cached by text hash.

## Known gaps

- Content an agent downloads to a file and reads later is only screened if a
  rule covers the reading tool. File provenance tracking is planned.
- Detection is probabilistic. Taster reduces risk; keep least-privilege
  tools and human approval for dangerous actions.

## Roadmap

- **0.2** — local PromptGuard 2 detector, ensembles, a public benchmark with
  false-positive rates
- **0.3** — file provenance tracking, an MCP proxy that screens any MCP server

## License

Apache-2.0
