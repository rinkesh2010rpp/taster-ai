---
name: taster-claude-code
description: Set up Taster in a project so Claude Code's own tool calls are checked against the owner's rules through a hook, with no API key. Use when someone asks to keep Claude Code within rules, add guardrails or a safety check to Claude Code, stop Claude Code sending or leaking things, or set up the `taster` command in a project.
---

# Taster for Claude Code

Taster checks Claude Code's tool calls against the owner's rules, in plain
English. A hook passes each call to a small background service, and the judge
is the Claude Code CLI on the owner's own login, so there's no API key.
Repo: https://github.com/rinkesh2010rpp/taster-ai

## When to use

The owner wants limits on Claude Code itself, in this project. If they're
building their own agent with LangChain or deepagents, use the `taster` skill
instead. This is experimental, and it's tested on Windows only; say so.

## Steps

1. **Write the rules with the owner first.** Do this before step 3, because
   afterwards your file tools can't change `.taster/`. Look at what this
   project's agent can do: MCP servers, and what it sends or publishes. Propose
   a few rules and ask the owner to confirm or edit them; never invent rules
   silently.

   ```markdown
   Never send secrets, API keys or passwords to anyone or anywhere.
   Never post private data to a public place.
   ```

   Write rules as limits on actions ("never", "only"), one sentence each. Each
   call is judged on its own, so a rule must be checkable from one call. Keep
   the list short, and don't put secrets in it: it goes to the judge. Rules can
   name real people, so `.taster/` isn't committed.

   Save the confirmed rules to `.taster/constitution.md`.

2. **Install.** Check `claude --version` works, since the judge needs it. Ask
   before installing anything, then:

   ```bash
   pipx install "taster-ai[claude-code] @ git+https://github.com/rinkesh2010rpp/taster-ai"
   ```

   Needs Python 3.10+. `pip install` works too.

3. **Set up the project**, from its root:

   ```bash
   taster init claude-code
   ```

   It adds the hook to `.claude/settings.local.json`, keeping existing hooks,
   and creates what's missing in `.taster/`. Your rules from step 1 stay as
   they are. It ends by asking the judge a test question. If the judge doesn't
   answer, the owner needs to run `claude` once and log in; never handle their
   login yourself.

4. **Match the scope to the rules.** A rule only applies to the tools listed in
   `.taster/taster.yaml`, so this is the step that decides whether the rules do
   anything. Read that file and tell the owner which rules it can't enforce.
   For example, a rule about deleting data needs `Write`, `Edit`, and Bash with
   `'\b(rm|rmdir)\b'`, which the sample doesn't list. The owner edits this file
   themselves, because your file tools can't.

5. **Check it runs.** The hook applies to the next tool call. Tell the owner to
   use Claude Code as normal for a while, then run:

   ```bash
   taster report
   ```

   "Calls judged" should be above zero. If it says no calls were judged, the
   scope doesn't match the tools in use.

6. **Switching to enforce is the owner's call.** Everything starts in shadow
   mode: judged and logged, nothing blocked. Suggest watching `taster report`
   on real work first. Then they set, per tool, `mode: enforce`,
   `on_error: block` and `threshold: 0.6`. Don't do this for them.

## Don't

- Don't remove, disable or work around the hook, the deny rule or `.taster/` to
  finish a task. A blocked call means the action broke a rule: find another way
  or ask the owner.
- Don't change the rules or scope through a script or shell command to get past
  the deny rule. Only the owner changes them.
- Don't promise it catches everything. Patterns miss things like `python
  send.py`, and the judge can be wrong. It stops mistakes and common attacks,
  not a determined one.

## More

- [Claude Code guide](https://github.com/rinkesh2010rpp/taster-ai/blob/main/docs/claude-code.md):
  every step, speed, limits, and how to remove it
