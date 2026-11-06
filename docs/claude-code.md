# Taster for Claude Code

Check what Claude Code does against your own rules, before it does it.
Back to the [README](../README.md).

This is experimental. It's for one person's own machine, and it needs Claude
Code installed and logged in. There's no API key to set up: the judge is the
Claude Code CLI on your own login, so checks count against your Claude plan.


## 1. Install

```bash
pipx install "taster-ai[claude-code] @ git+https://github.com/rinkesh2010rpp/taster-ai@v0.3.0.dev0"
```

Needs Python 3.10+. `pip install` works too.

If `taster` isn't found afterwards, run `pipx ensurepath` and open a new
terminal.


## 2. Set up a project

```bash
cd your-project
taster init claude-code
```

This creates three things, and asks the judge a test question to confirm your
login works:

| File | What it is |
|---|---|
| `.taster/constitution.md` | Your rules, in plain English |
| `.taster/taster.yaml` | Which tools are checked, and how strictly |
| `.claude/settings.local.json` | The hook, and a rule that keeps Claude's file tools out of `.taster/` |

It keeps any hooks you already have, saves a backup of your settings, and
leaves existing `.taster/` files alone, so it's safe to run twice. Nothing in
`.taster/` is committed, because rules can name real people.


## 3. Write your rules

Replace the samples in `.taster/constitution.md`, one plain sentence per rule:

```markdown
Never send secrets, API keys or passwords to anyone or anywhere.
Never contact dana@example.com by any channel, including calendar invites.
```

Each call is judged on its own, so a rule has to be checkable from one call.
Keep the list short: all of it goes to the judge on every check. Don't put
secrets in it.

Edit this file yourself. Claude's file tools can't change it once the project
is set up.


## 4. Choose what's checked

**A rule only applies to the tools listed in `taster.yaml`.** Nothing else is
judged. The sample covers the common ways things leave your machine: mail,
calendar, messaging and send tools, `WebFetch`, and Bash commands like `curl`,
`git push`, `gh` and `npm publish`.

```yaml
judge: claude-cli
model: haiku

checks:
  - check: constitution
    standard_file: constitution.md
    scope:
      tool_call:
        - { tool: "mcp__*mail*", mode: shadow }
        - { tool: WebFetch, mode: shadow }
        - tool: Bash
          mode: shadow
          when_args: { command: '\b(curl|wget|gh)\b' }
```

Tool names use wildcards. `when_args` is a regular expression on an argument,
and the first matching entry wins.

If you add a rule about deleting data, add the tools that delete: `Write`,
`Edit`, and Bash with `'\b(rm|rmdir)\b'`.

Patterns can't catch everything. `python send.py` isn't caught unless you list
it, because nothing in the command says it sends data. Every judged call uses
your plan, so list the tools that can do harm, not every tool.

Changes apply on the next tool call, with no restart.


## 5. Watch, then switch on

Use Claude Code as normal. Everything starts in shadow mode: calls are judged
in the background and logged, and nothing is blocked.

```bash
taster report                     # the last 14 days
taster report --days 3
taster report --export log.jsonl  # the records, to share
```

```text
Calls judged       412
Would be blocked   6
Blocked            0
Breaks, low conf.  2
Unclear            3
Judge failures     1
Median check time  2.0 s
```

Logs are in `.taster/logs/`. They hold tool names, verdicts and the judge's
reasons, not the calls themselves, but a reason can quote a call, so read an
export before sharing it.

To block, set `mode: enforce` on a tool:

```yaml
- tool: "mcp__*mail*"
  mode: enforce
  on_error: block
  threshold: 0.6
```

- `on_error: block` also blocks a call the judge can't answer.
- `threshold` is how sure the judge must be. The default of 0.85 let real
  breaks through in testing, because the judge's confidence moves around.
  `taster report` lists breaks below the threshold under "Breaks, low conf."

A blocked call goes back to Claude with the reason, so it can change course.
Taster only ever blocks or stays silent. It never approves anything, so your
normal permission prompts still apply.


## How it works

```text
Claude Code
   │  hook, on every tool call
   ▼
taster-claude-code-hook     small client, starts and exits each call
   │  127.0.0.1, with a token
   ▼
background service         one per project
   │
   ▼
claude CLIs                 started ahead of time, one check each
```

The hook starts the service when it's needed, and the service exits after 30
idle minutes, so there's nothing to run or manage. Each judge has no tools, no
hooks and no project settings, and answers once, so one call can't sway the
verdict on the next.


## What leaves your machine

Taster is an open-source package, not a service. It doesn't collect anything,
and it has no servers.

The judge runs on your own Claude login, so checks go to Anthropic under your
account, the same way Claude Code's own requests do. That includes your rules
and the tool calls being judged. If that text can't leave your machine, don't
use this judge on that data until there's a local option.


## Speed

Measured on one Windows PC:

| Step | Time |
|---|---|
| The hook, on any tool call | About 0.35 s |
| A checked call, shadow mode | No extra wait: judged in the background |
| A checked call, enforce mode | About 2 s more, longer in a burst |

The sample scope judged about 11% of the tool calls in one person's coding and
research history. Your numbers will differ.


## Limits

- **It stops mistakes and common attacks, not a determined one.** The agent
  runs as you, so a hijacked agent could stop the service, or change `.taster/`
  through a script. For a hard guarantee, run the agent in a sandbox or as
  another user.
- **If the hook fails, the call goes ahead.** That includes a broken
  `taster.yaml`, even for tools in enforce mode. Add `"onFailure": "block"` to
  the hook in your settings to change that (needs Claude Code 2.1.295 or
  later), but it applies to every tool, so a stopped service then blocks
  everything. That setting hasn't been tested yet.
- **Haiku's confidence is rough.** It ranged from 0.7 to 0.97 on clear breaks.
- **Tested on Windows only.** Not yet tested on Mac or Linux, with real MCP
  tools, or with judges left idle for long periods.
- **It depends on `claude -p` using your login.** Claude Code's docs say
  `--bare`, which needs an API key, will become the default for `-p`. If that
  happens, checks will show up as judge failures until Taster is updated.
- **It uses your plan.** Those limits assume ordinary, individual use, so keep
  your scope narrow.


## Removing it

1. Delete the Taster entries from `hooks` and `permissions.deny` in
   `.claude/settings.local.json`. If the project had a settings file before,
   your earlier version is at `.claude/settings.local.json.taster-backup`. If
   it didn't, there's no backup, and you can delete the file if nothing else
   is in it.
2. Delete `.taster/`.
3. Uninstall the package, if no other project uses it:

   ```bash
   pipx uninstall taster-ai
   ```

   Use `pip uninstall taster-ai` if you installed it with pip.

The background service exits on its own after 30 idle minutes. Once it has,
you can delete `~/.taster/` too.
