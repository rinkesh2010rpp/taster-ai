"""What `taster init claude-code` writes into a project.

Files that exist are left alone, and the hook is merged into the user's
settings, never written over them: a project may already have its own hooks.
Running it twice changes nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from importlib import resources
from typing import Any

SETTINGS = os.path.join(".claude", "settings.local.json")
HOOK_NAME = "taster-claude-code-hook"
#: Keeps Claude's file tools off the rules. Scripts can still reach them;
#: `taster report` warns when the rules change.
DENY_RULE = "Edit(/.taster/**)"
HOOK_TIMEOUT_S = 60


def hook_command() -> str:
    """The hook's full path, so it works whatever PATH Claude Code runs with."""
    folder = os.path.dirname(sys.executable)
    found = shutil.which(HOOK_NAME, path=folder) or shutil.which(HOOK_NAME)
    if not found:
        raise FileNotFoundError(f"{HOOK_NAME} is not installed; reinstall taster-ai")
    return found


def template(name: str) -> str:
    return resources.files(__package__).joinpath("templates", name).read_text(encoding="utf-8")


def merge_settings(settings: dict[str, Any], command: str) -> bool:
    """Add the hook and the deny rule; True if anything changed."""
    changed = False
    groups = settings.setdefault("hooks", {}).setdefault("PreToolUse", [])
    ours = [
        hook
        for group in groups
        for hook in group.get("hooks", [])
        if HOOK_NAME in os.path.basename(str(hook.get("command", "")))
    ]
    if not ours:
        groups.append(
            {
                "matcher": "*",
                "hooks": [{"type": "command", "command": command, "args": [], "timeout": HOOK_TIMEOUT_S}],
            }
        )
        changed = True
    deny = settings.setdefault("permissions", {}).setdefault("deny", [])
    if DENY_RULE not in deny:
        deny.append(DENY_RULE)
        changed = True
    return changed


def write_settings(project: str, command: str) -> str:
    """Merge into the project's local settings, keeping a backup."""
    path = os.path.join(project, SETTINGS)
    settings: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            settings = json.load(f)  # bad JSON stops here, before anything is written
    if not merge_settings(settings, command):
        return "already set up"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        shutil.copyfile(path, path + ".taster-backup")
    temporary = path + ".taster-new"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    os.replace(temporary, path)
    return "updated"


def write_files(project: str) -> list[str]:
    """Create .taster/ from the samples; the names of the files created."""
    folder = os.path.join(project, ".taster")
    os.makedirs(folder, exist_ok=True)
    created = []
    # The rules can name real people, so nothing here is committed by default.
    for name, text in (
        ("taster.yaml", template("taster.yaml")),
        ("constitution.md", template("constitution.md")),
        (".gitignore", "*\n"),
    ):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            created.append(os.path.join(".taster", name))
    return created


def check_judge() -> str | None:
    """Ask the judge one question: None if it answered, else what's wrong."""
    from ...detectors.claude_cli import ClaudeCliDetector

    judge = ClaudeCliDetector(pool_size=0)
    try:
        verdict = judge.safe_detect(
            'ACTION: send_email\nARGS: {"to": "blocked@example.com"}', "Never email blocked@example.com."
        )
    finally:
        judge.close()
    if verdict.ok:
        return None
    return verdict.error or "the judge did not answer"


def init(project: str) -> int:
    print(f"Setting up Taster for Claude Code in {project}")
    for name in write_files(project):
        print(f"  created {name}")
    print(f"  {SETTINGS}: {write_settings(project, hook_command())}")
    print("Asking the judge a test question...")
    problem = check_judge()
    if problem:
        print(f"  The judge didn't answer: {problem}")
        print("  Is Claude Code installed and logged in? Run `claude` once, then try again.")
        return 1
    print("  The judge answered.")
    print("\nNext: write your rules in .taster/constitution.md, then use Claude Code as normal.")
    print("Everything starts in shadow mode: checked and logged, nothing blocked.")
    print("See what happened with: taster report")
    return 0
