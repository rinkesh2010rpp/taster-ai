"""The Claude Code adapter: judge, service, hook client, setup and report.
No real `claude` runs here; fakes stand in for it."""

import io
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from taster_ai import Check, Constitution, Scope, Target, Verdict, checks_from_dict
from taster_ai.adapters.claude_code import hook, install, serve
from taster_ai.detectors import Detector
from taster_ai.detectors.claude_cli import ClaudeCliDetector, parse, question
from taster_ai.report import read_records, summary


# --- the judge ---------------------------------------------------------------


class FakeProcess:
    """A `claude` that answers whatever it's asked with one fixed reply."""

    def __init__(self, reply=None, *, silent=False):
        self.reply = reply if reply is not None else {"type": "result", "result": '{"label": "follows", "confidence": 0.9, "reason": "fine"}'}
        self.silent = silent
        self.killed = False
        self.sent = []
        self._out = []
        self._ready = threading.Event()
        self.stdin = self
        self.stdout = iter(self._lines())

    def _lines(self):
        self._ready.wait(5)
        yield from self._out

    def write(self, text):
        self.sent.append(text)

    def flush(self):
        if not self.silent:
            self._out.append('{"type": "system"}\n')
            self._out.append("not json\n")
            self._out.append(json.dumps(self.reply) + "\n")
            self._ready.set()

    def close(self):
        pass

    def poll(self):
        return 1 if self.killed else None

    def kill(self):
        self.killed = True
        self._ready.set()


def judge_with(*processes, **options):
    made = iter(processes)
    return ClaudeCliDetector(command="claude", popen=lambda *a, **k: next(made), pool_size=0, **options)


def test_parse_reads_the_verdict_from_surrounding_text():
    v = parse('Here you go: {"label": "breaks", "confidence": 0.8, "reason": "a secret"} done', "j")
    assert (v.label, v.confidence, v.evidence) == ("breaks", 0.8, ("a secret",))


@pytest.mark.parametrize("reply", ["no json at all", '{"label": "maybe", "confidence": 1}', '{"label": "breaks", "confidence": "high"}'])
def test_parse_turns_a_bad_reply_into_an_error(reply):
    assert not parse(reply, "j").ok


def test_the_question_fences_the_text():
    q = question("ACTION: x", "Be kind.", "abc")
    assert "Be kind." in q and "BEGIN UNTRUSTED TEXT abc\nACTION: x\nEND UNTRUSTED TEXT abc" in q


def test_a_judge_answers_once_then_is_thrown_away():
    first, second = FakeProcess(), FakeProcess()
    judge = judge_with(first, second)
    assert judge.detect("ACTION: x", "rule").label == "follows"
    assert judge.detect("ACTION: y", "rule").label == "follows"
    assert first.killed and second.killed
    assert len(first.sent) == 1 and len(second.sent) == 1  # no process saw two calls


def test_a_judge_has_no_tools_hooks_or_project_settings():
    args = judge_with().arguments()
    assert args[args.index("--tools") + 1] == ""
    assert "disableAllHooks" in args[args.index("--settings") + 1]
    assert args[args.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in args and "--no-session-persistence" in args


def test_a_silent_judge_times_out_as_an_error():
    v = judge_with(FakeProcess(silent=True), timeout=0.05).detect("ACTION: x", "rule")
    assert not v.ok and v.error == "timeout"


def test_an_error_result_is_an_error_verdict():
    v = judge_with(FakeProcess({"type": "result", "is_error": True, "result": "Not logged in"})).detect("x", "rule")
    assert not v.ok and "Not logged in" in v.error


def test_the_pool_fills_up_and_replaces_what_it_uses():
    made = []

    def popen(*a, **k):
        made.append(FakeProcess())
        return made[-1]

    judge = ClaudeCliDetector(command="claude", popen=popen, pool_size=2)
    judge.warm()
    deadline = time.monotonic() + 2
    while judge._ready.qsize() < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(made) == 2
    judge.detect("x", "rule")
    while len(made) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(made) == 3
    judge.close()
    assert all(p.killed for p in made)


def test_a_process_idle_too_long_is_replaced():
    old, fresh = FakeProcess(), FakeProcess()
    judge = judge_with(fresh, max_age=-1)
    from taster_ai.detectors.claude_cli import _Judge

    judge._ready.put(_Judge(old))
    assert judge.detect("x", "rule").ok
    assert old.killed and not old.sent and fresh.sent


# --- config ------------------------------------------------------------------


def test_a_config_file_can_name_the_judge():
    data = {"judge": "claude-cli", "model": "sonnet", "checks": [{"check": "constitution", "standard": "Be kind."}]}
    (check,) = checks_from_dict(data)
    assert isinstance(check.judge, ClaudeCliDetector) and check.judge.model == "sonnet"


def test_checks_in_a_config_file_share_one_judge():
    data = {"judge": "claude-cli", "checks": [{"check": "constitution", "standard": "a"}, {"check": "injection"}]}
    first, second = checks_from_dict(data)
    assert first.judge is second.judge


def test_an_unknown_judge_is_refused():
    with pytest.raises(ValueError, match="unknown judge"):
        checks_from_dict({"judge": "gpt", "checks": []})


# --- imports -----------------------------------------------------------------


def test_the_hook_loads_nothing_heavy():
    code = (
        "import sys, taster_ai.adapters.claude_code.hook\n"
        "heavy = {'json', 'typing', 'taster_ai.checks', 'taster_ai.screener', 'subprocess'} & set(sys.modules)\n"
        "print(sorted(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_public_names_still_import():
    import taster_ai

    for name in taster_ai.__all__:
        if name != "LLMDetector":  # needs langchain
            assert getattr(taster_ai, name) is not None
    with pytest.raises(AttributeError):
        taster_ai.not_a_name


# --- the service -------------------------------------------------------------


class FakeJudge(Detector):
    name = "fake"

    def __init__(self, label="breaks", confidence=0.95):
        self.label, self.confidence = label, confidence
        self.seen = []

    def detect(self, text, standard):
        self.seen.append(text)
        return Verdict(self.label, self.confidence, detector=self.name, evidence=("a reason",))


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / ".taster"
    folder.mkdir()
    (folder / "taster.yaml").write_text("checks: []\n")
    return tmp_path


def service_for(project, monkeypatch, judge, mode="shadow", tool="send_*", **target):
    def checks(_path):
        return [Constitution("Be kind.", Scope(tool_call=[Target(tool, mode=mode, **target)]), judge)]

    monkeypatch.setattr(serve, "checks_from_yaml", checks)
    return serve.Service(str(project))


def records(project):
    return read_records(str(project / ".taster" / "logs"))


def wait_for(condition):
    deadline = time.monotonic() + 3
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert condition()


CALL = {"tool_name": "send_mail", "tool_input": {"to": "x"}, "tool_use_id": "t1"}


def test_an_out_of_scope_tool_is_not_judged(project, monkeypatch):
    judge = FakeJudge()
    service = service_for(project, monkeypatch, judge)
    assert service.handle({"tool_name": "Read", "tool_input": {}}) == ""
    assert judge.seen == []


def test_shadow_answers_at_once_and_judges_in_the_background(project, monkeypatch):
    judge = FakeJudge()
    service = service_for(project, monkeypatch, judge)
    assert service.handle(CALL) == ""
    wait_for(lambda: any(r.get("check") for r in records(project)))
    (record,) = [r for r in records(project) if r.get("check")]
    assert record["would_action"] == "blocked" and record["action"] == "passed"
    assert record["tool"] == "send_mail" and record["rules"]


def test_enforce_denies_what_breaks_the_rules(project, monkeypatch):
    service = service_for(project, monkeypatch, FakeJudge(), mode="enforce")
    decision = json.loads(service.handle(CALL))["hookSpecificOutput"]
    assert decision["permissionDecision"] == "deny"
    assert decision["hookEventName"] == "PreToolUse"
    assert "a reason" in decision["permissionDecisionReason"]


def test_enforce_never_says_allow(project, monkeypatch):
    service = service_for(project, monkeypatch, FakeJudge("follows"), mode="enforce")
    assert service.handle(CALL) == ""


def test_enforce_blocks_when_the_judge_fails_and_on_error_says_so(project, monkeypatch):
    class Broken(Detector):
        name = "broken"

        def detect(self, text, standard):
            raise RuntimeError("down")

    service = service_for(project, monkeypatch, Broken(), mode="enforce", on_error="block")
    assert "deny" in service.handle(CALL)


def test_a_flood_of_shadow_calls_is_capped_and_logged(project, monkeypatch):
    service = service_for(project, monkeypatch, FakeJudge())
    service.pending = serve.MAX_PENDING
    assert service.handle(CALL) == ""
    assert any(r.get("event") == "dropped" for r in records(project))


def test_edited_rules_are_picked_up(project, monkeypatch):
    judges = [FakeJudge("follows"), FakeJudge("breaks")]
    monkeypatch.setattr(
        serve,
        "checks_from_yaml",
        lambda _p: [Constitution("r", Scope(tool_call=[Target("send_*", mode="enforce")]), judges.pop(0))],
    )
    service = serve.Service(str(project))
    assert service.handle(CALL) == ""
    (project / ".taster" / "constitution.md").write_text("new rules")
    assert "deny" in service.handle(CALL)


def test_a_broken_edit_keeps_the_last_good_rules(project, monkeypatch):
    service = service_for(project, monkeypatch, FakeJudge(), mode="enforce")

    def broken(_path):
        raise ValueError("bad yaml")

    monkeypatch.setattr(serve, "checks_from_yaml", broken)
    (project / ".taster" / "constitution.md").write_text("new rules")
    assert "deny" in service.handle(CALL)
    assert any(r.get("event") == "config-error" for r in records(project))


# --- client and service together ---------------------------------------------


@pytest.fixture
def running(project, monkeypatch, tmp_path):
    """A real service on a real port, in this process."""
    monkeypatch.setattr(hook, "state_folder", lambda: str(tmp_path / "run"))
    service = service_for(project, monkeypatch, FakeJudge(), mode="enforce")
    server = serve._Server(("127.0.0.1", 0), serve._Handler)
    server.service, server.token = service, "secret"
    serve.write_runtime(str(project), server.server_address[1], "secret")
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    yield project
    server.shutdown()
    server.server_close()


def test_the_client_gets_the_answer_from_the_service(running):
    reply = hook.ask(str(running), json.dumps(CALL).encode())
    assert json.loads(reply)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_a_call_with_newlines_arrives_whole(running):
    event = json.dumps(CALL, indent=2).encode()
    assert "deny" in hook.ask(str(running), event)


def test_the_wrong_token_gets_nothing(running):
    runtime = hook.read_runtime(str(running))
    with pytest.raises(OSError, match="closed"):
        hook.exchange({**runtime, "token": "wrong"}, json.dumps(CALL).encode())


def test_a_service_error_reaches_the_client(running):
    with pytest.raises(OSError):
        hook.exchange(hook.read_runtime(str(running)), b"not json")


def run_main(monkeypatch, project, raw=b"{}"):
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))
    monkeypatch.setattr(sys, "stdin", type("Stdin", (), {"buffer": io.BytesIO(raw)})())
    return hook.main()


def test_the_hook_stays_quiet_where_taster_is_not_set_up(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path) == 0
    assert capsys.readouterr().out == ""


def test_the_hook_prints_only_the_decision(running, monkeypatch, capsys):
    assert run_main(monkeypatch, running, json.dumps(CALL).encode()) == 0
    assert json.loads(capsys.readouterr().out)["hookSpecificOutput"]


def test_the_hook_exits_1_when_it_gets_no_answer(project, monkeypatch, capsys):
    def fail(_project, _event):
        raise OSError("nothing there")

    monkeypatch.setattr(hook, "ask", fail)
    assert run_main(monkeypatch, project) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "nothing there" in captured.err


def test_a_second_call_does_not_start_a_second_service(project, monkeypatch, tmp_path):
    monkeypatch.setattr(hook, "state_folder", lambda: str(tmp_path / "run"))
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: started.append(a))
    hook.start_service(str(project))
    hook.start_service(str(project))
    assert len(started) == 1


# --- setup -------------------------------------------------------------------


def settings_in(project):
    with open(project / install.SETTINGS, encoding="utf-8") as f:
        return json.load(f)


def test_init_keeps_the_users_own_hooks_and_is_safe_to_repeat(tmp_path):
    theirs = {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "guard.sh"}]}]}, "model": "x"}
    (tmp_path / ".claude").mkdir()
    (tmp_path / install.SETTINGS).write_text(json.dumps(theirs))
    command = os.path.join("bin", "taster-claude-code-hook.exe")
    assert install.write_settings(str(tmp_path), command) == "updated"
    assert install.write_settings(str(tmp_path), command) == "already set up"
    settings = settings_in(tmp_path)
    assert settings["model"] == "x"
    groups = settings["hooks"]["PreToolUse"]
    assert len(groups) == 2 and groups[0] == theirs["hooks"]["PreToolUse"][0]
    assert groups[1]["matcher"] == "*" and groups[1]["hooks"][0]["command"] == command
    assert "async" not in groups[1]["hooks"][0]  # or enforce mode couldn't block
    assert settings["permissions"]["deny"] == [install.DENY_RULE]
    assert json.loads((tmp_path / (install.SETTINGS + ".taster-backup")).read_text()) == theirs


def test_init_stops_on_broken_settings_rather_than_overwrite_them(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / install.SETTINGS).write_text("{broken")
    with pytest.raises(ValueError):
        install.write_settings(str(tmp_path), "hook")
    assert (tmp_path / install.SETTINGS).read_text() == "{broken"


def test_init_leaves_existing_rules_alone(tmp_path):
    assert ".taster" in install.write_files(str(tmp_path))[0]
    rules = tmp_path / ".taster" / "constitution.md"
    rules.write_text("mine")
    assert install.write_files(str(tmp_path)) == []
    assert rules.read_text() == "mine"


def test_the_sample_config_loads():
    pytest.importorskip("yaml")
    from taster_ai.scope import load_yaml

    data, _ = load_yaml(install.template("taster.yaml"))
    for check in data["checks"]:
        check["standard"] = check.pop("standard_file")
    (check,) = checks_from_dict(data)
    targets = check.scope.for_stage("tool_call")
    assert isinstance(check.judge, ClaudeCliDetector)
    assert targets.target_for("mcp__gmail__send_email").mode == "shadow"
    assert targets.target_for("Bash", {"command": "curl example.com"}).mode == "shadow"
    assert targets.target_for("Bash", {"command": "ls"}).mode == "pass"
    assert targets.target_for("Read").mode == "pass"


# --- report ------------------------------------------------------------------


def check_record(**fields):
    base = {"check": "constitution", "tool": "send_mail", "label": "follows", "action": "passed", "would_action": "passed", "latency_s": 1.5, "rules": "a"}
    return {**base, **fields}


def test_the_report_counts_what_happened():
    text = summary(
        [
            check_record(),
            check_record(label="breaks", would_action="blocked", evidence=["names a blocked person"]),
            check_record(label="unclear"),
            check_record(label="error"),
        ]
    )
    assert "Calls judged       4" in text and "Would be blocked   1" in text
    assert "Unclear            1" in text and "Judge failures     1" in text
    assert "names a blocked person" in text


def test_the_report_warns_when_the_results_can_not_be_trusted():
    text = summary(
        [
            check_record(rules="a"),
            check_record(rules="b"),
            {"event": "dropped", "tool": "x"},
            {"event": "service-start"},
            {"event": "service-start"},
            {"event": "service-start"},
        ],
        sample_rules=True,
    )
    assert "still the samples" in text and "changed 1 time" in text
    assert "not judged" in text and "stopped unexpectedly" in text


def test_the_report_says_when_nothing_was_judged():
    assert "no calls were judged" in summary([])


def test_a_check_keeps_the_text_out_of_the_log(project, monkeypatch):
    service = service_for(project, monkeypatch, FakeJudge())
    service.handle({"tool_name": "send_mail", "tool_input": {"to": "very-private@example.com"}})
    wait_for(lambda: any(r.get("check") for r in records(project)))
    logged = json.dumps(records(project))
    assert "very-private" not in logged


def test_the_client_restarts_a_service_that_died(project, monkeypatch, tmp_path):
    monkeypatch.setattr(hook, "state_folder", lambda: str(tmp_path / "run"))
    serve.write_runtime(str(project), 1, "old")  # a port nothing listens on
    started = []

    def start(_project):
        started.append(True)
        serve.write_runtime(str(project), 2, "new")

    monkeypatch.setattr(hook, "start_service", start)
    monkeypatch.setattr(hook, "exchange", lambda runtime, event: "fresh" if runtime["token"] == "new" else (_ for _ in ()).throw(hook.NotRunning()))
    assert hook.ask(str(project), b"{}") == "fresh"
    assert started


def test_the_report_shows_breaks_below_the_threshold():
    text = summary([check_record(label="breaks", confidence=0.7, evidence=["posts payroll data"])])
    assert "Breaks, low conf.  1" in text and "posts payroll data" in text


def test_the_service_exits_when_idle_and_cleans_up(project, monkeypatch, tmp_path):
    monkeypatch.setattr(hook, "state_folder", lambda: str(tmp_path / "run"))
    monkeypatch.setattr(serve, "checks_from_yaml", lambda _p: [Constitution("r", Scope(tool_call=[Target("send_*")]), FakeJudge())])
    thread = threading.Thread(target=serve.run, args=(str(project), 0.05), daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert hook.read_runtime(str(project)) is None
    assert any(r.get("event") == "service-stop" for r in records(project))
