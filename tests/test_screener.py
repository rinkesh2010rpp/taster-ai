import json
import os

import pytest
from conftest import FakeDetector, verdict

from taster_ai import JsonlSink, Policy, Rule, Screener, Verdict
from taster_ai.screener import chunk, combine


def screener_for(detector, **rule_options):
    rule = Rule("search_web", **{"mode": "enforce", **rule_options})
    return Screener(Policy(rules=[rule], detector=detector))


@pytest.mark.parametrize(
    "label, confidence, rule_options, action",
    [
        ("clean", 0.99, {}, "passed"),
        ("injection", 0.97, {}, "withheld"),
        ("injection", 0.60, {}, "wrapped"),
        ("unclear", 0.70, {}, "wrapped"),
        ("unclear", 0.70, {"unclear": "withhold"}, "withheld"),
        ("unclear", 0.70, {"unclear": "pass"}, "passed"),
        ("injection", 0.97, {"threshold": 0.99}, "wrapped"),
    ],
)
def test_enforce_actions(fake, label, confidence, rule_options, action):
    d = screener_for(fake(label, confidence), **rule_options).screen("search_web", {}, "page")
    assert d.action == d.would_action == action


def test_detector_failure_follows_on_error():
    failing = FakeDetector(lambda _t: Verdict.failed("fake", "timeout"))
    assert screener_for(failing).screen("search_web", {}, "page").action == "labelled"
    assert screener_for(failing, on_error="block").screen("search_web", {}, "page").action == "withheld"


def test_shadow_never_changes_the_result(fake):
    d = screener_for(fake("injection", 1.0), mode="shadow").screen("search_web", {}, "page")
    assert d.action == "passed" and d.would_action == "withheld"
    assert d.render("page") is None


def test_pass_rules_and_empty_text_are_not_screened(fake):
    detector = fake("injection", 1.0)
    s = screener_for(detector)
    assert s.screen("read_file", {}, "anything") is None
    assert s.screen("search_web", {}, "   ") is None
    assert detector.calls == 0


def test_chunks_overlap_so_a_straddling_attack_is_seen_whole():
    size, overlap = 100, 30
    text = "a" * 90 + "EVILEVIL" + "b" * 150
    pieces = chunk(text, size, overlap)
    assert any("EVILEVIL" in p for p in pieces)
    assert "".join(p[: size - overlap] for p in pieces[:-1]) + pieces[-1] == text


def test_one_flagged_chunk_flags_the_whole_result():
    detector = FakeDetector(lambda t: verdict("injection", 0.95) if "EVIL" in t else verdict("clean", 0.99), max_chars=100)
    d = screener_for(detector).screen("search_web", {}, "x" * 120 + "EVIL" + "y" * 120)
    assert d.verdict.label == "injection" and d.action == "withheld" and d.chunks > 1


def test_too_many_chunks_is_unscreenable(fake):
    s = Screener(Policy(rules=[Rule("*", mode="enforce", on_error="block")], detector=fake(max_chars=10)), max_chunks=3)
    d = s.screen("t", {}, "z" * 500)
    assert d.verdict.error.startswith("too-large") and d.action == "withheld"


def test_combine_order():
    clean, unclear, bad = verdict("clean", 0.9), verdict("unclear", 0.6), verdict("injection", 0.8)
    failed = Verdict.failed("fake", "timeout")
    assert combine([clean, bad, failed]).label == "injection"
    assert combine([clean, failed, unclear]).label == "error"
    assert combine([clean, unclear]).label == "unclear"
    assert combine([clean, verdict("clean", 0.5)]).confidence == 0.5


def test_cache_judges_the_same_text_once(fake):
    detector = fake("clean")
    s = screener_for(detector)
    for _ in range(3):
        s.screen("search_web", {}, "same page")
    assert detector.calls == 1


def test_errors_are_not_cached():
    detector = FakeDetector(lambda _t: Verdict.failed("fake", "timeout"))
    s = screener_for(detector)
    s.screen("search_web", {}, "page")
    s.screen("search_web", {}, "page")
    assert detector.calls == 2


def test_internal_bug_is_contained_and_on_error_applies(fake, monkeypatch):
    s = screener_for(fake(), on_error="block")
    monkeypatch.setattr(s, "_verdict", lambda *a: 1 / 0)
    d = s.screen("search_web", {}, "page")
    assert d.verdict.error == "internal-ZeroDivisionError" and d.action == "withheld"


def test_sinks_get_a_record_without_the_text(tmp_path, fake):
    records = []

    def broken_sink(_):
        raise OSError("disk full")

    s = Screener(
        Policy(rules=[Rule("*", mode="shadow")], detector=fake("injection", 0.97)),
        sinks=[records.append, broken_sink, JsonlSink(str(tmp_path))],
    )
    s.screen("search_web", {"q": "x"}, "secret page text", tool_call_id="c1")
    rec = records[0]
    assert rec["tool_call_id"] == "c1" and rec["would_action"] == "withheld"
    assert "secret page text" not in json.dumps(rec)
    [logfile] = os.listdir(tmp_path)
    assert json.loads((tmp_path / logfile).read_text())["input_hash"] == rec["input_hash"]


def test_render_shapes(fake):
    s = screener_for(fake("unclear", 0.6))
    d = s.screen("search_web", {}, "page")
    assert d.render("page").startswith("[search_web result below is UNTRUSTED")
    blocks = [{"type": "text", "text": "a"}, {"type": "image", "url": "x"}]
    assert len(d.render(blocks)) == 4
    d = screener_for(fake("injection", 0.99)).screen("search_web", {}, "page")
    assert "withheld" in d.render(blocks)
