import json

import pytest

from taster_ai import FallbackDetector, HeuristicDetector, JevDetector, Policy, Rule


def test_first_match_wins_and_unmatched_passes(fake):
    policy = Policy(
        rules=[Rule("search_web", mode="enforce"), Rule("search_*", mode="shadow")],
        detector=fake(),
    )
    assert policy.rule_for("search_web").mode == "enforce"
    assert policy.rule_for("search_news").mode == "shadow"
    assert policy.rule_for("read_file").mode == "pass"


def test_when_args_falls_through_when_not_matching(fake):
    policy = Policy(
        rules=[
            Rule("execute", mode="enforce", when_args={"command": r"\bcurl\b|https?://"}),
            Rule("*", mode="pass"),
        ],
        detector=fake(),
    )
    assert policy.rule_for("execute", {"command": "curl -s example.com"}).mode == "enforce"
    assert policy.rule_for("execute", {"command": "ls /data"}).mode == "pass"
    assert policy.rule_for("execute", {}).mode == "pass"


def test_screening_rule_needs_a_detector():
    with pytest.raises(ValueError, match="no detector"):
        Policy(rules=[Rule("search_web", mode="shadow")])
    Policy(rules=[Rule("*", mode="pass")])  # pass-only is fine


def test_rule_validation():
    with pytest.raises(ValueError, match="mode"):
        Rule("x", mode="block")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="threshold"):
        Rule("x", threshold=1.5)
    with pytest.raises(ValueError, match="unclear"):
        Rule("x", unclear="drop")  # type: ignore[arg-type]


def test_from_dict_builds_detectors_and_rejects_typos():
    policy = Policy.from_dict(
        {
            "detector": {"type": "fallback", "primary": {"type": "jev", "timeout": 3}, "fallback": "heuristic"},
            "rules": [
                {"tool": "search_web", "mode": "enforce", "threshold": 0.8},
                {"tool": "execute", "mode": "shadow", "detector": "heuristic", "when_args": {"command": "curl"}},
            ],
        }
    )
    assert isinstance(policy.detector, FallbackDetector)
    assert isinstance(policy.detector.primary, JevDetector) and policy.detector.primary.timeout == 3
    assert isinstance(policy.rules[1].detector, HeuristicDetector)
    with pytest.raises(ValueError, match="unknown keys"):
        Policy.from_dict({"detector": "heuristic", "rules": [{"tool": "x", "mdoe": "enforce"}]})
    with pytest.raises(ValueError, match="unknown policy keys"):
        Policy.from_dict({"detectors": "heuristic"})


def test_from_dict_uses_named_detectors(fake):
    mine = fake()
    policy = Policy.from_dict({"detector": "mine", "rules": [{"tool": "*"}]}, detectors={"mine": mine})
    assert policy.detector is mine


def test_from_yaml_text():
    policy = Policy.from_yaml("detector: heuristic\nrules:\n  - tool: search_web\n    mode: enforce\n")
    assert policy.rule_for("search_web").mode == "enforce"


def test_from_env(monkeypatch, fake):
    default = Policy(rules=[], detector=fake())
    monkeypatch.delenv("TASTER_POLICY", raising=False)
    assert Policy.from_env(default=default) is default
    with pytest.raises(ValueError):
        Policy.from_env()
    monkeypatch.setenv("TASTER_POLICY", json.dumps({"detector": "heuristic", "rules": [{"tool": "*", "mode": "enforce"}]}))
    assert Policy.from_env().rule_for("anything").mode == "enforce"
