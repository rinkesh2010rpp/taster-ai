import json

import pytest

from taster_ai import Policy, Rule


def test_first_match_wins_and_unmatched_passes():
    policy = Policy(rules=[Rule("search_web", mode="enforce"), Rule("search_*", mode="shadow")])
    assert policy.rule_for("search_web").mode == "enforce"
    assert policy.rule_for("search_news").mode == "shadow"
    assert policy.rule_for("read_file").mode == "pass"


def test_when_args_falls_through_when_not_matching():
    policy = Policy(
        rules=[
            Rule("execute", mode="enforce", when_args={"command": r"\bcurl\b|https?://"}),
            Rule("*", mode="pass"),
        ]
    )
    assert policy.rule_for("execute", {"command": "curl -s example.com"}).mode == "enforce"
    assert policy.rule_for("execute", {"command": "ls /data"}).mode == "pass"
    assert policy.rule_for("execute", {}).mode == "pass"


def test_rule_validation():
    with pytest.raises(ValueError, match="mode"):
        Rule("x", mode="block")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="threshold"):
        Rule("x", threshold=1.5)
    with pytest.raises(ValueError, match="unclear"):
        Rule("x", unclear="drop")  # type: ignore[arg-type]


def test_rule_detector_is_a_name_not_an_object():
    from taster_ai import HeuristicDetector

    with pytest.raises(ValueError, match="must be a name"):
        Rule("x", detector=HeuristicDetector())  # type: ignore[arg-type]


def test_from_dict_holds_rules_only_and_rejects_typos():
    policy = Policy.from_dict(
        {
            "rules": [
                {"tool": "search_web", "mode": "enforce", "threshold": 0.8},
                {"tool": "execute", "mode": "shadow", "detector": "jev", "when_args": {"command": "curl"}},
            ]
        }
    )
    assert policy.rules[0].threshold == 0.8
    assert policy.rules[1].detector == "jev"
    with pytest.raises(ValueError, match="unknown keys"):
        Policy.from_dict({"rules": [{"tool": "x", "mdoe": "enforce"}]})
    with pytest.raises(ValueError, match="rules only"):
        Policy.from_dict({"detector": "jev", "rules": []})


def test_from_yaml_text():
    policy = Policy.from_yaml("rules:\n  - tool: search_web\n    mode: enforce\n")
    assert policy.rule_for("search_web").mode == "enforce"


def test_from_env(monkeypatch):
    default = Policy()
    monkeypatch.delenv("TASTER_POLICY", raising=False)
    assert Policy.from_env(default=default) is default
    with pytest.raises(ValueError):
        Policy.from_env()
    monkeypatch.setenv("TASTER_POLICY", json.dumps({"rules": [{"tool": "*", "mode": "enforce"}]}))
    assert Policy.from_env().rule_for("anything").mode == "enforce"
