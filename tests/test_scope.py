import json

import pytest

from taster_ai import Injection, Scope, Target, checks_from_env, checks_from_json
from taster_ai.scope import Targets


def test_first_match_wins_and_unmatched_passes():
    targets = Targets([Target("search_web", mode="enforce"), Target("search_*", mode="shadow")])
    assert targets.target_for("search_web").mode == "enforce"
    assert targets.target_for("search_news").mode == "shadow"
    assert targets.target_for("read_file").mode == "pass"


def test_when_args_falls_through_when_not_matching():
    targets = Targets(
        [
            Target("execute", mode="enforce", when_args={"command": r"\bcurl\b|https?://"}),
            Target("*", mode="pass"),
        ]
    )
    assert targets.target_for("execute", {"command": "curl -s example.com"}).mode == "enforce"
    assert targets.target_for("execute", {"command": "ls /data"}).mode == "pass"
    assert targets.target_for("execute", {}).mode == "pass"


def test_target_validation():
    with pytest.raises(ValueError, match="mode"):
        Target("x", mode="block")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="threshold"):
        Target("x", threshold=1.5)
    with pytest.raises(ValueError, match="unclear"):
        Target("x", unclear="drop")  # type: ignore[arg-type]


def test_target_detector_is_a_name_not_an_object():
    from taster_ai import InjectionPatterns

    with pytest.raises(ValueError, match="must be a name"):
        Target("x", detector=InjectionPatterns())  # type: ignore[arg-type]


def test_scope_from_dict_rejects_typos():
    scope = Scope.from_dict(
        {
            "tool_call": [{"tool": "execute", "mode": "shadow", "detector": "jev", "when_args": {"command": "curl"}}],
            "tool_result": [{"tool": "search_web", "mode": "enforce", "threshold": 0.8}],
        }
    )
    assert scope.tool_result[0].threshold == 0.8 and scope.tool_call[0].detector == "jev"
    with pytest.raises(ValueError, match="unknown keys"):
        Scope.from_dict({"tool_call": [{"tool": "x", "mdoe": "enforce"}]})
    with pytest.raises(ValueError, match="unknown scope stages"):
        Scope.from_dict({"rules": []})


def test_checks_from_json_and_env(monkeypatch):
    data = {"checks": [{"check": "injection", "scope": {"tool_result": [{"tool": "*", "mode": "enforce"}]}}]}
    [c] = checks_from_json(json.dumps(data))
    assert isinstance(c, Injection) and c.scope.tool_result[0].mode == "enforce"

    default = [Injection()]
    monkeypatch.delenv("TASTER_CHECKS", raising=False)
    assert checks_from_env(default=default) is default
    with pytest.raises(ValueError, match="TASTER_CHECKS is not set"):
        checks_from_env()
    monkeypatch.setenv("TASTER_CHECKS", json.dumps(data))
    assert checks_from_env()[0].scope.tool_result[0].mode == "enforce"
