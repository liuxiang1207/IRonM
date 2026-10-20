"""Agent 输出与字段归属（spec [2026-10-20] §4）。"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ironm.contracts.agents import AGENT_OUTPUT_ADAPTER, AgentOutput
from ironm.contracts.plan import FIELD_OWNERS, PlanField
from ironm.contracts.roles import AGENT_OUTPUT_KINDS, AgentId, OutputKind


def _proposal(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "proposal_id": "p1",
        "field": "daily_kcal_target",
        "from_value": 1900,
        "to_value": 1750,
        "reason_facts": ["metric:weight_trend"],
        "rule_id": "R-DIET-ADJ-4",
        "expected_effect": "两周内体重趋势达到每周 -0.3 kg",
        "rollback_condition": "饥饿感连续两周 ≥ 4 分",
        "review_date": "2026-11-03",
        "evidence_ids": ["EV-AHA-2013"],
    }
    data.update(overrides)
    return data


def _output(agent: str, kind: str, items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"agent": agent, "skill": "diet_review@1.0.0", "kind": kind}
    if items is not None:
        data["items"] = items
    return data


def _parse(data: dict[str, Any]) -> AgentOutput:
    return AGENT_OUTPUT_ADAPTER.validate_json(json.dumps(data))


def test_every_field_has_one_proposal_owner() -> None:
    assert set(FIELD_OWNERS) == set(PlanField)
    for field, owner in FIELD_OWNERS.items():
        assert OutputKind.PROPOSAL in AGENT_OUTPUT_KINDS[owner], field


def test_ten_agents() -> None:
    assert len(AgentId) == 10
    assert set(AGENT_OUTPUT_KINDS) == set(AgentId)


def test_owner_proposal_accepted() -> None:
    out = _parse(_output("nutritionist", "proposal", [_proposal()]))
    assert out.kind == "proposal"


def test_first_time_setting_has_null_from() -> None:
    _parse(_output("nutritionist", "proposal", [_proposal(from_value=None)]))


def test_cross_owner_proposal_rejected() -> None:
    bad = _proposal(field="weekly_cardio_minutes", from_value=90, to_value=120)
    with pytest.raises(ValidationError, match=r"weekly_cardio_minutes.*training_planner"):
        _parse(_output("nutritionist", "proposal", [bad]))


@pytest.mark.parametrize(
    ("agent", "kind", "items"),
    [
        ("nutritionist", "verdict", [{"proposal_id": "p1", "result": "pass"}]),
        ("evidence_reviewer", "proposal", [_proposal()]),
        ("training_planner", "constraint", []),
    ],
)
def test_kind_not_allowed_for_agent(agent: str, kind: str, items: list[dict[str, Any]]) -> None:
    with pytest.raises(ValidationError):
        _parse(_output(agent, kind, items))


@pytest.mark.parametrize("agent", [a.value for a in AgentId])
def test_any_agent_may_have_no_opinion(agent: str) -> None:
    _parse(_output(agent, "no_opinion"))


@pytest.mark.parametrize("missing", ["expected_effect", "rollback_condition", "review_date"])
def test_adjustment_triple_required(missing: str) -> None:
    p = _proposal()
    del p[missing]
    with pytest.raises(ValidationError):
        _parse(_output("nutritionist", "proposal", [p]))


def test_empty_triple_text_rejected() -> None:
    with pytest.raises(ValidationError):
        _parse(_output("nutritionist", "proposal", [_proposal(expected_effect="")]))


def test_no_change_is_not_a_proposal() -> None:
    with pytest.raises(ValidationError):
        _parse(_output("nutritionist", "proposal", [_proposal(to_value=1900)]))


def test_same_field_twice_rejected() -> None:
    items = [_proposal(), _proposal(proposal_id="p2", to_value=1800)]
    with pytest.raises(ValidationError):
        _parse(_output("nutritionist", "proposal", items))


def test_proposal_output_needs_items() -> None:
    with pytest.raises(ValidationError):
        _parse(_output("nutritionist", "proposal", []))


def test_skill_must_be_versioned() -> None:
    data = _output("nutritionist", "no_opinion") | {"skill": "diet_review"}
    with pytest.raises(ValidationError):
        _parse(data)


@pytest.mark.parametrize("result", ["reject", "warn"])
def test_negative_verdict_needs_reason(result: str) -> None:
    with pytest.raises(ValidationError):
        _parse(_output("evidence_reviewer", "verdict", [{"proposal_id": "p1", "result": result}]))
    _parse(
        _output(
            "evidence_reviewer",
            "verdict",
            [{"proposal_id": "p1", "result": result, "reason_code": "EVIDENCE_WEAK"}],
        )
    )


def test_pass_verdict_without_reason() -> None:
    _parse(_output("quality_reviewer", "verdict", [{"proposal_id": "p1", "result": "pass"}]))


def test_constraint_accepted() -> None:
    item = {
        "field": "daily_kcal_target",
        "op": "min",
        "value": 1200,
        "rule_id": "R-SAFE-KCAL-FLOOR-F",
        "reason_code": "KCAL_FLOOR",
    }
    out = _parse(_output("safety_officer", "constraint", [item]))
    assert out.kind == "constraint"
