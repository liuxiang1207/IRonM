"""依据卡、复核与第一版计划（spec [2026-10-20] §31）。"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from ironm.contracts.agents import Constraint, Proposal
from ironm.contracts.common import deep_thaw
from ironm.contracts.plan import PlanField
from ironm.orchestrator.evidence import EvidenceError, resolve_source
from ironm.orchestrator.first_plan import FirstPlan, first_plan
from ironm.orchestrator.review import review_proposal
from ironm.rules.loader import RULES_DIR, load_rules
from ironm.safety.decision import plan_allowed
from ironm.skills.loader import load_skills
from sim.first_plan_demo import PERSONAS, main

BOOK = load_rules(RULES_DIR)
SKILLS = load_skills()
AS_OF = datetime(2026, 10, 21, 1, tzinfo=UTC)
TZ = ZoneInfo("Asia/Shanghai")
GYM, OLDER, PREGNANT, NEW = PERSONAS.values()


def _run(profile: dict[str, Any], **kw: bool) -> FirstPlan:
    return first_plan(profile, as_of=AS_OF, tz=TZ, rulebook=BOOK, skills=SKILLS, **kw)


# ---- 复核 ----


def _proposal(field: str, value: Any) -> Proposal:
    data = {
        "proposal_id": f"p:{field}",
        "field": field,
        "from_value": None,
        "to_value": value,
        "reason_facts": ["测试"],
        "rule_id": "R-TR-START",
        "expected_effect": "测试",
        "rollback_condition": "测试",
        "review_date": "2026-11-04",
        "evidence_ids": [],
    }
    return Proposal.model_validate_json(json.dumps(data))


def _constraint(field: str, op: str, value: Any) -> Constraint:
    data = {"field": field, "op": op, "value": value, "rule_id": "R-X", "reason_code": "t"}
    return Constraint.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    ("field", "value", "constraint", "result"),
    [
        ("intensity_target", {"rpe": [6, 7]}, ("max", {"rpe": 7}), "pass"),
        ("intensity_target", {"rpe": [6, 8]}, ("max", {"rpe": 7}), "reject"),
        ("weekly_cardio_minutes", 150, ("max", 150), "pass"),
        ("weekly_cardio_minutes", 151, ("max", 150), "reject"),
        ("daily_kcal_target", 1200, ("min", 1200), "pass"),
        ("daily_kcal_target", 1199, ("min", 1200), "reject"),
        ("daily_kcal_target", 2000, ("exclude", "deficit"), "reject"),
        ("diet_mode", "precise", ("exclude", "precise"), "reject"),
        ("diet_mode", "habit", ("exclude", "precise"), "pass"),
    ],
)
def test_review(field: str, value: Any, constraint: tuple[str, Any], result: str) -> None:
    op, limit = constraint
    verdict = review_proposal(_proposal(field, value), [_constraint(field, op, limit)])
    assert verdict.result == result
    if result == "reject":
        assert verdict.reason_code == f"violates:R-X:{field}:{op}"


def test_review_ignores_other_fields() -> None:
    verdict = review_proposal(
        _proposal("weekly_cardio_minutes", 500), [_constraint("daily_kcal_target", "min", 1200)]
    )
    assert verdict.result == "pass"


# ---- 依据 ----


def test_sources_resolved_or_rejected() -> None:
    rule = resolve_source("R-TR-START", BOOK)
    assert (rule.kind, rule.review) == ("guideline", "approved") and rule.citation
    assert resolve_source("KB-EXERCISES", BOOK).kind == "knowledge"
    with pytest.raises(EvidenceError, match="不存在"):
        resolve_source("R-NOT-THERE", BOOK)


# ---- 第一版计划 ----


def test_a_tier_full_plan_with_cards() -> None:
    result = _run(GYM)
    assert result.plan is not None
    assert set(result.plan.fields) == {
        "training_days_per_week",
        "session_slots",
        "intensity_target",
        "weekly_cardio_minutes",
        "exercise_selection",
        "minimum_task",
        "alternative_version",
        "daily_kcal_target",
        "daily_protein_g_target",
        "diet_mode",
    }
    # 每条进入计划的提案都有依据卡，依据都有来源与审定状态
    assert {c.field.value for c in result.cards} == set(result.plan.fields)
    for card in result.cards:
        assert card.source.review == "approved" and card.facts and card.data_used
    assert result.plan.skill_versions and result.plan.rule_set_versions
    assert all(v.result == "pass" for v in result.verdicts)


def test_level_two_gym_user_gets_matching_exercises() -> None:
    result = _run(GYM)
    assert result.plan is not None
    chosen: Any = deep_thaw(result.plan.fields[PlanField.EXERCISE_SELECTION])
    assert any(item["exercise_id"] == "sq_goblet" for item in chosen)  # 不是一律最容易的


def test_b_tier_respects_safety_officer() -> None:
    result = _run(OLDER)
    assert result.plan is not None
    fields: Any = deep_thaw(dict(result.plan.fields))
    assert max(fields["intensity_target"]["rpe"]) <= 7
    assert fields["weekly_cardio_minutes"] <= 150
    assert fields["daily_kcal_target"] >= 1200
    assert result.baseline.test_ids == ("knee_push_up@1", "sit_to_stand_30s@1", "resting_hr@1")


@pytest.mark.parametrize("profile", [PREGNANT, NEW])
def test_no_prescription_without_plan_tier(profile: dict[str, Any]) -> None:
    result = _run(profile)
    assert result.plan is None and result.cards == ()
    assert not any(getattr(o, "kind", None) == "proposal" for o in result.outputs)
    assert result.baseline.test_ids == ("resting_hr@1",)


def test_new_user_stops_at_questions() -> None:
    result = _run(NEW)
    assert [q.item for q in result.questions] == [
        "profile.conditions",
        "profile.birth_date",
        "profile.height_cm",
    ]


def test_c_tier_notice() -> None:
    result = _run(PREGNANT)
    tier = result.tier.tier
    assert not plan_allowed(tier)  # type: ignore[arg-type]
    assert result.notices == ("tier_c",)


def test_unwell_today_skips_fitness_tests_but_plan_still_made() -> None:
    result = _run(GYM, unwell_today=True)
    assert result.baseline.test_ids == ("resting_hr@1",)
    assert result.plan is not None


def test_demo_runs(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    out = capsys.readouterr().out
    for name in PERSONAS:
        assert name in out


def test_stops_before_planners_when_blocking_questions() -> None:
    """停在追问时，训练规划师、营养师、动作教练都不运行。"""
    result = _run(NEW)
    assert {o.agent.value for o in result.outputs} <= {"assessor", "safety_officer"}


EXCLUDE_HABIT = """---
id: exclude_habit
owner_agent: safety_officer
version: 0.1.0
description: 测试用：排除习惯模式
triggers:
  intents: [goal.create]
applies_to:
  scene_packs: [fat_loss]
  population_tiers: [a]
requires_data: []
produces: []
priority: 99
prescriptive: false
review: approved
reviewed_by: 测试
tests: tests/test_first_plan.py
---

测试用。
"""

EXCLUDE_HABIT_RUN = """
from ironm.skills.outputs import make_output


def run(view, context):
    item = {
        "field": "diet_mode",
        "op": "exclude",
        "value": "habit",
        "rule_id": "R-NU-DIET-MODE",
        "reason_code": "test",
    }
    return make_output(context, "constraint", items=[item])
"""


def test_rejected_proposal_not_in_plan_and_has_no_card(tmp_path: Path) -> None:
    folder = tmp_path / "safety_officer" / "exclude_habit"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(EXCLUDE_HABIT, encoding="utf-8")
    (folder / "skill.py").write_text(EXCLUDE_HABIT_RUN, encoding="utf-8")
    skills = (*SKILLS, *load_skills(tmp_path))
    result = first_plan(GYM, as_of=AS_OF, tz=TZ, rulebook=BOOK, skills=skills)
    rejected = [v for v in result.verdicts if v.result == "reject"]
    assert [v.proposal_id for v in rejected] == ["diet_mode:diet_mode"]
    assert rejected[0].reason_code == "violates:R-NU-DIET-MODE:diet_mode:exclude"
    assert result.plan is not None and PlanField.DIET_MODE not in result.plan.fields
    assert "diet_mode" not in {c.field.value for c in result.cards}
