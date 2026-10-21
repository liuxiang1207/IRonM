"""冲刺 2 末独立审查后的修复（spec [2026-10-20] §32）。

急症词表的漏判句子见 tests/safety/test_emergency.py。
"""

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import yaml
from pydantic import ValidationError

from ironm.contracts.agents import Constraint, Proposal
from ironm.contracts.rules import RuleParamError
from ironm.contracts.screening import SafetyFlag, Tier
from ironm.engine.rule_params import validate_rule_params
from ironm.knowledge.baseline import BASELINE_PATH, BaselineLoadError, load_baseline
from ironm.orchestrator.first_plan import first_plan
from ironm.orchestrator.review import review_proposal
from ironm.rules.loader import RULES_DIR, load_rules
from ironm.skills.loader import load_skills
from sim.first_plan_demo import PERSONAS

BOOK = load_rules(RULES_DIR)
SKILLS = load_skills()
AS_OF = datetime(2026, 10, 21, 1, tzinfo=UTC)
TZ = ZoneInfo("Asia/Shanghai")
GYM, OLDER, PREGNANT, NEW = PERSONAS.values()


def _plan(profile: dict[str, Any], **kw: Any) -> Any:
    skills = kw.pop("skills", SKILLS)
    return first_plan(profile, as_of=AS_OF, tz=TZ, rulebook=BOOK, skills=skills, **kw)


# ---- 复核：无法比较一律拒绝（H3） ----


def _p(field: str, value: Any) -> Proposal:
    data = {
        "proposal_id": "p",
        "field": field,
        "from_value": None,
        "to_value": value,
        "reason_facts": ["t"],
        "rule_id": "R",
        "expected_effect": "t",
        "rollback_condition": "t",
        "review_date": "2026-11-04",
        "evidence_ids": [],
    }
    return Proposal.model_validate_json(json.dumps(data))


def _c(field: str, op: str, value: Any) -> Constraint:
    data = {"field": field, "op": op, "value": value, "rule_id": "R-X", "reason_code": "t"}
    return Constraint.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    ("field", "value", "limit"),
    [
        ("intensity_target", {"rpe_max": 10}, {"rpe": 7}),
        ("intensity_target", 10, {"rpe": 7}),
        ("intensity_target", {"rpe": "10"}, {"rpe": 7}),
        ("weekly_cardio_minutes", "400", 150),
        ("weekly_cardio_minutes", {"minutes": 400}, 150),
        ("weekly_cardio_minutes", [], 150),
        ("weekly_cardio_minutes", True, 150),
    ],
)
def test_uncomparable_rejected(field: str, value: Any, limit: Any) -> None:
    verdict = review_proposal(_p(field, value), [_c(field, "max", limit)])
    assert verdict.result == "reject"
    assert verdict.reason_code == f"uncomparable:R-X:{field}:max"


@pytest.mark.parametrize("limit", ["150", [60, 150], {}, {"rpe": "7"}, True])
def test_constraint_limit_shape_checked(limit: Any) -> None:
    with pytest.raises(ValidationError, match="上限与下限"):
        _c("weekly_cardio_minutes", "max", limit)


# ---- 处方闸门（H4） ----

SNEAKY = """---
id: sneaky
owner_agent: nutritionist
version: 0.1.0
description: 测试用：自称非处方却产出热量提案
triggers:
  intents: [goal.create]
applies_to:
  scene_packs: [fat_loss]
  population_tiers: [a, c]
requires_data: []
produces: []
priority: 1
prescriptive: false
review: approved
reviewed_by: 测试
tests: tests/test_sprint2_review_fixes.py
---

测试用。
"""

SNEAKY_RUN = """
from ironm.skills.outputs import make_output


def run(view, context):
    item = {
        "proposal_id": "x", "field": "daily_kcal_target", "from_value": None, "to_value": 1300,
        "reason_facts": ["x"], "rule_id": "R-CAL-DEFICIT", "expected_effect": "x",
        "rollback_condition": "x", "review_date": "2026-11-04", "evidence_ids": [],
    }
    return make_output(context, "proposal", items=[item])
"""


def _sneaky(tmp_path: Path) -> tuple[Any, ...]:
    folder = tmp_path / "nutritionist" / "sneaky"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(SNEAKY, encoding="utf-8")
    (folder / "skill.py").write_text(SNEAKY_RUN, encoding="utf-8")
    return (*SKILLS, *load_skills(tmp_path))


def test_c_tier_never_gets_numbers_even_from_mislabeled_skill(tmp_path: Path) -> None:
    result = _plan(PREGNANT, skills=_sneaky(tmp_path))
    assert result.plan is None and result.cards == ()
    assert {o.agent.value for o in result.outputs} <= {"assessor", "safety_officer"}


def test_mislabeled_skill_output_dropped_at_runtime(tmp_path: Path) -> None:
    """A 层也会运行营养师：非处方类产出提案 → 该 Skill 降级为无意见，不进计划。"""
    result = _plan(GYM, skills=_sneaky(tmp_path))
    sneaky = [o for o in result.outputs if o.skill == "sneaky@0.1.0"]
    assert [o.reason_code for o in sneaky] == ["skill_error:AgentContractError"]
    assert all(c.card_id != "x" for c in result.cards)


# ---- 画像校验、红旗、推出的状况（M2、M5） ----


@pytest.mark.parametrize(
    ("field", "value"),
    [("height_cm", "165"), ("birth_date", "1990/06/15"), ("conditions", None), ("weight_kg", 10)],
)
def test_bad_screening_field_treated_as_unknown(field: str, value: Any) -> None:
    profile = {**GYM, field: value}
    result = _plan(profile)
    assert result.plan is None
    assert f"profile.{field}" in {q.item for q in result.questions}


def test_profile_must_be_json() -> None:
    with pytest.raises(ValueError, match="JSON"):
        _plan({**GYM, "note": datetime(2026, 1, 1)})


def test_uncleared_red_flag_pauses() -> None:
    flag = SafetyFlag(at=AS_OF - timedelta(hours=1), level="red_flag", rule_ids=("RF-CHEST",))
    result = _plan(GYM, flags=[flag])
    assert result.tier.tier is Tier.PAUSED and result.plan is None
    cleared = _plan(GYM, flags=[flag], clearances=[AS_OF - timedelta(minutes=10)])
    assert cleared.plan is not None


def test_chat_derived_eating_signal_reaches_safety_officer() -> None:
    flag = SafetyFlag(
        at=AS_OF - timedelta(days=2),
        level="red_flag",
        rule_ids=("RF-PURGE",),
        categories=("eating_disorder",),
    )
    result = _plan(GYM, flags=[flag], clearances=[AS_OF - timedelta(days=1)])
    assert result.tier.tier is Tier.C
    fields = {(c.field.value, c.op) for c in result.constraints}
    assert ("weigh_in_days", "max") in fields


# ---- 时区（M3） ----


def test_age_uses_one_timezone() -> None:
    """上海时间生日当天满 65 岁：分层与热量用同一个时区（BMI < 24 → 老年维持，缺口 0）。"""
    profile = {
        **OLDER,
        "birth_date": "1961-10-21",
        "conditions": [],
        "weight_kg": 58.0,
        "height_cm": 158.0,
    }
    result = _plan(profile)
    assert result.tier.tier is Tier.B and "age_senior" in result.tier.reasons
    kcal = next(c for c in result.cards if c.field.value == "daily_kcal_target")
    assert any("只给维持热量" in f for f in kcal.facts)


# ---- 有氧起点（M4） ----


def test_b_tier_cardio_start_uses_tier_step() -> None:
    result = _plan({**OLDER, "current_cardio_minutes": 100})
    assert result.plan is not None
    assert result.plan.fields["weekly_cardio_minutes"] == 120  # B 层 +20，不是 +30


# ---- 跨规则一致（L3） ----


def _rules(tmp_path: Path, rule_set: str, rule_id: str, **params: Any) -> Any:
    rules = tmp_path / "rules"
    shutil.copytree(RULES_DIR, rules)
    path = rules / f"{rule_set}.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    [rule] = [r for r in doc["rules"] if r["id"] == rule_id]
    rule["params"].update(params)
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return load_rules(rules)


@pytest.mark.parametrize(
    ("rule_set", "rule_id", "params", "message"),
    [
        ("calories", "R-CAL-DEFICIT", {"senior_age": 60}, "老年年龄界值不一致"),
        ("metrics", "R-MT-JUDGEMENT-WINDOW", {"min_window_days": 5}, "短于"),
    ],
)
def test_cross_rule_mismatch_fails(
    tmp_path: Path, rule_set: str, rule_id: str, params: dict[str, Any], message: str
) -> None:
    with pytest.raises(RuleParamError, match=message):
        validate_rule_params(_rules(tmp_path, rule_set, rule_id, **params))


def test_fitness_test_cannot_apply_to_c_tier(tmp_path: Path) -> None:
    data = yaml.safe_load(BASELINE_PATH.read_text(encoding="utf-8"))
    data["tests"][0]["tiers"].append("c")
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(BaselineLoadError, match="不可给计划"):
        load_baseline(path)
