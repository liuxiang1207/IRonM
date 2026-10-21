"""发布前的三道关（spec [2026-10-21] §17）：整合、盲审、终审。"""

import json
from typing import Any

from ironm.contracts.agents import Constraint, Proposal
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Tier
from ironm.orchestrator.gates import (
    Candidate,
    blind,
    blind_review,
    final_gates,
    integrate,
)
from ironm.rules.loader import RULES_DIR, load_rules

BOOK = load_rules(RULES_DIR)
SLOTS = [{"pattern": "squat", "sets": 2, "reps": [8, 12], "rpe": [6, 7]}]


def _proposal(field: str, before: Any, after: Any, rule: str = "R-RV-WEEKLY") -> Proposal:
    data = {
        "proposal_id": f"x:{field}", "field": field, "from_value": before, "to_value": after,
        "reason_facts": ["测试"], "rule_id": rule, "expected_effect": "两周内能完成",
        "rollback_condition": "完成率低于一半时回退", "review_date": "2026-11-18",
        "evidence_ids": [],
    }  # fmt: skip
    return Proposal.model_validate_json(json.dumps(data))


def _c(
    field: str,
    before: Any,
    after: Any,
    *,
    agent: AgentId = AgentId.TRAINING_PLANNER,
    source: str = "weekly_review",
    category: str = "progress",
    rule: str = "R-RV-WEEKLY",
) -> Candidate:
    return Candidate(_proposal(field, before, after, rule), agent, "weekly_adjust@1.0.0",
                     source, category)  # type: ignore[arg-type]  # fmt: skip


def _constraint(field: str, op: str, value: Any, rule: str = "R-SF-TIER-LIMITS") -> Constraint:
    data = {"field": field, "op": op, "value": value, "rule_id": rule, "reason_code": "t"}
    return Constraint.model_validate_json(json.dumps(data))


# ---- 整合 ----


def test_at_most_two_changes_one_per_domain_by_priority() -> None:
    progress = _c("weekly_cardio_minutes", 90, 120)
    rollback = _c("session_slots", [*SLOTS], [{**SLOTS[0], "sets": 1}], source="rollback",
                  category="recovery")  # fmt: skip
    diet = _c("daily_kcal_target", 2000, 1850, agent=AgentId.NUTRITIONIST, source="diet_review",
              rule="R-RV-DIET")  # fmt: skip
    out = integrate([progress, rollback, diet], constraints=(), profile={})
    kept = {c.proposal.field.value for c in out.accepted}
    assert kept == {"session_slots", "daily_kcal_target"}  # 恢复优先于进阶；训练与饮食各一项
    assert [(c.proposal.field.value, why) for c, why in out.rejected] == [
        ("weekly_cardio_minutes", "over_limit")
    ]


def test_three_domains_capped_at_two_by_priority() -> None:
    """训练、其他（习惯）、饮食各想改一项：总共最多 2 项。
    按优先级留下恢复与现实约束两项，进阶的饮食退回。"""
    training = _c("session_slots", [*SLOTS], [{**SLOTS[0], "sets": 1}], category="recovery")
    habit = _c("weekly_habit_step", "每天走 6000 步", "每天走 7000 步",
               agent=AgentId.BEHAVIOR_COACH, source="habit_review", category="reality")  # fmt: skip
    diet = _c("daily_kcal_target", 2000, 1850, agent=AgentId.NUTRITIONIST, source="diet_review",
              rule="R-RV-DIET")  # fmt: skip
    out = integrate([diet, habit, training], constraints=(), profile={})
    assert {c.proposal.field.value for c in out.accepted} == {"session_slots", "weekly_habit_step"}
    assert [(c.proposal.field.value, why) for c, why in out.rejected] == [
        ("daily_kcal_target", "over_limit")
    ]


def test_same_source_changes_count_as_one_item() -> None:
    cardio = _c("weekly_cardio_minutes", 90, 72, category="recovery")
    slots = _c("session_slots", [*SLOTS], [{**SLOTS[0], "sets": 1}], category="recovery")
    out = integrate([cardio, slots], constraints=(), profile={})
    assert len(out.accepted) == 2 and out.rejected == ()


def test_overreach_constraint_and_inconsistency_rejected() -> None:
    overreach = _c("daily_kcal_target", 2000, 1800, agent=AgentId.TRAINING_PLANNER)
    over_cap = _c("weekly_cardio_minutes", 90, 200)
    too_many_days = _c("training_days_per_week", 3, 6)
    out = integrate(
        [overreach, over_cap, too_many_days],
        constraints=(_constraint("weekly_cardio_minutes", "max", 150),),
        profile={"available_days": 4},
    )
    reasons = {c.proposal.field.value: why for c, why in out.rejected}
    assert reasons["daily_kcal_target"] == "overreach"
    assert reasons["weekly_cardio_minutes"].startswith("violates:")
    assert reasons["training_days_per_week"] == "inconsistent"
    assert out.accepted == ()


# ---- 盲审：植入的错误都被拦下 ----


def _lanes(
    cands: list[Candidate], constraints: tuple[Constraint, ...] = (), tier: Any = Tier.A
) -> Any:
    blinds = blind(cands)
    return blinds, blind_review(blinds, tier=tier, constraints=constraints, rulebook=BOOK)


def test_reviewers_see_no_author_or_user() -> None:
    [b] = blind([_c("weekly_cardio_minutes", 90, 120)])
    seen = json.dumps(b.__dict__, default=str, ensure_ascii=False)
    assert "training_planner" not in seen and "weekly_adjust" not in seen and "u1" not in seen


def test_planted_errors_are_caught() -> None:
    cases = [
        (_c("daily_kcal_target", 1600, 1400, agent=AgentId.NUTRITIONIST, rule="R-RV-DIET"),
         (_constraint("daily_kcal_target", "min", 1500, "R-CAL-FLOOR"),), "safety"),
        (_c("weekly_cardio_minutes", 120, 180), (_constraint("weekly_cardio_minutes", "max", 150),),
         "safety"),
        (_c("session_slots", [*SLOTS], [{**SLOTS[0], "sets": 10}]), (), "quality"),
        (_c("weekly_cardio_minutes", 90, 120, rule="R-NOT-THERE"), (), "evidence"),
        (_c("daily_kcal_target", 2000, 1850, agent=AgentId.NUTRITIONIST, rule="R-RV-DIET"),
         (_constraint("daily_kcal_target", "exclude", "deficit", "R-SF-EATING"),), "safety"),
    ]  # fmt: skip
    for cand, constraints, lane in cases:
        blinds, lanes = _lanes([cand], constraints)
        reason = lanes.blocked(blinds[0].key)
        assert reason is not None and reason.startswith(lane), (cand.proposal.field, reason)


def test_no_plan_tier_vetoed_by_safety_lane() -> None:
    blinds, lanes = _lanes([_c("weekly_cardio_minutes", 90, 120)], tier=Tier.C)
    assert lanes.blocked(blinds[0].key) == "safety:reject:tier_no_plan"


def test_design_rules_are_warnings_not_blocks() -> None:
    blinds, lanes = _lanes([_c("weekly_cardio_minutes", 90, 120)])  # R-RV-WEEKLY 来源为设计
    assert lanes.evidence[blinds[0].key] == "engineering" and lanes.blocked(blinds[0].key) is None


# ---- 终审 ----


def _gates(accepted: list[Candidate], lanes_for: list[Candidate], **kw: Any) -> Any:
    blinds, lanes = _lanes(lanes_for)
    keys = {b.key: c for b, c in zip(blinds, lanes_for, strict=True)}
    plan = kw.pop("plan", {"rule_set_versions": {"plan": "1"}, "skill_versions": {"x": "1"}})
    return final_gates(
        plan=plan,
        accepted=accepted,
        cards=kw.pop("cards", []),
        constraints=(),
        lanes=lanes,
        keys=keys,
        data_fresh=kw.pop("fresh", True),
    )


def test_gates_block_missing_cards_stale_data_and_versions() -> None:
    good = _c("weekly_cardio_minutes", 90, 120)
    missing_card = _gates([good], [good])
    assert "trio:weekly_cardio_minutes" in missing_card.blocking and not missing_card.publish
    assert "data_stale" in _gates([], [], fresh=False).blocking
    assert (
        "version" in _gates([], [], plan={"rule_set_versions": {}, "skill_versions": {}}).blocking
    )
    assert "format" in _gates([good], [good], plan=None).blocking


def test_partial_publish_keeps_good_change() -> None:
    """一项被审查拦下、一项通过：只有通过的那项进入发布，终审不因被拦下的那项卡住。"""
    good = _c("weekly_cardio_minutes", 90, 120)
    bad = _c("session_slots", [*SLOTS], [{**SLOTS[0], "sets": 10}])
    blinds, lanes = _lanes([good, bad])
    keys = {b.key: c for b, c in zip(blinds, [good, bad], strict=True)}
    passed = [c for k, c in keys.items() if lanes.blocked(k) is None]
    assert passed == [good]
    review_blocks = [b for b in _gates(passed, [good, bad]).blocking if b.startswith("review:")]
    assert review_blocks == []
