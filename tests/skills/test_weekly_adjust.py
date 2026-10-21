"""训练规划师 · 每周训练复盘（spec [2026-10-21] §17）：决策表的每一行。"""

from typing import Any

import pytest

from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Tier
from tests.skills.helpers import proposed, run_agent

SLOTS = [
    {"pattern": p, "sets": 2, "reps": [8, 12], "rpe": [6, 7]}
    for p in ("squat", "hinge", "push", "pull", "core")
]
PLAN = {"weekly_cardio_minutes": 120, "session_slots": SLOTS}
OK = {"completion": 0.75, "prev_completion": 0.8, "avg_rpe": 7.0, "fatigue": 3,
      "fatigue_baseline": 3.0, "sleep_quality": 4}  # fmt: skip


def _adjust(tier: Tier = Tier.A, plan: dict[str, Any] | None = None, **review: Any) -> Any:
    out = run_agent(
        AgentId.TRAINING_PLANNER,
        "review.weekly",
        tier,
        plan_fields=plan or PLAN,
        review={**OK, **review},
    )
    return out.get("weekly_adjust")


def _changes(out: Any) -> dict[str, Any]:
    return proposed(out) if out is not None and out.kind == "proposal" else {}


def _sets(changes: dict[str, Any]) -> list[int]:
    return [s["sets"] for s in changes["session_slots"]]


def test_recovery_row_cuts_cardio_and_one_set() -> None:
    changes = _changes(_adjust(fatigue=5, fatigue_baseline=3.0))
    assert changes["weekly_cardio_minutes"] == 95  # 120 × 0.8 = 96，取 5 的倍数
    assert _sets(changes) == [1] * 5
    assert _changes(_adjust(sleep_quality=2))["weekly_cardio_minutes"] == 95


def test_rpe_high_row_cuts_one_set_only() -> None:
    changes = _changes(_adjust(avg_rpe=9.2))
    assert "weekly_cardio_minutes" not in changes and _sets(changes) == [1] * 5


def test_low_completion_needs_two_weeks() -> None:
    assert _changes(_adjust(completion=0.4, prev_completion=0.8)) == {}
    changes = _changes(_adjust(completion=0.4, prev_completion=0.3))
    assert changes["weekly_cardio_minutes"] == 95 and "session_slots" not in changes


def test_low_completion_not_below_start() -> None:
    plan = {**PLAN, "weekly_cardio_minutes": 100}
    changes = _changes(_adjust(plan=plan, completion=0.4, prev_completion=0.3))
    assert changes["weekly_cardio_minutes"] == 90  # 起点 90，不再往下


def test_progress_adds_cardio_step_then_sets_then_reps() -> None:
    changes = _changes(_adjust(completion=0.95, avg_rpe=6.5))
    assert changes["weekly_cardio_minutes"] == 150 and _sets(changes) == [3] * 5
    three = [{**s, "sets": 3} for s in SLOTS]
    changes = _changes(_adjust(plan={**PLAN, "session_slots": three}, completion=1.0, avg_rpe=6))
    assert all(s["reps"] == [8, 15] for s in changes["session_slots"])


def test_progress_without_rpe_needs_two_good_weeks() -> None:
    assert _changes(_adjust(completion=0.95, avg_rpe=None, prev_completion=0.7)) == {}
    assert _changes(_adjust(completion=0.95, avg_rpe=None, prev_completion=0.92))


def test_conservative_tier_smaller_step_and_cap() -> None:
    changes = _changes(_adjust(Tier.B, completion=0.95, avg_rpe=6))
    assert changes["weekly_cardio_minutes"] == 140  # B 层每档 20


def test_cardio_never_above_safety_cap() -> None:
    cap = {"field": "weekly_cardio_minutes", "op": "max", "value": 130,
           "rule_id": "R-SF-TIER-LIMITS", "reason_code": "tier_b_cardio_cap"}  # fmt: skip
    out = run_agent(
        AgentId.TRAINING_PLANNER, "review.weekly", Tier.B, plan_fields=PLAN,
        review={**OK, "completion": 0.95, "avg_rpe": 6}, constraints=(cap,),
    )["weekly_adjust"]  # fmt: skip
    assert proposed(out)["weekly_cardio_minutes"] == 130


@pytest.mark.parametrize(
    "review",
    [
        {"partial_week": True, "fatigue": 5},  # 只记录了一部分：不调整
        {"completion": None},  # 数据不足：不调整
        {"menstruation": True, "fatigue": 5},  # 经期：不下调
        {"completion": 0.7},  # 其余：不变
    ],
)
def test_no_change_rows(review: dict[str, Any]) -> None:
    assert _changes(_adjust(**review)) == {}


def test_not_loaded_for_first_plan_or_c_tier() -> None:
    assert "weekly_adjust" not in run_agent(AgentId.TRAINING_PLANNER, "goal.create", Tier.A)
    assert "weekly_adjust" not in run_agent(AgentId.TRAINING_PLANNER, "review.weekly", Tier.C)
