"""营养师 · 两周饮食复盘（spec [2026-10-21] §17）：决策表的每一行。"""

from typing import Any

import pytest

from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Tier
from ironm.contracts.skills import ScenePack
from tests.skills.helpers import proposed, run_agent

PLAN = {"daily_kcal_target": 2000}
GOOD = {"red_flag": False, "meal_days_min": 5, "weigh_ins_min": 2, "loss_kg_per_week": 0.1,
        "ci_half_width": 0.2, "completion": 0.8, "adherence_days": 5, "prev_slow": True,
        "days_since_start": 28}  # fmt: skip
FLOOR = {"field": "daily_kcal_target", "op": "min", "value": 1500, "rule_id": "R-CAL-FLOOR",
         "reason_code": "kcal_floor"}  # fmt: skip


def _kcal(
    tier: Tier = Tier.A, constraints: tuple[dict[str, Any], ...] = (FLOOR,), **review: Any
) -> Any:
    out = run_agent(
        AgentId.NUTRITIONIST, "review.biweekly", tier, plan_fields=PLAN,
        review={**GOOD, **review}, constraints=constraints, scene=ScenePack.FAT_LOSS,
    ).get("diet_adjust")  # fmt: skip
    if out is None or out.kind != "proposal":
        return None
    return proposed(out)["daily_kcal_target"]


def test_slow_loss_two_periods_cuts_150() -> None:
    assert _kcal() == 1850


def test_trend_ci_up_to_point_four_is_enough() -> None:
    """可信范围半宽 0.35 千克 / 周：按修正后的门槛（0.4）可以判断，减重偏慢即下调。"""
    assert _kcal(ci_half_width=0.35) == 1850
    assert _kcal(ci_half_width=0.41) is None


def test_slow_loss_first_period_waits() -> None:
    assert _kcal(prev_slow=False) is None


def test_floor_never_broken() -> None:
    floor = {**FLOOR, "value": 1900}
    assert _kcal(constraints=(floor,)) is None


def test_fast_loss_after_four_weeks_adds_150() -> None:
    assert _kcal(loss_kg_per_week=1.2, prev_slow=False) == 2150
    assert (
        _kcal(loss_kg_per_week=1.2, prev_slow=False, days_since_start=20) is None
    )  # 前 4 周不判断


@pytest.mark.parametrize(
    "review",
    [
        {"red_flag": True},
        {"meal_days_min": 3},
        {"weigh_ins_min": 0},
        {"ci_half_width": 0.45},  # 可信范围宽于每周 0.4 千克：数据不足
        {"completion": 0.4},
        {"adherence_days": 2},
        {"partial_week": True},
        {"menstruation": True},
        {"loss_kg_per_week": 0.4},  # 在目标附近：不变
    ],
)
def test_no_change_rows(review: dict[str, Any]) -> None:
    assert _kcal(**review) is None


def test_a_prime_target_lower() -> None:
    # 每周减 0.15 千克：A 层（下限 0.5 的一半是 0.25）算偏慢；A′ 层（下限 0.25 的一半是 0.125）不算
    assert _kcal(Tier.A, loss_kg_per_week=0.15) == 1850
    assert _kcal(Tier.A_PRIME, loss_kg_per_week=0.15) is None
    assert _kcal(Tier.A_PRIME, loss_kg_per_week=0.05) == 1850


def test_only_for_fat_loss_and_a_tiers() -> None:
    for scene in (ScenePack.FITNESS, ScenePack.STRENGTH):
        out = run_agent(AgentId.NUTRITIONIST, "review.biweekly", Tier.A, plan_fields=PLAN,
                        review=GOOD, scene=scene)  # fmt: skip
        assert "diet_adjust" not in out
    assert _kcal(Tier.B) is None
