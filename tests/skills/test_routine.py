"""行为教练 · 作息（spec [2026-10-21] §15）：称重日、签到日与提醒时间。"""

from typing import Any

import pytest

from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Tier
from tests.skills.helpers import proposed, run_agent


def _routine(tier: Tier = Tier.A, **kwargs: Any) -> dict[str, Any]:
    return proposed(run_agent(AgentId.BEHAVIOR_COACH, "goal.create", tier, **kwargs)["routine"])


def test_default_weigh_twice_a_week_and_sunday_checkin() -> None:
    out = _routine()
    assert out["weigh_in_days"] == [0, 3]  # 周一、周四
    assert out["checkin_day"] == 6
    assert out["reminder_schedule"]["max_per_day"] == 1


def test_safety_cap_on_weigh_ins_respected() -> None:
    cap = {"field": "weigh_in_days", "op": "max", "value": 1, "rule_id": "R-SF-EATING",
           "reason_code": "eating_relationship_weigh_weekly_at_most"}  # fmt: skip
    assert _routine(constraints=(cap,))["weigh_in_days"] == [0]


@pytest.mark.parametrize("tier", [Tier.C, Tier.PAUSED])
def test_not_loaded_when_no_plan_allowed(tier: Tier) -> None:
    assert "routine" not in run_agent(AgentId.BEHAVIOR_COACH, "goal.create", tier)
