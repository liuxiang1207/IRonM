"""安全官 · 分层限制（spec §29）。"""

from typing import Any

from ironm.contracts.agents import ConstraintOutput
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Tier
from tests.skills.helpers import run_agent


def _limits(tier: Tier | None) -> dict[str, Any]:
    outputs = run_agent(AgentId.SAFETY_OFFICER, "goal.create", tier, sex="female", conditions=[])
    out = outputs.get("tier_limits")
    if not isinstance(out, ConstraintOutput):
        return {}
    return {c.field.value: (c.op, c.value) for c in out.items}


def test_a_tier() -> None:
    assert _limits(Tier.A) == {"intensity_target": ("max", {"rpe": 8})}


def test_b_and_collaborative_are_stricter() -> None:
    expected = {
        "intensity_target": ("max", {"rpe": 7}),
        "weekly_cardio_minutes": ("max", 150),
    }
    assert _limits(Tier.B) == expected
    assert _limits(Tier.COLLABORATIVE) == expected


def test_not_loaded_for_c_or_unknown() -> None:
    for tier in (Tier.C, None):
        outputs = run_agent(AgentId.SAFETY_OFFICER, "goal.create", tier, conditions=[])
        assert "tier_limits" not in outputs
