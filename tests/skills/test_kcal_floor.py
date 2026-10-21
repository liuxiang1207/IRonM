"""安全官 · 热量底线（spec §29）。"""

import pytest

from ironm.contracts.agents import ConstraintOutput
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Tier
from tests.skills.helpers import run_agent


@pytest.mark.parametrize(("sex", "floor"), [("female", 1200), ("male", 1500), (None, 1500)])
def test_floor_by_sex(sex: str | None, floor: int) -> None:
    profile = {"sex": sex} if sex else {}
    out = run_agent(AgentId.SAFETY_OFFICER, "goal.create", Tier.A, profile=profile)["kcal_floor"]
    assert isinstance(out, ConstraintOutput)
    [c] = out.items
    assert (c.field.value, c.op, c.value, c.rule_id) == (
        "daily_kcal_target",
        "min",
        floor,
        "R-CAL-FLOOR",
    )
