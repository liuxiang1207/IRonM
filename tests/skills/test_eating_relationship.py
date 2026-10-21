"""安全官 · 饮食关系警示（spec §29）。"""

import pytest

from ironm.contracts.agents import ConstraintOutput
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.screening import Tier
from tests.skills.helpers import run_agent


@pytest.mark.parametrize("condition", ["eating_disorder_history", "eating_disorder_signal"])
@pytest.mark.parametrize("tier", [Tier.A, Tier.C])
def test_signal_limits_deficit_and_weighing(condition: str, tier: Tier) -> None:
    out = run_agent(AgentId.SAFETY_OFFICER, "review.weekly", tier, conditions=[condition])[
        "eating_relationship"
    ]
    assert isinstance(out, ConstraintOutput)
    assert {(c.field.value, c.op, c.value) for c in out.items} == {
        ("daily_kcal_target", "exclude", "deficit"),
        ("weigh_in_days", "max", 1),
    }


def test_no_signal_no_opinion() -> None:
    out = run_agent(AgentId.SAFETY_OFFICER, "review.weekly", Tier.A, conditions=["hypertension"])[
        "eating_relationship"
    ]
    assert out.kind is OutputKind.NO_OPINION
