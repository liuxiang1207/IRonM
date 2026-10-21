"""营养师 · 饮食模式（spec §30）。"""

import pytest

from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Tier
from tests.skills.helpers import proposed, run_agent


def _mode(**kwargs: object) -> object:
    out = run_agent(AgentId.NUTRITIONIST, "goal.create", Tier.A, profile=kwargs)["diet_mode"]
    return proposed(out)["diet_mode"]


@pytest.mark.parametrize(
    ("precision", "mode"), [("P3", "precise"), ("P2", "simple"), ("P1", "habit"), ("P0", "habit")]
)
def test_by_precision(precision: str, mode: str) -> None:
    assert _mode(diet_precision=precision) == mode


def test_user_preference_wins() -> None:
    assert _mode(diet_precision="P0", diet_mode_preference="precise") == "precise"
    assert _mode(diet_precision="P3", diet_mode_preference="unknown_mode") == "precise"


def test_not_loaded_for_c_tier() -> None:
    assert "diet_mode" not in run_agent(AgentId.NUTRITIONIST, "goal.create", Tier.C)
