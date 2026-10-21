"""训练规划师 · 每周结构（spec §30）。"""

from typing import Any

import pytest

from ironm.contracts.agents import ProposalOutput
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.screening import Tier
from tests.skills.helpers import proposed, run_agent


def _plan(tier: Tier | None, **kwargs: Any) -> Any:
    return run_agent(AgentId.TRAINING_PLANNER, "goal.create", tier, **kwargs).get(
        "weekly_structure"
    )


def _cap(field: str, value: Any) -> dict[str, Any]:
    return {
        "field": field,
        "op": "max",
        "value": value,
        "rule_id": "R-SF-TIER-LIMITS",
        "reason_code": "t",
    }


@pytest.mark.parametrize(("available", "days"), [(3, 2), (4, 2), (5, 3), (None, 2)])
def test_strength_days(available: int | None, days: int) -> None:
    profile = {"available_days": available} if available is not None else {}
    assert proposed(_plan(Tier.A, **profile))["training_days_per_week"] == days


def test_slots_and_intensity() -> None:
    out = proposed(_plan(Tier.A))
    assert [s["pattern"] for s in out["session_slots"]] == [
        "squat",
        "hinge",
        "push",
        "pull",
        "core",
    ]
    assert all(s["sets"] == 2 and s["reps"] == [8, 12] for s in out["session_slots"])
    assert out["intensity_target"] == {"rpe": [6, 7]}


@pytest.mark.parametrize(
    ("tier", "current", "minutes"),
    [(Tier.A, 120, 150), (Tier.A, 30, 90), (Tier.A, None, 90), (Tier.B, 0, 60), (Tier.B, 50, 80)],
)
def test_cardio_start(tier: Tier, current: int | None, minutes: int) -> None:
    profile = {"current_cardio_minutes": current} if current is not None else {}
    assert proposed(_plan(tier, **profile))["weekly_cardio_minutes"] == minutes


def test_respects_safety_constraints() -> None:
    caps = (_cap("intensity_target", {"rpe": 6}), _cap("weekly_cardio_minutes", 150))
    out = proposed(_plan(Tier.B, current_cardio_minutes=200, constraints=caps))
    assert out["intensity_target"] == {"rpe": [6, 6]}
    assert out["weekly_cardio_minutes"] == 150


def test_three_piece_and_review_date() -> None:
    out = _plan(Tier.A)
    assert isinstance(out, ProposalOutput)
    for p in out.items:
        assert p.expected_effect and p.rollback_condition and p.reason_facts
        assert p.review_date.isoformat() == "2026-11-04"  # 2026-10-21 + 14 天
        assert p.from_value is None  # 第一版


def test_unchanged_plan_no_opinion() -> None:
    first = proposed(_plan(Tier.A))
    again = _plan(Tier.A, plan_fields=first)
    assert again.kind is OutputKind.NO_OPINION and again.reason_code == "no_change"


@pytest.mark.parametrize("tier", [Tier.C, Tier.D, Tier.PAUSED, Tier.NOT_APPLICABLE, None])
def test_not_loaded_without_plan_tier(tier: Tier | None) -> None:
    assert _plan(tier) is None
