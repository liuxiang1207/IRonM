"""营养师 · 能量目标（spec §30）。"""

from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ironm.contracts.calories import Activity
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.screening import Screening, Sex, Tier, TierResult
from ironm.engine.calories import calorie_target
from tests.skills.helpers import BOOK, FULL_PROFILE, proposed, run_agent


def _energy(tier: Tier, **kwargs: Any) -> Any:
    return run_agent(AgentId.NUTRITIONIST, "goal.create", tier, **kwargs).get("energy_target")


def test_matches_calorie_engine() -> None:
    out = proposed(_energy(Tier.A, timezone="Asia/Shanghai", **FULL_PROFILE))
    screening = Screening(
        birth_date=date(1990, 6, 15), sex=Sex.FEMALE, height_cm=165.0, weight_kg=72.0
    )
    tier = TierResult(
        tier=Tier.A, status="ok", matched=(), reasons=(), missing=(), bmi=26.4, rule_id="R"
    )
    expected = calorie_target(
        screening,
        Activity.SEDENTARY,
        tier,
        as_of=datetime(2026, 10, 21, tzinfo=UTC),
        tz=ZoneInfo("Asia/Shanghai"),
        rulebook=BOOK,
    )
    assert out["daily_kcal_target"] == expected.target_kcal
    assert out["daily_protein_g_target"] == getattr(expected.macros, "protein_g", None)


def test_uses_smallest_deficit_across_matched_tiers() -> None:
    """主层 B（高血压）、同时满足 A′：缺口取 300 而不是 500。"""
    only_b = proposed(_energy(Tier.B, **FULL_PROFILE))["daily_kcal_target"]
    with_a_prime = proposed(_energy(Tier.B, matched=(Tier.B, Tier.A_PRIME), **FULL_PROFILE))
    assert with_a_prime["daily_kcal_target"] - only_b == 200


def test_collaborative_states_minimum_intake() -> None:
    out = _energy(Tier.COLLABORATIVE, **FULL_PROFILE)
    kcal = next(p for p in out.items if p.field.value == "daily_kcal_target")
    assert any("至少要吃到" in f for f in kcal.reason_facts)


def test_missing_info_no_opinion_with_reason() -> None:
    profile = {k: v for k, v in FULL_PROFILE.items() if k != "sex"}
    out = _energy(Tier.A, **profile)
    assert out.kind is OutputKind.NO_OPINION
    assert out.reason_code == "calories_insufficient:sex"


def test_high_protein_option() -> None:
    normal = proposed(_energy(Tier.A, **FULL_PROFILE))["daily_protein_g_target"]
    high = proposed(_energy(Tier.A, high_protein=True, **FULL_PROFILE))["daily_protein_g_target"]
    assert high > normal
