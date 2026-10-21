"""评估师 · 建档追问（spec §29）。"""

from typing import Any

from ironm.contracts.agents import QuestionsOutput
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.screening import Tier
from tests.skills.helpers import FULL_PROFILE, run_agent


def _ask(tier: Tier | None, **profile: Any) -> Any:
    return run_agent(AgentId.ASSESSOR, "goal.create", tier, **profile)["intake_questions"]


def test_new_user_asked_three_most_important() -> None:
    out = _ask(None)  # 还没建档：分层未知也要加载
    assert isinstance(out, QuestionsOutput)
    assert [q.item for q in out.items] == [
        "profile.conditions",
        "profile.birth_date",
        "profile.height_cm",
    ]
    assert {q.fallback for q in out.items} == {"no_plan_until_answered"}


def test_only_waist_missing() -> None:
    profile = {k: v for k, v in FULL_PROFILE.items() if k != "waist_cm"}
    out = _ask(Tier.A_PRIME, **profile)
    assert isinstance(out, QuestionsOutput)
    [q] = out.items
    assert (q.item, q.fallback) == ("profile.waist_cm", "assume_a_prime")


def test_activity_falls_back_to_sedentary() -> None:
    profile = {k: v for k, v in FULL_PROFILE.items() if k not in ("activity", "sex")}
    out = _ask(Tier.A, **profile)
    assert isinstance(out, QuestionsOutput)
    assert [(q.item, q.fallback) for q in out.items] == [
        ("profile.sex", "no_calorie_target_until_answered"),
        ("profile.activity", "assume_sedentary"),
    ]


def test_complete_profile_no_opinion() -> None:
    out = _ask(Tier.A, **FULL_PROFILE)
    assert out.kind is OutputKind.NO_OPINION and out.reason_code == "info_sufficient"
