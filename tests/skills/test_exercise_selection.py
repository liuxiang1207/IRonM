"""动作教练 · 选动作（spec §30）。"""

from typing import Any

from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.screening import Tier
from ironm.knowledge.exercises import default_library
from tests.skills.helpers import proposed, run_agent

LIB = default_library()
SLOTS = [{"pattern": p} for p in ("squat", "hinge", "lunge", "push", "pull", "core")]


def _coach(**kwargs: Any) -> Any:
    out = run_agent(AgentId.EXERCISE_COACH, "goal.create", Tier.A, **kwargs)
    return out.get("exercise_selection")


def _picked(**kwargs: Any) -> dict[str, Any]:
    return proposed(_coach(plan_fields={"session_slots": SLOTS}, **kwargs))


def test_hotel_uses_no_equipment_beyond_chair() -> None:
    out = _picked(environment="hotel", training_level=3)
    hotel = LIB.environment("hotel")
    for item in out["exercise_selection"]:
        assert set(LIB.get(item["exercise_id"]).equipment) <= hotel
        assert 1 <= len(item["substitutes"]) <= 2 and item["key_points"]


def test_two_substitutes_at_home_and_gym() -> None:
    for env in ("home", "gym"):
        out = _picked(environment=env, training_level=3)
        assert all(len(i["substitutes"]) == 2 for i in out["exercise_selection"]), env


def test_unknown_environment_treated_as_hotel() -> None:
    assert _picked() == _picked(environment="hotel")


def test_knee_pain_avoided_and_cautions_shown() -> None:
    out = _picked(environment="gym", training_level=3, restrictions=["knee_pain"])
    for item in out["exercise_selection"]:
        exercise = LIB.get(item["exercise_id"])
        assert "knee_pain" not in exercise.avoid_if
        for sub in item["substitutes"]:
            assert "knee_pain" not in LIB.get(sub).avoid_if
    lunge = next(i for i in out["exercise_selection"] if i["slot"] == "lunge")
    assert lunge["cautions"]  # 单腿动作对膝盖疼都标为谨慎


def test_level_limits_difficulty() -> None:
    """水平 1：选中的动作与替代动作都只能是难度 1；水平 3 时替代里会出现更难的。"""
    out = _picked(environment="gym", training_level=1)
    for item in out["exercise_selection"]:
        ids = [item["exercise_id"], *item["substitutes"]]
        assert all(LIB.get(i).difficulty == 1 for i in ids), item["slot"]
    advanced = _picked(environment="gym", training_level=3)
    assert any(
        LIB.get(s).difficulty > 1 for i in advanced["exercise_selection"] for s in i["substitutes"]
    )


def test_minimum_task_and_hotel_version() -> None:
    out = _picked(environment="gym", training_level=2)
    assert out["minimum_task"] == [
        {"exercise_id": "sq_sit_to_stand", "sets": 2},
        {"exercise_id": "ps_wall_push", "sets": 2},
        {"exercise_id": "hg_glute_bridge", "sets": 2},
    ]
    hotel = LIB.environment("hotel")
    assert [h["slot"] for h in out["alternative_version"]["hotel"]] == [s["pattern"] for s in SLOTS]
    for h in out["alternative_version"]["hotel"]:
        assert set(LIB.get(h["exercise_id"]).equipment) <= hotel


def test_no_slots_no_opinion() -> None:
    out = _coach()
    assert out.kind is OutputKind.NO_OPINION and out.reason_code == "no_slots"
