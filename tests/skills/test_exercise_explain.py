"""动作教练 · 动作讲解（问答链路）：内容来自动作库、依据可追溯、找不到时如实说明。"""

from typing import Any

from ironm.contracts.exercises import Restriction
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.screening import Tier
from ironm.contracts.skills import SkillMode
from ironm.knowledge.exercises import default_library
from tests.skills.helpers import run_agent

LIB = default_library()


def _explain(text: str, **kwargs: Any) -> Any:
    return run_agent(
        AgentId.EXERCISE_COACH,
        "exercise_technique",
        Tier.A,
        mode=SkillMode.ANSWER,
        text=text,
        **kwargs,
    )["exercise_explain"]


def _opinion(out: Any) -> dict[str, Any]:
    assert out.kind is OutputKind.OPINION
    item = out.items[0]
    data: dict[str, Any] = item.model_dump()
    return data


def test_explains_matched_exercise_from_library() -> None:
    got = _opinion(_explain("高脚杯深蹲怎么做"))
    ex = next(e for e in LIB.exercises if e.name == "高脚杯深蹲")
    assert ex.name in got["conclusion"]
    knowledge = [c for c in got["citations"] if c["kind"] == "knowledge"]
    assert knowledge and knowledge[0]["ref"] == f"KB-EXERCISES:{ex.id}"
    assert ex.key_points[0] in got["how"]
    assert ex.common_errors[0] in got["caution"]


def test_unknown_exercise_says_so_instead_of_inventing() -> None:
    out = _explain("那个转体抛接动作怎么做")
    assert out.kind is OutputKind.NO_OPINION
    assert out.reason_code == "unknown_exercise"


def test_context_item_pins_the_exercise_without_guessing() -> None:
    got = _opinion(_explain("这个怎么做", context_item="exercise:sq_goblet"))
    assert "高脚杯深蹲" in got["conclusion"]


def test_user_restriction_adds_library_caution() -> None:
    got = _opinion(_explain("高脚杯深蹲怎么做", restrictions=["knee_pain"]))
    cautions = [c for c in got["citations"] if c["kind"] == "rule"]
    assert cautions and cautions[0]["ref"].endswith("knee_pain")
    assert LIB.restrictions[Restriction.KNEE_PAIN].caution in got["caution"]


def test_cannot_do_offers_regression() -> None:
    got = _opinion(_explain("高脚杯深蹲我做不了", restrictions=[]))
    assert "降阶" in got["how"] or "换成" in got["how"]


def test_every_number_in_opinion_is_cited() -> None:
    # 意见层的数字校验在契约里：能构造出意见就说明结论里的数字都有依据
    for text in ("高脚杯深蹲怎么做", "平板支撑怎么做", "俯卧撑我做不了"):
        _opinion(_explain(text))
