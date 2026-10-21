"""选动作（spec §30）：槽位落成具体动作，附要点与替代；最低任务与酒店版本。"""

from collections.abc import Iterable
from datetime import timedelta
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from ironm.contracts.agents import AgentOutput
from ironm.contracts.exercises import Equipment, Pattern, Restriction
from ironm.knowledge.exercises import Library, default_library
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import Change, current_value, no_opinion, proposals
from ironm.skills.view import SnapshotView

RULE = "R-EX-MINIMUM"
LIBRARY_REF = "KB-EXERCISES"  # 动作库（knowledge/exercises.yaml）
ENVIRONMENTS = ("hotel", "home", "gym")


class MinimumParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    patterns: tuple[str, ...]
    sets: int

    @model_validator(mode="after")
    def _check(self) -> Self:
        unknown = [s for s in self.patterns if s not in set(Pattern)]
        if unknown:
            raise ValueError(f"不认识的动作模式：{unknown}")
        if not 2 <= len(self.patterns) <= 4 or not 1 <= self.sets <= 3:
            raise ValueError("最低任务应为 2–4 个动作、每个 1–3 组")
        return self


RULE_PARAMS = {RULE: MinimumParams}


def _pick(
    lib: Library,
    pattern: Pattern,
    equipment: frozenset[Equipment],
    restrictions: tuple[Restriction, ...],
    level: int,
) -> dict[str, Any]:
    found = lib.candidates(
        pattern, equipment=equipment, restrictions=restrictions, max_difficulty=level
    )
    if found:
        # 取不超过用户水平、最接近其水平的动作（同难度按编号）；不是一律取最容易的
        top = max(c.difficulty for c in found)
        choice = next(c for c in found if c.difficulty == top)
    else:
        default = lib.default_for(pattern).id
        choice = lib.regress(default, equipment=equipment, restrictions=restrictions)
    exercise = lib.get(choice.exercise_id)
    subs = sorted(
        lib.substitutes(
            choice.exercise_id, equipment=equipment, restrictions=restrictions, max_difficulty=level
        ),
        key=lambda s: (abs(s.difficulty - choice.difficulty), s.exercise_id),  # 难度相近的优先
    )
    return {
        "slot": pattern.value,
        "exercise_id": exercise.id,
        "name": exercise.name,
        "key_points": list(exercise.key_points),
        "cautions": list(choice.cautions),
        "substitutes": [s.exercise_id for s in subs[:2]],
    }


def _restrictions(value: object) -> tuple[Restriction, ...]:
    items = value if isinstance(value, list | tuple) else ()
    return tuple(Restriction(r) for r in items if r in set(Restriction))


def _patterns(slots: Iterable[Any]) -> list[Pattern]:
    found: list[Pattern] = []
    for slot in slots:
        name = slot.get("pattern") if isinstance(slot, dict) else None
        if name in set(Pattern):
            found.append(Pattern(name))
    return found


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    slots = current_value(view, "session_slots")
    patterns = _patterns(slots) if isinstance(slots, list) else []
    if not patterns:
        return no_opinion(context, "no_slots")
    lib = default_library()
    p = context.rulebook.get(RULE).params_as(MinimumParams)
    env = view.get("profile.environment")
    environment = env if env in ENVIRONMENTS else "hotel"  # 不知道时按最少器材
    level = view.get("profile.training_level")
    max_difficulty = level if isinstance(level, int) and 1 <= level <= 3 else 1
    restrictions = _restrictions(view.get("profile.restrictions"))
    equipment = lib.environment(environment)  # type: ignore[arg-type]
    hotel = lib.environment("hotel")

    selection = [_pick(lib, pt, equipment, restrictions, max_difficulty) for pt in patterns]
    minimum = [
        {"exercise_id": lib.default_for(Pattern(pt)).id, "sets": p.sets} for pt in p.patterns
    ]
    hotel_version = [
        {"slot": pt.value, "exercise_id": _pick(lib, pt, hotel, restrictions, 1)["exercise_id"]}
        for pt in patterns
    ]
    review = (view.as_of + timedelta(days=14)).date()
    rollback = "动作做不标准、出现疼痛或连续两次完成不了时，换成替代动作或降阶"
    facts = (f"训练环境：{environment}", f"身体限制：{'、'.join(restrictions) or '无'}")
    changes = [
        Change(
            field="exercise_selection",
            to_value=selection,
            reason_facts=facts,
            rule_id=LIBRARY_REF,
            expected_effect="每个槽位都有能做、不痛的具体动作",
            rollback_condition=rollback,
        ),
        Change(
            field="minimum_task",
            to_value=minimum,
            reason_facts=("没时间或状态差时的最低版本",),
            rule_id=RULE,
            expected_effect="忙碌的日子也能保持训练习惯",
            rollback_condition=rollback,
        ),
        Change(
            field="alternative_version",
            to_value={"hotel": hotel_version},
            reason_facts=("出差时只用徒手与椅子",),
            rule_id=LIBRARY_REF,
            expected_effect="出差期间也能按槽位训练",
            rollback_condition=rollback,
        ),
    ]
    return proposals(context, view, changes, review)
