"""动作库：加载、校验与筛选（spec §27）。

数据在 exercises.yaml，由人维护。筛选只做查表，不调用模型：
按动作模式、可用器材、身体限制与最高难度给出候选；“谨慎”的限制附上提示。
"""

import json
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, ValidationError, model_validator

from ironm.contracts.common import ContractModel
from ironm.contracts.exercises import Equipment, Exercise, Pattern, Restriction
from ironm.core.yamlfile import load_yaml

__all__ = [
    "LIBRARY_PATH",
    "Choice",
    "Library",
    "LibraryLoadError",
    "default_library",
    "load_library",
]

LIBRARY_PATH = Path(__file__).with_name("exercises.yaml")

type Text = Annotated[str, Field(min_length=1)]
type Environment = Literal["hotel", "home", "gym"]
ENVIRONMENTS: tuple[Environment, ...] = ("hotel", "home", "gym")


class LibraryLoadError(ValueError):
    pass


class RestrictionText(ContractModel):
    name: Text
    caution: Text  # 选了“谨慎”的动作时附上的提示


class Choice(ContractModel):
    """一个候选动作；cautions 为用户的限制中该动作标为“谨慎”的提示。"""

    exercise_id: str
    name: Text
    difficulty: int
    cautions: tuple[Text, ...] = ()


class Library(ContractModel):
    version: Text
    review: Literal["pending_review", "approved"]
    restrictions: dict[Restriction, RestrictionText]
    environments: dict[Environment, tuple[Equipment, ...]]
    exercises: tuple[Exercise, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        missing = set(Restriction) - set(self.restrictions)
        if missing:
            raise ValueError(f"身体限制缺少说明：{sorted(missing)}")
        if set(self.environments) != set(ENVIRONMENTS):
            raise ValueError(f"环境必须且只能是 {ENVIRONMENTS}")
        hotel, home, gym = (set(self.environments[e]) for e in ENVIRONMENTS)
        if not hotel <= home <= gym or gym != set(Equipment):
            raise ValueError("器材应满足 酒店 ⊆ 家里 ⊆ 健身房，且健身房包含全部器材")
        by_id: dict[str, Exercise] = {}
        for e in self.exercises:
            if e.id in by_id:
                raise ValueError(f"动作编号重复：{e.id}")
            by_id[e.id] = e
        for e in self.exercises:
            for link, harder in (("easier", False), ("harder", True)):
                target_id = getattr(e, link)
                if target_id is None:
                    continue
                target = by_id.get(target_id)
                if target is None:
                    raise ValueError(f"{e.id}.{link} 引用了不存在的动作：{target_id}")
                if target.pattern is not e.pattern:
                    raise ValueError(f"{e.id}.{link} 与 {target_id} 不是同一动作模式")
                if (
                    (target.difficulty < e.difficulty)
                    if harder
                    else (target.difficulty > e.difficulty)
                ):
                    raise ValueError(f"{e.id}.{link} 的难度方向不对：{target_id}")
        for e in self.exercises:  # 引用都已核对，再查是否绕回自己
            for link in ("easier", "harder"):
                _check_chain(e, link, by_id)
        for pattern in Pattern:
            defaults = [e for e in self.exercises if e.pattern is pattern and e.default]
            if len(defaults) != 1:
                raise ValueError(
                    f"动作模式 {pattern} 应恰好有一个默认安全动作，实际 {len(defaults)} 个"
                )
            d = defaults[0]
            if d.difficulty != 1 or d.avoid_if or not set(d.equipment) <= hotel:
                raise ValueError(f"默认安全动作 {d.id} 应为难度 1、酒店可做、没有“避免”的限制")
        return self

    def get(self, exercise_id: str) -> Exercise:
        for e in self.exercises:
            if e.id == exercise_id:
                return e
        raise KeyError(f"动作不存在：{exercise_id}")

    def environment(self, name: Environment) -> frozenset[Equipment]:
        return frozenset(self.environments[name])

    def default_for(self, pattern: Pattern) -> Exercise:
        return next(e for e in self.exercises if e.pattern is pattern and e.default)

    def _choice(self, e: Exercise, restrictions: frozenset[Restriction]) -> Choice:
        cautions = tuple(self.restrictions[r].caution for r in e.caution_if if r in restrictions)
        return Choice(exercise_id=e.id, name=e.name, difficulty=e.difficulty, cautions=cautions)

    def _eligible(
        self,
        e: Exercise,
        equipment: frozenset[Equipment],
        restrictions: frozenset[Restriction],
        max_difficulty: int,
    ) -> bool:
        return (
            set(e.equipment) <= equipment
            and not set(e.avoid_if) & restrictions
            and e.difficulty <= max_difficulty
        )

    def candidates(
        self,
        pattern: Pattern,
        *,
        equipment: Iterable[Equipment],
        restrictions: Iterable[Restriction] = (),
        max_difficulty: int = 3,
    ) -> tuple[Choice, ...]:
        """该模式下可用的动作，按难度从低到高（同难度按编号）。"""
        eq, rs = frozenset(equipment), frozenset(restrictions)
        found = [
            e
            for e in self.exercises
            if e.pattern is pattern and self._eligible(e, eq, rs, max_difficulty)
        ]
        found.sort(key=lambda e: (e.difficulty, e.id))
        return tuple(self._choice(e, rs) for e in found)

    def substitutes(
        self,
        exercise_id: str,
        *,
        equipment: Iterable[Equipment],
        restrictions: Iterable[Restriction] = (),
        max_difficulty: int = 3,
    ) -> tuple[Choice, ...]:
        """替代动作：同一模式的其他候选。"""
        e = self.get(exercise_id)
        return tuple(
            c
            for c in self.candidates(
                e.pattern,
                equipment=equipment,
                restrictions=restrictions,
                max_difficulty=max_difficulty,
            )
            if c.exercise_id != exercise_id
        )

    def regress(
        self,
        exercise_id: str,
        *,
        equipment: Iterable[Equipment],
        restrictions: Iterable[Restriction] = (),
    ) -> Choice:
        """降阶：沿“更容易”一路找第一个可用的；找不到时给该模式的默认安全动作。"""
        eq, rs = frozenset(equipment), frozenset(restrictions)
        current = self.get(exercise_id)
        while current.easier is not None:
            current = self.get(current.easier)
            if self._eligible(current, eq, rs, 3):
                return self._choice(current, rs)
        return self._choice(self.default_for(current.pattern), rs)


def _check_chain(start: Exercise, link: str, by_id: dict[str, Exercise]) -> None:
    seen = {start.id}
    current: str | None = getattr(start, link)
    while current is not None:
        if current in seen:
            raise ValueError(f"{start.id} 沿 {link} 绕回了 {current}")
        seen.add(current)
        current = getattr(by_id[current], link)


def load_library(path: Path) -> Library:
    """读取并严格校验动作库；任何问题都抛 LibraryLoadError。"""
    try:
        return Library.model_validate_json(json.dumps(load_yaml(path)))
    except ValidationError as e:
        detail = "；".join(f"{'.'.join(map(str, x['loc']))}：{x['msg']}" for x in e.errors())
        raise LibraryLoadError(f"{path}：动作库有误：{detail}") from e
    except (OSError, ValueError, TypeError, yaml.YAMLError) as e:
        raise LibraryLoadError(f"{path}：动作库有误：{e}") from e


@cache
def default_library() -> Library:
    return load_library(LIBRARY_PATH)
