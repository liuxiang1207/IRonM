"""基线测试方案（spec §31）：测试项、按分层选择、测前筛查。数据在 baseline_tests.yaml，由人维护。"""

import json
from functools import cache
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, ValidationError, model_validator

from ironm.contracts.common import ContractModel
from ironm.contracts.screening import Tier
from ironm.contracts.unknown import Unknown
from ironm.core.yamlfile import load_yaml
from ironm.safety.decision import plan_allowed

__all__ = [
    "BASELINE_PATH",
    "BaselineBook",
    "BaselineLoadError",
    "BaselineTest",
    "Protocol",
    "default_baseline",
    "load_baseline",
]

BASELINE_PATH = Path(__file__).with_name("baseline_tests.yaml")

type Text = Annotated[str, Field(min_length=1)]


class BaselineLoadError(ValueError):
    pass


class BaselineTest(ContractModel):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    version: Annotated[int, Field(ge=1)]
    name: Text
    unit: Literal["reps", "s", "bpm"]
    kind: Literal["fitness", "vital"]  # 体能项目在测前筛查不通过时跳过
    tiers: tuple[Tier, ...] = Field(min_length=1)
    instructions: tuple[Text, ...] = Field(min_length=2, max_length=4)
    common_errors: tuple[Text, ...] = Field(min_length=1, max_length=3)
    stop_when: tuple[Text, ...] = Field(min_length=1, max_length=3)

    @property
    def test_id(self) -> str:
        """记录中使用的编号：只与同一版本比较。"""
        return f"{self.id}@{self.version}"


class Protocol(ContractModel):
    """本次要做的测试项（编号@版本，按文件顺序）与跳过的原因。"""

    test_ids: tuple[str, ...]
    skipped_reason: Text | None = None


class BaselineBook(ContractModel):
    version: Text
    review: Literal["pending_review", "approved"]
    tests: tuple[BaselineTest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [t.id for t in self.tests]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"测试项编号重复：{dup}")
        if not any(t.kind == "vital" for t in self.tests):
            raise ValueError("至少要有一项任何人都能测的生命体征")
        for t in self.tests:
            blocked = [tier for tier in t.tiers if not plan_allowed(tier)]
            if t.kind == "fitness" and blocked:
                raise ValueError(f"体能项目 {t.id} 不能适用于不可给计划的分层：{blocked}")
        return self

    def get(self, test_id: str) -> BaselineTest:
        for t in self.tests:
            if t.test_id == test_id:
                return t
        raise KeyError(f"测试项不存在：{test_id}")

    def protocol(
        self, tier: Tier | Unknown, *, unwell_today: bool, recent_injury: bool
    ) -> Protocol:
        """按分层与测前筛查选测试项。分层未知、今天不适或近期受伤时只测生命体征。"""
        reason: str | None = None
        if isinstance(tier, Unknown):
            reason = "tier_unknown"
        elif unwell_today:
            reason = "pretest_unwell"
        elif recent_injury:
            reason = "pretest_recent_injury"
        chosen = [
            t
            for t in self.tests
            if (t.kind == "vital" or reason is None)
            and (isinstance(tier, Unknown) or tier in t.tiers)
        ]
        if reason is None and not any(t.kind == "fitness" for t in chosen):
            reason = "tier_no_fitness_test"  # 暂停、C、D、不适用：不做体能测试
        return Protocol(test_ids=tuple(t.test_id for t in chosen), skipped_reason=reason)


def load_baseline(path: Path) -> BaselineBook:
    try:
        return BaselineBook.model_validate_json(json.dumps(load_yaml(path)))
    except ValidationError as e:
        detail = "；".join(f"{'.'.join(map(str, x['loc']))}：{x['msg']}" for x in e.errors())
        raise BaselineLoadError(f"{path}：基线测试方案有误：{detail}") from e
    except (OSError, ValueError, TypeError, yaml.YAMLError) as e:
        raise BaselineLoadError(f"{path}：基线测试方案有误：{e}") from e


@cache
def default_baseline() -> BaselineBook:
    return load_baseline(BASELINE_PATH)
