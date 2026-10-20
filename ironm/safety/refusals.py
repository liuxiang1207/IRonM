"""四项拒答识别（spec §24）：不诊断、不开处方、不制定治疗方案、不提供紧急医疗服务。

匹配方式与急症词表相同（见 matching）；结果不含用户原话。
"""

from functools import cache
from pathlib import Path
from typing import Annotated, Literal, Self, get_args

from pydantic import Field, model_validator

from ironm.contracts.common import ContractModel
from ironm.safety.matching import Phrasebook, Term, load_file

__all__ = ["REFUSALS_PATH", "RefusalBook", "RefusalCheck", "RefusalKind", "detect", "load_refusals"]

REFUSALS_PATH = Path(__file__).with_name("refusal_terms.yaml")

type RefusalKind = Literal["diagnosis", "prescription", "treatment", "emergency_service"]
KINDS: tuple[RefusalKind, ...] = get_args(RefusalKind.__value__)


class RefusalRule(ContractModel):
    id: Annotated[str, Field(pattern=r"^RJ-[A-Z0-9]+(-[A-Z0-9]+)*$")]
    kind: RefusalKind
    title: Term
    all: tuple[Term, ...] = Field(min_length=1)


class RefusalCheck(ContractModel):
    """命中的拒答类型（按固定顺序）与条目编号（按词表顺序）。"""

    kinds: tuple[RefusalKind, ...] = ()
    rule_ids: tuple[str, ...] = ()


class RefusalBook(Phrasebook):
    rules: tuple[RefusalRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        self.check_rules((r.id, r.all) for r in self.rules)
        absent = set(KINDS) - {r.kind for r in self.rules}
        if absent:
            raise ValueError(f"拒答类型没有任何条目：{sorted(absent)}")
        return self

    def detect(self, text: str) -> RefusalCheck:
        found = self.present(text)
        fired = [r for r in self.rules if all(s in found for s in r.all)]
        hit = {r.kind for r in fired}
        return RefusalCheck(
            kinds=tuple(k for k in KINDS if k in hit), rule_ids=tuple(r.id for r in fired)
        )


def load_refusals(path: Path) -> RefusalBook:
    return load_file(path, RefusalBook, "拒答词表")


@cache
def default_refusals() -> RefusalBook:
    return load_refusals(REFUSALS_PATH)


def detect(text: str) -> RefusalCheck:
    """用内置拒答词表判断一句话需要哪些拒答。"""
    return default_refusals().detect(text)
