"""急症词表与匹配（spec §21–22）。

判定只看词表：不调用模型、知识库或其他业务模块，离线与模型故障时照常工作。
匹配方式见 matching；App 离线版（app/src/safety/emergency.ts）用同一份词表实现，结果必须一致。
判定结果不含用户原话。
"""

from functools import cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ironm.contracts.common import ContractModel
from ironm.safety.matching import (
    CLAUSE_SEPARATORS,
    WHITESPACE,
    Phrasebook,
    Term,
    TermLoadError,
    clauses,
    load_file,
)

__all__ = [
    "CLAUSE_SEPARATORS",
    "TERMS_PATH",
    "WHITESPACE",
    "Assessment",
    "TermBook",
    "TermLoadError",
    "assess",
    "clauses",
    "default_book",
    "load_terms",
]

TERMS_PATH = Path(__file__).with_name("emergency_terms.yaml")

type Level = Literal["emergency", "red_flag", "none"]
_RANK: dict[Level, int] = {"none": 0, "red_flag": 1, "emergency": 2}


class EmergencyRule(ContractModel):
    """all 中的症状都出现时给出 level。范围 within（背景状况不受限制，整句任何位置都算）：
    clause 同一分句；nearby 同一分句或相邻两个分句（默认）；sentence 整句。
    """

    id: Annotated[str, Field(pattern=r"^(EM|RF)-[A-Z0-9]+(-[A-Z0-9]+)*$")]
    level: Literal["emergency", "red_flag"]
    category: Term
    title: Term
    all: tuple[Term, ...] = Field(min_length=1)
    within: Literal["clause", "nearby", "sentence"] = "nearby"

    @model_validator(mode="after")
    def _check(self) -> Self:
        prefix = "EM" if self.level == "emergency" else "RF"
        if not self.id.startswith(prefix + "-"):
            raise ValueError(f"条目 {self.id} 的编号前缀应为 {prefix}（与级别一致）")
        return self


class Assessment(ContractModel):
    """判定结果：级别、命中的条目编号与类别（按词表顺序）。不含用户原话。"""

    level: Level
    rule_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()


class TermBook(Phrasebook):
    rules: tuple[EmergencyRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        self.check_rules((r.id, r.all) for r in self.rules)
        if self.negation.past_after is None or not self.negation.recurrence:
            raise ValueError("急症词表必须写明“以前发生过”的标记与再次发生的字样")
        return self

    def fires(self, rule: EmergencyRule, per_clause: list[frozenset[str]]) -> bool:
        anywhere = frozenset().union(*per_clause)
        context = {s.id for s in self.symptoms if s.context}
        if not all(s in anywhere for s in rule.all if s in context):
            return False
        acute = [s for s in rule.all if s not in context]
        if rule.within == "sentence":
            windows = [anywhere]
        else:
            size = 1 if rule.within == "clause" else 2
            windows = [frozenset().union(*per_clause[i : i + size]) for i in range(len(per_clause))]
        return any(all(s in w for s in acute) for w in windows)

    def assess(self, text: str) -> Assessment:
        per_clause = self.present_by_clause(text)
        fired = [r for r in self.rules if self.fires(r, per_clause)]
        levels: list[Level] = [r.level for r in fired]
        level: Level = max(levels, key=_RANK.__getitem__, default="none")
        categories = tuple(dict.fromkeys(r.category for r in fired))
        return Assessment(level=level, rule_ids=tuple(r.id for r in fired), categories=categories)


def load_terms(path: Path) -> TermBook:
    """读取并严格校验急症词表；任何问题都抛 TermLoadError。"""
    return load_file(path, TermBook, "急症词表")


@cache
def default_book() -> TermBook:
    return load_terms(TERMS_PATH)


def assess(text: str) -> Assessment:
    """用内置词表判定一句话。"""
    return default_book().assess(text)
