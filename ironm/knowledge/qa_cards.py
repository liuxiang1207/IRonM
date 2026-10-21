"""问答卡（spec [2026-10-21] §6）：审定的标准答案，按统一结构分段；也是 L1 回答的知识来源。

匹配只做查表：每张卡有几组说法，每组至少命中一个才算匹配；组数越多越具体，排在前面。
卡里的数字视为登记过的数字（spec §7）。
"""

import json
from functools import cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, ValidationError, model_validator

from ironm.contracts.common import ContractModel
from ironm.contracts.roles import AgentId
from ironm.core.yamlfile import load_yaml
from ironm.safety.matching import check_term, clauses

__all__ = [
    "CARDS_PATH",
    "CardBook",
    "CardLoadError",
    "PartKind",
    "QACard",
    "default_cards",
    "load_cards",
]

CARDS_PATH = Path(__file__).with_name("qa_cards.yaml")

type Text = Annotated[str, Field(min_length=1)]
type PartKind = Literal["conclusion", "why", "how", "caution", "uncertain"]
PART_KINDS: tuple[PartKind, ...] = ("conclusion", "why", "how", "caution", "uncertain")


class CardLoadError(ValueError):
    pass


class QACard(ContractModel):
    id: Annotated[str, Field(pattern=r"^QA-[A-Z0-9]+(-[A-Z0-9]+)*$")]
    intents: tuple[Text, ...] = Field(min_length=1)
    agent: AgentId
    match: tuple[tuple[Text, ...], ...] = Field(min_length=1)
    conclusion: Text
    why: Text | None = None
    how: Text | None = None
    caution: Text | None = None
    uncertain: Text | None = None
    sources: tuple[Text, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        for group in self.match:
            if not group:
                raise ValueError(f"问答卡 {self.id} 的匹配说法有空组")
            for term in group:
                check_term(term)
        return self

    def parts(self) -> tuple[tuple[PartKind, str], ...]:
        """按统一结构排列的各部分（不适用的省略）。"""
        values = {k: getattr(self, k) for k in PART_KINDS}
        return tuple((k, v) for k, v in values.items() if v)

    def score(self, text: str) -> int:
        """匹配上的组数；有一组没命中即为 0。"""
        joined = "".join(clauses(text))
        return len(self.match) if all(any(t in joined for t in g) for g in self.match) else 0


class CardBook(ContractModel):
    version: Text
    review: Literal["pending_review", "approved"]
    cards: tuple[QACard, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [c.id for c in self.cards]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"问答卡编号重复：{dup}")
        return self

    def matches(self, text: str, intents: set[str] | None = None) -> list[tuple[QACard, int]]:
        """匹配上的卡，按具体程度（命中组数）从高到低；intents 不为空时只看这些意图的卡。"""
        found = [
            (c, s)
            for c in self.cards
            if (intents is None or intents & set(c.intents)) and (s := c.score(text)) > 0
        ]
        return sorted(found, key=lambda cs: -cs[1])

    def get(self, card_id: str) -> QACard:
        for c in self.cards:
            if c.id == card_id:
                return c
        raise KeyError(card_id)


def load_cards(path: Path = CARDS_PATH) -> CardBook:
    try:
        return CardBook.model_validate_json(json.dumps(load_yaml(path)))
    except (OSError, ValueError, ValidationError) as e:
        raise CardLoadError(f"{path}：问答卡有误：{e}") from e


@cache
def default_cards() -> CardBook:
    return load_cards()
