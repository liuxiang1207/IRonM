"""急症词表与匹配（spec §21）。

判定只看词表：不调用模型、知识库或其他业务模块，离线与模型故障时照常工作。
匹配方式刻意保持简单（文字包含、分句内多组同时出现、血压数值），前端离线版用同一份词表得到相同结果。
判定结果不含用户原话。
"""

import json
import re
import unicodedata
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, ValidationError, model_validator

from ironm.contracts.common import ContractModel
from ironm.core.yamlfile import load_yaml

__all__ = [
    "TERMS_PATH",
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

# 统一后（全角已转半角）的分句符号与空白。前端离线版使用同一组字符。
CLAUSE_SEPARATORS = ",。!?;、\n\r~"
WHITESPACE = " \t\f\v             ​    　﻿"
_CLAUSE_RE = re.compile(f"[{re.escape(CLAUSE_SEPARATORS)}]+")
_SPACE_RE = re.compile(f"[{re.escape(WHITESPACE)}]+")
# 血压：120/80 这样的分数，或“高压 190”“低压 120”；位数在代码里判断，正则保持两端都能用的写法
_BP_PAIR_RE = re.compile(r"([0-9]+)/([0-9]+)")
_BP_HIGH_RE = re.compile(r"高压([0-9]+)")
_BP_LOW_RE = re.compile(r"低压([0-9]+)")


class TermLoadError(ValueError):
    pass


def clauses(text: str) -> list[str]:
    """统一文字（全角转半角、英文转小写）后按分句符号切开，再去掉空白；空分句丢弃。"""
    unified = unicodedata.normalize("NFKC", text).lower()
    parts = (_SPACE_RE.sub("", p) for p in _CLAUSE_RE.split(unified))
    return [p for p in parts if p]


def _check_term(term: str) -> str:
    if not term:
        raise ValueError("说法不能为空")
    if clauses(term) != [term]:
        raise ValueError(f"说法“{term}”不是统一后的写法，或含有分句符号、空白")
    return term


type Term = Annotated[str, Field(min_length=1)]


def _bp_value(digits: str) -> int | None:
    return int(digits) if 2 <= len(digits) <= 3 else None


class BloodPressure(ContractModel):
    context: tuple[Term, ...] = Field(min_length=1)  # 同一分句里须出现的字样，如“血压”
    systolic_at_least: int = Field(gt=0)
    diastolic_at_least: int = Field(gt=0)

    def matches(self, clause: str) -> bool:
        if not any(c in clause for c in self.context):
            return False
        systolic: list[int | None] = [_bp_value(m.group(1)) for m in _BP_PAIR_RE.finditer(clause)]
        diastolic: list[int | None] = [_bp_value(m.group(2)) for m in _BP_PAIR_RE.finditer(clause)]
        systolic += [_bp_value(m.group(1)) for m in _BP_HIGH_RE.finditer(clause)]
        diastolic += [_bp_value(m.group(1)) for m in _BP_LOW_RE.finditer(clause)]
        return any(v is not None and v >= self.systolic_at_least for v in systolic) or any(
            v is not None and v >= self.diastolic_at_least for v in diastolic
        )


class Symptom(ContractModel):
    """一组同义说法：terms 任一出现即命中；combos 每项是几组词，同一分句里每组都出现才命中。"""

    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    name: Term
    terms: tuple[Term, ...] = ()
    combos: tuple[tuple[tuple[Term, ...], ...], ...] = ()
    blood_pressure: BloodPressure | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not (self.terms or self.combos or self.blood_pressure):
            raise ValueError(f"症状 {self.id} 没有任何说法")
        for combo in self.combos:
            if len(combo) < 2 or any(not group for group in combo):
                raise ValueError(f"症状 {self.id} 的组合至少两组，且每组至少一个说法")
        words = [*self.terms, *(t for combo in self.combos for g in combo for t in g)]
        if self.blood_pressure is not None:
            words += self.blood_pressure.context
        for word in words:
            _check_term(word)
        return self

    def matches(self, clause: str) -> bool:
        if any(t in clause for t in self.terms):
            return True
        if any(all(any(t in clause for t in g) for g in combo) for combo in self.combos):
            return True
        return self.blood_pressure is not None and self.blood_pressure.matches(clause)


class EmergencyRule(ContractModel):
    id: Annotated[str, Field(pattern=r"^(EM|RF)-[A-Z0-9]+(-[A-Z0-9]+)*$")]
    level: Literal["emergency", "red_flag"]
    category: Term
    title: Term
    all: tuple[Term, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        prefix = "EM" if self.level == "emergency" else "RF"
        if not self.id.startswith(prefix + "-"):
            raise ValueError(f"条目 {self.id} 的编号前缀应为 {prefix}（与级别一致）")
        if len(set(self.all)) != len(self.all):
            raise ValueError(f"条目 {self.id} 重复引用了同一症状")
        return self


class Assessment(ContractModel):
    """判定结果：级别、命中的条目编号与类别（按词表顺序）。不含用户原话。"""

    level: Level
    rule_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()


class TermBook(ContractModel):
    version: Term
    review: Literal["pending_review", "approved"]
    symptoms: tuple[Symptom, ...] = Field(min_length=1)
    rules: tuple[EmergencyRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _unique((s.id for s in self.symptoms), "症状")
        _unique((r.id for r in self.rules), "条目")
        known = {s.id for s in self.symptoms}
        used: set[str] = set()
        for rule in self.rules:
            missing = [s for s in rule.all if s not in known]
            if missing:
                raise ValueError(f"条目 {rule.id} 引用了不存在的症状：{missing}")
            used.update(rule.all)
        unused = [s.id for s in self.symptoms if s.id not in used]
        if unused:
            raise ValueError(f"症状没有被任何条目使用：{unused}")
        return self

    def present(self, text: str) -> frozenset[str]:
        """这句话里出现的症状编号。"""
        parts = clauses(text)
        return frozenset(s.id for s in self.symptoms if any(s.matches(c) for c in parts))

    def assess(self, text: str) -> Assessment:
        found = self.present(text)
        fired = [r for r in self.rules if all(s in found for s in r.all)]
        levels: list[Level] = [r.level for r in fired]
        level: Level = max(levels, key=_RANK.__getitem__, default="none")
        categories = tuple(dict.fromkeys(r.category for r in fired))
        return Assessment(level=level, rule_ids=tuple(r.id for r in fired), categories=categories)


def _unique(ids: Iterable[str], kind: str) -> None:
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"{kind}编号重复：{i}")
        seen.add(i)


def load_terms(path: Path) -> TermBook:
    """读取并严格校验词表；任何问题都抛 TermLoadError。"""
    try:
        data = load_yaml(path)
        return TermBook.model_validate_json(json.dumps(data))
    except (OSError, ValueError, TypeError) as e:
        # ValidationError 与 YAML 错误都是 ValueError 的子类
        detail = e if not isinstance(e, ValidationError) else _summary(e)
        raise TermLoadError(f"{path}：急症词表有误：{detail}") from e
    except yaml.YAMLError as e:
        raise TermLoadError(f"{path}：急症词表格式错误：{e}") from e


def _summary(e: ValidationError) -> str:
    return "；".join(f"{'.'.join(map(str, err['loc']))}：{err['msg']}" for err in e.errors())


@cache
def default_book() -> TermBook:
    return load_terms(TERMS_PATH)


def assess(text: str) -> Assessment:
    """用内置词表判定一句话。"""
    return default_book().assess(text)
