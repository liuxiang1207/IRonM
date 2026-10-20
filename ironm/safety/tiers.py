"""人群安全分层与实时重新分层（spec §23）。

界值与健康状况的归属取自规则表 R-SF-TIER；分层顺序写在这里。
安全模块不导入规则模块：只通过共享契约里的 Rule 读取参数（调用方传入规则簿）。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, Protocol, Self, runtime_checkable
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, model_validator

from ironm.contracts.rules import Rule
from ironm.contracts.screening import (
    TIER_ORDER,
    Condition,
    SafetyFlag,
    Screening,
    Sex,
    Tier,
    TierResult,
)
from ironm.contracts.unknown import Unknown, UnknownReason, is_known
from ironm.core.clock import age_on
from ironm.safety.emergency import Assessment

__all__ = [
    "TIER_RULE",
    "RuleLookup",
    "TierParams",
    "classify",
    "note_message",
    "tier_change",
]

TIER_RULE = "R-SF-TIER"
EATING_DISORDER_CATEGORY = "eating_disorder"  # 急症词表中的类别：命中即记为进食障碍信号


@runtime_checkable
class RuleLookup(Protocol):
    """按编号取规则（规则簿满足此接口）。"""

    def get(self, rule_id: str) -> Rule: ...


class TierParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    adult_age: int
    senior_age: int
    bmi_underweight: float
    bmi_overweight: float
    bmi_staged: float
    waist_male_cm: float
    waist_female_cm: float
    # 规则表里是文字；严格模式下文字不会自动转成枚举，这里按名称校验后再转
    c_conditions: tuple[str, ...]
    collaborative_conditions: tuple[str, ...]
    b_conditions: tuple[str, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not 0 < self.adult_age < self.senior_age:
            raise ValueError("年龄界值应满足 0 < 成年 < 老年")
        if not 0 < self.bmi_underweight < self.bmi_overweight < self.bmi_staged:
            raise ValueError("BMI 界值应依次增大")
        if self.waist_male_cm <= 0 or self.waist_female_cm <= 0:
            raise ValueError("腰围界值应为正数")
        groups = [*self.c_conditions, *self.collaborative_conditions, *self.b_conditions]
        unknown = [c for c in groups if c not in set(Condition)]
        if unknown:
            raise ValueError(f"不认识的健康状况：{unknown}")
        dup = {c for c in groups if groups.count(c) > 1}
        if dup:
            raise ValueError(f"健康状况只能归入一层：{sorted(dup)}")
        absent = set(Condition) - set(groups)
        if absent:
            raise ValueError(f"健康状况没有归入任何一层：{sorted(absent)}")
        return self


# 三值判断：True 满足、False 不满足、None 信息不足无法判断
type Tri = bool | None


@dataclass
class _Check:
    """一层的判断结果：是否满足、依据代码、无法判断时缺哪些信息。"""

    value: Tri
    reasons: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _any(parts: Sequence[_Check]) -> _Check:
    """任一满足即满足；没有满足且有无法判断的，整体无法判断。"""
    hits = [p for p in parts if p.value is True]
    if hits:
        return _Check(True, [r for p in hits for r in p.reasons])
    unknown = [p for p in parts if p.value is None]
    if unknown:
        return _Check(None, missing=[m for p in unknown for m in p.missing])
    return _Check(False)


def _bmi(height_cm: float, weight_kg: float) -> float:
    """BMI，按一位小数四舍五入（用十进制计算，避免二进制小数在界值处出错）。"""
    metres = Decimal(str(height_cm)) / 100
    value = Decimal(str(weight_kg)) / (metres * metres)
    return float(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _paused(flags: Sequence[SafetyFlag], clearances: Sequence[datetime]) -> _Check:
    if not flags:
        return _Check(False)
    last_flag = max(f.at for f in flags)
    # 就医确认必须晚于最后一次红旗；同一时刻按未确认处理
    if any(c > last_flag for c in clearances):
        return _Check(False)
    return _Check(True, ["red_flag_uncleared"])


def _has(conditions: frozenset[Condition] | None, group: Sequence[Condition]) -> _Check:
    if conditions is None:
        return _Check(None, missing=["conditions"])
    hits = [c for c in group if c in conditions]
    return _Check(bool(hits), [f"condition:{c}" for c in hits])


def _derived_conditions(flags: Sequence[SafetyFlag]) -> frozenset[Condition]:
    """从红旗记录推出的健康状况：命中进食障碍类条目即为进食障碍信号（就医确认后仍保留）。"""
    if any(EATING_DISORDER_CATEGORY in f.categories for f in flags):
        return frozenset({Condition.EATING_DISORDER_SIGNAL})
    return frozenset()


def _compare(value: float | None, at_least: float, reason: str, missing: list[str]) -> _Check:
    if value is None:
        return _Check(None, missing=missing)
    return _Check(value >= at_least, [reason] if value >= at_least else [])


def _waist(s: Screening, p: TierParams) -> _Check:
    """腰围达到中心型肥胖前期。性别未知时，两个界值都达到才算满足、都没达到才算不满足。"""
    if not is_known(s.waist_cm):
        return _Check(None, missing=["waist_cm"])
    waist = s.waist_cm
    if is_known(s.sex):
        limit = p.waist_male_cm if s.sex is Sex.MALE else p.waist_female_cm
        return _compare(waist, limit, "waist_at_least_central", [])
    if waist >= max(p.waist_male_cm, p.waist_female_cm):
        return _Check(True, ["waist_at_least_central"])
    if waist < min(p.waist_male_cm, p.waist_female_cm):
        return _Check(False)
    return _Check(None, missing=["sex"])


def classify(
    screening: Screening, *, as_of: datetime, tz: ZoneInfo, rulebook: RuleLookup
) -> TierResult:
    """按筛查信息分层。只计入截至 as_of 的红旗与就医确认；年龄按 tz 的当地日期。"""
    p = rulebook.get(TIER_RULE).params_as(TierParams)
    s = screening
    flags = [f for f in s.flags if f.at <= as_of]
    clearances = [c for c in s.clearances if c <= as_of]
    today = as_of.astimezone(tz).date()

    age = age_on(s.birth_date, today) if is_known(s.birth_date) else None
    bmi = (
        _bmi(s.height_cm, s.weight_kg) if is_known(s.height_cm) and is_known(s.weight_kg) else None
    )
    bmi_missing = [f for f in ("height_cm", "weight_kg") if not is_known(getattr(s, f))]
    derived = _derived_conditions(flags)
    conditions = frozenset(s.conditions) | derived if is_known(s.conditions) else None

    def from_conditions(names: Sequence[str]) -> _Check:
        group = [Condition(n) for n in names]
        # 推出的状况即使问卷没答也成立
        hits = [c for c in group if c in derived]
        if hits:
            return _Check(True, [f"condition:{c}" for c in hits])
        return _has(conditions, group)

    def bmi_below(limit: float) -> _Check:
        if bmi is None:
            return _Check(None, missing=bmi_missing)
        return _Check(bmi < limit, ["bmi_below_underweight"] if bmi < limit else [])

    age_missing = ["birth_date"]
    checks: dict[Tier, Callable[[], _Check]] = {
        Tier.PAUSED: lambda: _paused(flags, clearances),
        Tier.D: lambda: (
            _Check(None, missing=age_missing)
            if age is None
            else _Check(age < p.adult_age, ["age_below_adult"] if age < p.adult_age else [])
        ),
        Tier.C: lambda: from_conditions(p.c_conditions),
        Tier.NOT_APPLICABLE: lambda: bmi_below(p.bmi_underweight),
        Tier.COLLABORATIVE: lambda: from_conditions(p.collaborative_conditions),
        Tier.B: lambda: _any(
            [
                _compare(age, p.senior_age, "age_senior", age_missing),
                from_conditions(p.b_conditions),
                _compare(bmi, p.bmi_staged, "bmi_at_least_staged", bmi_missing),
            ]
        ),
        Tier.A: lambda: _any(
            [_compare(bmi, p.bmi_overweight, "bmi_at_least_overweight", bmi_missing), _waist(s, p)]
        ),
    }
    results = {tier: check() for tier, check in checks.items()}
    # A′：BMI 不低于下限、未达 A 的条件
    a = results[Tier.A]
    results[Tier.A_PRIME] = (
        _Check(True, ["bmi_normal_waist_below"])
        if a.value is False and results[Tier.NOT_APPLICABLE].value is False
        else _Check(False if a.value is True else None)
    )

    matched = tuple(t for t in TIER_ORDER if results[t].value is True)
    reasons = tuple(r for t in matched for r in results[t].reasons)
    bmi_out: float | Unknown = bmi if bmi is not None else _insufficient()

    def result(
        tier: Tier | Unknown, status: Literal["ok", "assumed", "insufficient"], missing: list[str]
    ) -> TierResult:
        return TierResult(
            tier=tier,
            status=status,
            matched=matched,
            reasons=reasons,
            missing=tuple(dict.fromkeys(missing)),
            bmi=bmi_out,
            rule_id=TIER_RULE,
        )

    # 主层：从严到宽，第一个满足的层；更严的层无法判断时给不出分层
    blocked: list[str] = []
    for tier in (t for t in TIER_ORDER if t not in (Tier.A_PRIME, Tier.A)):
        check = results[tier]
        if check.value is None:
            blocked += check.missing
        elif check.value and not blocked:
            return result(tier, "ok", [])
    if blocked:
        return result(_insufficient(), "insufficient", blocked)
    if a.value is True:
        return result(Tier.A, "ok", [])
    if results[Tier.A_PRIME].value is True:
        return result(Tier.A_PRIME, "ok", [])
    # 只缺腰围或性别、BMI 在正常范围：按 A′ 保守假设
    return result(Tier.A_PRIME, "assumed", a.missing)


def _insufficient() -> Unknown:
    return Unknown(unknown=UnknownReason.INSUFFICIENT_DATA)


def note_message(screening: Screening, assessment: Assessment, at: datetime) -> Screening:
    """实时重新分层：一句话判为红旗或急症时，记一条红旗记录（不含原话）。"""
    if assessment.level == "none":
        return screening
    flag = SafetyFlag(
        at=at,
        level=assessment.level,
        rule_ids=assessment.rule_ids,
        categories=assessment.categories,
    )
    return screening.model_copy(update={"flags": (*screening.flags, flag)})


def tier_change(
    before: TierResult, after: TierResult
) -> Literal["stricter", "looser", "same", "unknown"]:
    """新旧分层相比：更严、更松、不变；任一方未知时为 unknown。"""
    if not is_known(before.tier) or not is_known(after.tier):
        return "unknown"
    old, new = TIER_ORDER.index(before.tier), TIER_ORDER.index(after.tier)
    if new == old:
        return "same"
    return "stricter" if new < old else "looser"
