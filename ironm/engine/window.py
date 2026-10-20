"""判断窗口：先定“需要多确定”，再算“需要多久”，并说明还差什么（spec §19）。参数取自规则表。

模型：窗口 T 天内均匀称重，读数 n = ⌊每周次数 × T / 7⌋ + 1，间隔 d = T / (n−1)，
Sxx = d²·n·(n²−1)/12，可信范围半宽 = t(½ + 置信水平/2, n−2) × 波动 / √Sxx × 7（千克/周）。
"""

import math
from collections.abc import Iterable
from datetime import datetime, tzinfo
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ironm.contracts.events import WeightEvent
from ironm.contracts.metrics import Need
from ironm.contracts.unknown import Unknown, UnknownReason
from ironm.engine.data_quality import effective_weights, pending_weights
from ironm.engine.stats import t_quantile
from ironm.engine.trend import WEIGHT_TREND, TrendParams, weight_trend
from ironm.rules.loader import RuleBook

JUDGEMENT_WINDOW = "R-MT-JUDGEMENT-WINDOW"
_INSUFFICIENT = Unknown(unknown=UnknownReason.INSUFFICIENT_DATA)


class WindowParams(BaseModel):
    target_half_width_kg_per_week: Annotated[float, Field(gt=0)]
    default_residual_sd_kg: Annotated[float, Field(gt=0)]
    min_residual_sd_kg: Annotated[float, Field(gt=0)]
    min_readings_for_sd: Annotated[int, Field(ge=3)]
    frequency_window_days: Annotated[int, Field(gt=0)]
    min_window_days: Annotated[int, Field(ge=1)]
    max_window_days: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.min_window_days > self.max_window_days:
            raise ValueError("min_window_days 不能大于 max_window_days")
        return self


class WindowPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "insufficient"]
    window_days: int | Unknown
    needed: tuple[Need, ...]
    rule_id: str


class WindowAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "insufficient"]
    required_days: int | Unknown
    covered_days: int
    readings_per_week: float
    residual_sd_kg: float
    sd_source: Literal["estimated", "default"]
    faster_by_days: int | Unknown  # 每周多称一次可提前的天数
    needed: tuple[Need, ...]
    rule_id: str


def _half_width(days: int, per_week: float, sd: float, ci_level: float) -> float | None:
    n = math.floor(per_week * days / 7) + 1
    if n < 3:
        return None
    gap = days / (n - 1)
    sxx = gap * gap * n * (n * n - 1) / 12
    return t_quantile(0.5 + ci_level / 2, n - 2) * sd / math.sqrt(sxx) * 7


def _shortest(per_week: float, sd: float, p: WindowParams, ci_level: float) -> int | None:
    for days in range(p.min_window_days, p.max_window_days + 1):
        half = _half_width(days, per_week, sd, ci_level)
        if half is not None and half <= p.target_half_width_kg_per_week:
            return days
    return None


def required_window_days(
    *, rulebook: RuleBook, readings_per_week: float, residual_sd_kg: float
) -> WindowPlan:
    """满足确定程度要求的最少天数；超过上限为数据不足，并说明每周至少需称几次。"""
    p = rulebook.get(JUDGEMENT_WINDOW).params_as(WindowParams)
    ci_level = rulebook.get(WEIGHT_TREND).params_as(TrendParams).ci_level
    if readings_per_week <= 0:
        need = (Need(item="weigh_ins_per_week", count=1),)
        return WindowPlan(
            status="insufficient", window_days=_INSUFFICIENT, needed=need, rule_id=JUDGEMENT_WINDOW
        )
    days = _shortest(readings_per_week, residual_sd_kg, p, ci_level)
    if days is not None:
        return WindowPlan(status="ok", window_days=days, needed=(), rule_id=JUDGEMENT_WINDOW)
    enough = next(
        (
            f
            for f in range(math.floor(readings_per_week) + 1, 8)
            if _shortest(f, residual_sd_kg, p, ci_level)
        ),
        None,
    )
    # 每周称 7 次也不够：再多称没有用，提示固定时间称重以减少波动
    need = (
        (Need(item="weigh_ins_per_week", count=enough),)
        if enough is not None
        else (Need(item="consistent_weighing_time", count=1),)
    )
    return WindowPlan(
        status="insufficient", window_days=_INSUFFICIENT, needed=need, rule_id=JUDGEMENT_WINDOW
    )


def judgement_window(
    events: Iterable[WeightEvent], *, as_of: datetime, tz: tzinfo, rulebook: RuleBook
) -> WindowAssessment:
    """按用户实际的称重频率与波动，判断现在的数据够不够下结论；不够时说明还差什么。"""
    p = rulebook.get(JUDGEMENT_WINDOW).params_as(WindowParams)
    items = effective_weights(events, as_of=as_of)
    pending = pending_weights(items, rulebook)
    trusted = [e for e in items if e.event_id not in pending]
    if trusted:
        stream = trusted[-1].stream_key  # 不同的秤不混算
        trusted = [e for e in trusted if e.stream_key == stream]
    today = as_of.astimezone(tz).date()

    def local_age(e: WeightEvent) -> int:
        return (today - e.occurred_at.astimezone(tz).date()).days

    recent_days = {local_age(e) for e in trusted if local_age(e) < p.frequency_window_days}
    per_week = len(recent_days) / (p.frequency_window_days / 7)

    sd_source: Literal["estimated", "default"]
    fit = weight_trend(trusted, as_of=as_of, window_days=p.frequency_window_days, rulebook=rulebook)
    if (
        fit.status == "ok"
        and fit.n >= p.min_readings_for_sd
        and isinstance(fit.residual_sd_kg, float)
    ):
        sd, sd_source = max(fit.residual_sd_kg, p.min_residual_sd_kg), "estimated"
    else:
        sd, sd_source = p.default_residual_sd_kg, "default"

    plan = required_window_days(rulebook=rulebook, readings_per_week=per_week, residual_sd_kg=sd)
    in_reach = [local_age(e) for e in trusted if local_age(e) <= p.max_window_days]
    covered = max(in_reach) if in_reach else 0

    faster: int | Unknown = _INSUFFICIENT
    needed = plan.needed
    status: Literal["ok", "insufficient"] = plan.status
    if isinstance(plan.window_days, int):
        more = required_window_days(
            rulebook=rulebook, readings_per_week=min(per_week + 1, 7), residual_sd_kg=sd
        )
        faster = plan.window_days - more.window_days if isinstance(more.window_days, int) else 0
        if covered < plan.window_days:
            status = "insufficient"
            needed = (Need(item="days", count=plan.window_days - covered),)
    return WindowAssessment(
        status=status,
        required_days=plan.window_days,
        covered_days=covered,
        readings_per_week=per_week,
        residual_sd_kg=sd,
        sd_source=sd_source,
        faster_by_days=max(0, faster) if isinstance(faster, int) else faster,
        needed=needed,
        rule_id=JUDGEMENT_WINDOW,
    )


def extra_days_for_target(window_days: int, half_width: float, target: float) -> int:
    """实际半宽超过目标时，大约还需多少天。标准误约与天数的 1.5 次方成反比。"""
    if half_width <= target:
        return 0
    needed: float = window_days * (half_width / target) ** (2 / 3)
    return max(1, math.ceil(needed - window_days))
