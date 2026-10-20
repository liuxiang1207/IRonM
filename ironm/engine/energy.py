"""每日精力与当天训练量调整。各级调整方式取自规则表（recovery 规则集），代码里不写比例。

精力只能让训练量下降，不能超过原计划；当天没有选择精力时不给任何调整。
"""

import math
from collections.abc import Iterable
from datetime import date, tzinfo
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ironm.contracts.events import EnergyCheckEvent
from ironm.contracts.unknown import Unknown, not_recorded
from ironm.core.clock import local_day_bounds
from ironm.engine.effective import resolve
from ironm.rules.loader import RuleBook

type Mode = Literal["full", "reduced", "easy", "minimum"]


def rule_id_for(level: int) -> str:
    return f"R-RC-ENERGY-{level}"


class EnergyParams(BaseModel):
    mode: Mode
    volume_factor: float
    rpe_cap_delta: int
    safety_hint: str | None = None


class EnergyAdjustment(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Mode
    volume_factor: float
    rpe_cap_delta: int
    rule_id: str
    safety_hint: str | None


class EnergyToday(BaseModel):
    model_config = ConfigDict(frozen=True)

    selected: bool
    level: int | Unknown
    adjustment: EnergyAdjustment | Unknown


class AdjustedItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    sets: int
    rpe_cap: float


def adjustment_for(level: int, rulebook: RuleBook) -> EnergyAdjustment:
    rule_id = rule_id_for(level)
    p = rulebook.get(rule_id).params_as(EnergyParams)
    return EnergyAdjustment(
        mode=p.mode,
        volume_factor=p.volume_factor,
        rpe_cap_delta=p.rpe_cap_delta,
        rule_id=rule_id,
        safety_hint=p.safety_hint,
    )


def energy_today(
    events: Iterable[EnergyCheckEvent], *, day: date, tz: tzinfo, rulebook: RuleBook
) -> EnergyToday:
    """用户所在时区当天最后一次精力选择；没有选择时 level 与 adjustment 均为“未知（没记录）”。"""
    start, end = local_day_bounds(day, tz)
    todays = [e for e in resolve(events) if start <= e.occurred_at < end]
    if not todays:
        return EnergyToday(selected=False, level=not_recorded(), adjustment=not_recorded())
    latest = max(todays, key=lambda e: (e.occurred_at, str(e.event_id)))
    level = latest.payload.level
    return EnergyToday(selected=True, level=level, adjustment=adjustment_for(level, rulebook))


def adjust_item(
    *, planned_sets: int, planned_rpe_cap: float, adjustment: EnergyAdjustment
) -> AdjustedItem:
    """一个训练项调整后的组数与用力程度上限；只降不升。"""
    if adjustment.mode == "minimum":
        return AdjustedItem(sets=0, rpe_cap=planned_rpe_cap)
    # 四舍五入（0.5 进位）；Python 的 round 是银行家舍入，2.5 会变成 2
    sets = max(1, math.floor(planned_sets * adjustment.volume_factor + 0.5)) if planned_sets else 0
    return AdjustedItem(
        sets=min(planned_sets, sets),
        rpe_cap=planned_rpe_cap + min(0, adjustment.rpe_cap_delta),
    )
