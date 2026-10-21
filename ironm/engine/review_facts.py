"""复盘用的事实（spec [2026-10-21] §17）：只算事实，不做决定。

- 一周从周一算起，按用户时区；周期以周一的日期标识。
- 完成率：与完成度指标同一口径（[2026-10-20] §15）：安排的训练日里，练了 1、练一半 0.5、没练 0
  的平均；没有打卡的训练日不计入（没记录不等于没练）；有记录的天数不够时为“数据不足”（None）。
- 疲劳基线：这周之前最近两次签到的疲劳平均。
"""

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, tzinfo
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.events import (
    CheckinEvent,
    Event,
    MealEvent,
    SetLogEvent,
    SymptomEvent,
    WeightEvent,
    WorkoutCheckEvent,
)
from ironm.contracts.unknown import is_known
from ironm.engine.completeness import training_completion
from ironm.engine.effective import resolve
from ironm.rules.loader import RuleBook

__all__ = [
    "RETURN_RULE",
    "ReturnParams",
    "ReturnState",
    "WeekFacts",
    "return_state",
    "training_days",
    "week_facts",
    "week_start",
]


def week_start(day: date) -> date:
    return day - timedelta(days=day.weekday())


@dataclass(frozen=True)
class WeekFacts:
    week: str  # 周一的日期
    scheduled_days: int
    recorded_days: int
    completion: float | None  # 数据不足时为 None
    avg_rpe: float | None
    fatigue: int | None
    sleep_quality: int | None
    fatigue_baseline: float | None
    partial_week: bool  # 签到说“只记录了一部分”
    menstruation: bool
    diet_adherence_days: int | None
    meal_log_days: int
    weigh_ins: int
    symptoms: int

    def as_review(self, prefix: str = "") -> dict[str, Any]:
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def _local(e: Event, tz: tzinfo) -> date:
    return e.occurred_at.astimezone(tz).date()


def week_facts(
    events: Iterable[Event],
    *,
    schedule: Sequence[int],
    start: date,
    tz: tzinfo,
    as_of: datetime,
    rulebook: RuleBook,
) -> WeekFacts:
    """start 是这一周的周一；as_of 之后才记的记录不算（补跑时数据截止到周期结束）。"""
    end = start + timedelta(days=7)
    usable = [e for e in resolve(list(events), as_of=as_of) if e.recorded_at <= as_of]
    this_week = [e for e in usable if start <= _local(e, tz) < end]

    planned = [start + timedelta(days=d) for d in sorted(set(schedule))]
    done = training_completion(
        [e for e in this_week if isinstance(e, WorkoutCheckEvent)],
        planned_days=planned,
        tz=tz,
        rulebook=rulebook,
    )  # 与完成度指标同一口径（[2026-10-20] §15）
    completion = round(float(done.rate), 3) if done.status == "ok" and is_known(done.rate) else None

    rpes = [
        float(e.payload.rpe)
        for e in this_week
        if isinstance(e, SetLogEvent) and is_known(e.payload.rpe)
    ]
    checkins = sorted(
        (e for e in usable if isinstance(e, CheckinEvent)), key=lambda e: e.occurred_at
    )
    current = [c for c in checkins if start <= _local(c, tz) < end]
    before = [c for c in checkins if _local(c, tz) < start]
    latest = current[-1].payload if current else None
    prior = [c.payload.fatigue for c in before[-2:] if is_known(c.payload.fatigue)]

    def known_int(value: object) -> int | None:
        return value if isinstance(value, int) and is_known(value) else None

    return WeekFacts(
        week=start.isoformat(),
        scheduled_days=len(planned),
        recorded_days=done.recorded_days,
        completion=completion,
        avg_rpe=round(sum(rpes) / len(rpes), 2) if rpes else None,
        fatigue=known_int(latest.fatigue) if latest else None,
        sleep_quality=known_int(latest.sleep_quality) if latest else None,
        fatigue_baseline=round(sum(prior) / len(prior), 2) if len(prior) == 2 else None,
        partial_week=bool(latest and latest.record_completeness == "partial"),
        menstruation=bool(latest and latest.menstruation is True),
        diet_adherence_days=known_int(latest.diet_adherence_days) if latest else None,
        meal_log_days=len({_local(e, tz) for e in this_week if isinstance(e, MealEvent)}),
        weigh_ins=len({_local(e, tz) for e in this_week if isinstance(e, WeightEvent)}),
        symptoms=sum(1 for e in this_week if isinstance(e, SymptomEvent)),
    )


RETURN_RULE = "R-RV-RETURN"


class ReturnParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    gap_days: int = Field(ge=1, le=60)
    rebaseline_after_days: int = Field(ge=1, le=365)
    halve_for_days: int = Field(ge=1, le=60)


@dataclass(frozen=True)
class ReturnState:
    status: str  # normal / welcome（今天回来，训练量减半）/ halved（回来后的头两周）/ rebaseline
    gap_days: int | None  # 最近一次中断的天数


def _trained(e: Event) -> bool:
    if isinstance(e, WorkoutCheckEvent):
        return e.payload.status in ("done", "half")
    if isinstance(e, CheckinEvent):  # 签到说按计划练了但没记，或在别处练
        return e.payload.record_completeness in ("followed_plan_unrecorded", "elsewhere")
    return isinstance(e, SetLogEvent)


def training_days(events: Iterable[Event], tz: tzinfo) -> list[date]:
    """练过的日子：打卡练了或练一半、有按组记录，或签到说练了但没记的那天（先按更正取有效记录）。"""
    return sorted({_local(e, tz) for e in resolve(list(events)) if _trained(e)})


def return_state(
    days: Sequence[date],
    today: date,
    *,
    gap_days: int,
    rebaseline_after_days: int,
    halve_for_days: int,
) -> ReturnState:
    """中断回归（spec [2026-10-21] §17，R-RV-RETURN）。从没练过的新用户不算中断。"""
    past = sorted(d for d in set(days) if d <= today)
    if not past:
        return ReturnState("normal", None)
    away = (today - past[-1]).days
    if away > gap_days:  # 现在正处在中断后回来的这一天
        status = "rebaseline" if away > rebaseline_after_days else "welcome"
        return ReturnState(status, away)
    for before, after in zip(reversed(past[:-1]), reversed(past[1:]), strict=True):
        gap = (after - before).days
        if gap > gap_days:
            back = (today - after).days < halve_for_days
            return ReturnState("halved" if back else "normal", gap)
        if (today - after).days >= halve_for_days:
            break
    return ReturnState("normal", None)
