"""精度判定：按近 4 周实际记录判定各领域 P0–P3。周数、天数、比例取自规则表（precision 规则集）。

两层：*_from_days / *_from_levels 是纯规则判定（黄金用例逐档卡边界）；
body_precision / training_precision 从记录汇总出判定所需的输入，遵守全局规则清单（spec §16）。
"""

from collections.abc import Iterable, Iterator
from datetime import date, datetime, timedelta, tzinfo
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ironm.contracts.common import Precision
from ironm.contracts.events import (
    CheckinEvent,
    EnergyCheckEvent,
    MealEvent,
    SetLogEvent,
    Source,
    WeightEvent,
    WorkoutCheckEvent,
)
from ironm.contracts.unknown import Unknown
from ironm.engine.effective import resolve
from ironm.rules.loader import RuleBook

BODY = "R-PR-BODY"
TRAINING = "R-PR-TRAINING"
DIET = "R-PR-DIET"
RECOVERY = "R-PR-RECOVERY"
_LEVELS = (Precision.P3, Precision.P2, Precision.P1)


class BodyParams(BaseModel):
    window_days: Annotated[int, Field(gt=0)]
    p3_total_days: Annotated[int, Field(ge=0)]
    p3_weekly_days: Annotated[int, Field(ge=1, le=7)]
    p2_weekly_days: Annotated[int, Field(ge=1, le=7)]
    p1_weekly_days: Annotated[int, Field(ge=1, le=7)]
    weeks_required: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.window_days % 7:
            raise ValueError("window_days 必须是 7 的倍数")
        if self.weeks_required > self.window_days // 7:
            raise ValueError("weeks_required 不能超过窗口内的周数")
        return self


class TrainingParams(BaseModel):
    window_days: Annotated[int, Field(gt=0)]
    min_training_days: Annotated[int, Field(ge=1)]
    share_required: Annotated[float, Field(gt=0, le=1)]


class _WeeklyWindow(BaseModel):
    window_days: Annotated[int, Field(gt=0)]
    weeks_required: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _check_weeks(self) -> Self:
        if self.window_days % 7:
            raise ValueError("window_days 必须是 7 的倍数")
        if self.weeks_required > self.window_days // 7:
            raise ValueError("weeks_required 不能超过窗口内的周数")
        return self


class DietParams(_WeeklyWindow):
    min_logged_days: Annotated[int, Field(ge=1)]
    share_required: Annotated[float, Field(gt=0, le=1)]


class RecoveryParams(_WeeklyWindow):
    min_energy_days: Annotated[int, Field(ge=1)]


class PrecisionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    precision: Precision
    rule_id: str
    basis: dict[str, object]  # 判定依据，供依据卡展示


def body_precision_from_days(*, rulebook: RuleBook, weigh_days: Iterable[int]) -> PrecisionResult:
    """weigh_days：窗口内有称重的日序号（0 起，第 n 周为 7n 到 7n+6）。"""
    p = rulebook.get(BODY).params_as(BodyParams)
    days = sorted(set(weigh_days))
    if any(not 0 <= d < p.window_days for d in days):
        raise ValueError(f"日序号必须在 0–{p.window_days - 1} 之间")
    weekly = [sum(1 for d in days if d // 7 == w) for w in range(p.window_days // 7)]

    def weeks_with(at_least: int) -> int:
        return sum(1 for n in weekly if n >= at_least)

    if len(days) >= p.p3_total_days and weeks_with(p.p3_weekly_days) >= p.weeks_required:
        level = Precision.P3
    elif weeks_with(p.p2_weekly_days) >= p.weeks_required:
        level = Precision.P2
    elif weeks_with(p.p1_weekly_days) >= p.weeks_required:
        level = Precision.P1
    else:
        level = Precision.P0
    return PrecisionResult(
        precision=level, rule_id=BODY, basis={"weigh_days": len(days), "weekly_days": weekly}
    )


def training_precision_from_levels(
    *, rulebook: RuleBook, day_levels: Iterable[int]
) -> PrecisionResult:
    """day_levels：每个训练日的记录程度（1–3）。"""
    p = rulebook.get(TRAINING).params_as(TrainingParams)
    levels = list(day_levels)
    if any(level not in (1, 2, 3) for level in levels):
        raise ValueError("训练日记录程度只能是 1、2、3")
    result = Precision.P0
    if len(levels) >= p.min_training_days:
        for precision, at_least in zip(_LEVELS, (3, 2, 1), strict=True):
            share = sum(1 for level in levels if level >= at_least) / len(levels)
            if share >= p.share_required:
                result = precision
                break
    return PrecisionResult(
        precision=result,
        rule_id=TRAINING,
        basis={"training_days": len(levels), "day_levels": levels},
    )


def diet_precision_from_days(
    *, rulebook: RuleBook, day_levels: Iterable[int], adherence_weeks: int
) -> PrecisionResult:
    """day_levels：每个有饮食记录的日子的记录程度（1–3）；adherence_weeks：签到回答了执行天数的周数。"""
    p = rulebook.get(DIET).params_as(DietParams)
    levels = list(day_levels)
    if any(level not in (1, 2, 3) for level in levels):
        raise ValueError("饮食记录程度只能是 1、2、3")
    result = Precision.P0
    if len(levels) >= p.min_logged_days:
        for precision, at_least in ((Precision.P3, 3), (Precision.P2, 2)):
            if sum(1 for level in levels if level >= at_least) / len(levels) >= p.share_required:
                result = precision
                break
    if result is Precision.P0 and adherence_weeks >= p.weeks_required:
        result = Precision.P1
    return PrecisionResult(
        precision=result,
        rule_id=DIET,
        basis={
            "logged_days": len(levels),
            "day_levels": levels,
            "adherence_weeks": adherence_weeks,
        },
    )


def recovery_precision_from_counts(
    *, rulebook: RuleBook, checkin_weeks: int, energy_days: int
) -> PrecisionResult:
    """设备接入前最高 P1：签到填疲劳或睡眠的周数够，或每日精力的天数够。"""
    p = rulebook.get(RECOVERY).params_as(RecoveryParams)
    enough = checkin_weeks >= p.weeks_required or energy_days >= p.min_energy_days
    return PrecisionResult(
        precision=Precision.P1 if enough else Precision.P0,
        rule_id=RECOVERY,
        basis={"checkin_weeks": checkin_weeks, "energy_days": energy_days},
    )


def _window_start(as_of: datetime, tz: tzinfo, window_days: int) -> date:
    return as_of.astimezone(tz).date() - timedelta(days=window_days - 1)


def body_precision(
    events: Iterable[WeightEvent], *, as_of: datetime, tz: tzinfo, rulebook: RuleBook
) -> PrecisionResult:
    """截止时间所在当地日期往前 window_days 天（含当天）内，按当地日期统计称重天数。"""
    p = rulebook.get(BODY).params_as(BodyParams)
    start = _window_start(as_of, tz, p.window_days)
    offsets = {
        (e.occurred_at.astimezone(tz).date() - start).days for e in resolve(events, as_of=as_of)
    }
    in_window = [d for d in offsets if 0 <= d < p.window_days]
    return body_precision_from_days(rulebook=rulebook, weigh_days=in_window)


def training_precision(
    events: Iterable[WorkoutCheckEvent | SetLogEvent],
    *,
    as_of: datetime,
    tz: tzinfo,
    rulebook: RuleBook,
) -> PrecisionResult:
    """按当地日期确定训练日及每日记录程度。"""
    p = rulebook.get(TRAINING).params_as(TrainingParams)
    start = _window_start(as_of, tz, p.window_days)
    by_day: dict[date, list[WorkoutCheckEvent | SetLogEvent]] = {}
    for e in resolve(events, as_of=as_of):
        if isinstance(e, WorkoutCheckEvent) and e.payload.status == "skipped":
            continue  # 只有“没练”的打卡不构成训练日
        day = e.occurred_at.astimezone(tz).date()
        if 0 <= (day - start).days < p.window_days:
            by_day.setdefault(day, []).append(e)
    levels = [_day_level(records) for _, records in sorted(by_day.items())]
    return training_precision_from_levels(rulebook=rulebook, day_levels=levels)


def _day_level(records: list[WorkoutCheckEvent | SetLogEvent]) -> int:
    sets = [r for r in records if isinstance(r, SetLogEvent)]
    if sets:
        with_effort = all(
            not isinstance(s.payload.rir, Unknown) or not isinstance(s.payload.rpe, Unknown)
            for s in sets
        )
        return 3 if with_effort else 2
    checks = [r for r in records if isinstance(r, WorkoutCheckEvent)]
    return 2 if any(not isinstance(c.payload.effort, Unknown) for c in checks) else 1


def _in_window[E: MealEvent | CheckinEvent | EnergyCheckEvent](
    events: Iterable[E], as_of: datetime, tz: tzinfo, window_days: int
) -> Iterator[tuple[int, E]]:
    """有效记录中落在窗口内的，连同它的日序号（0 起）。"""
    start = _window_start(as_of, tz, window_days)
    for e in resolve(events, as_of=as_of):
        offset = (e.occurred_at.astimezone(tz).date() - start).days
        if 0 <= offset < window_days:
            yield offset, e


def _diet_day_level(meals: list[MealEvent]) -> int:
    def has_kcal(m: MealEvent) -> bool:
        trusted = m.source is not Source.ESTIMATE or m.confirmed  # 未确认的估算视为没有热量
        return trusted and not isinstance(m.payload.kcal, Unknown)

    if all(
        has_kcal(m)
        and m.source is not Source.ESTIMATE
        and not isinstance(m.payload.protein_g, Unknown)
        for m in meals
    ):
        return 3
    return 2 if all(has_kcal(m) for m in meals) else 1


def diet_precision(
    events: Iterable[MealEvent | CheckinEvent], *, as_of: datetime, tz: tzinfo, rulebook: RuleBook
) -> PrecisionResult:
    p = rulebook.get(DIET).params_as(DietParams)
    by_day: dict[int, list[MealEvent]] = {}
    weeks: set[int] = set()
    for offset, e in _in_window(events, as_of, tz, p.window_days):
        if isinstance(e, MealEvent):
            by_day.setdefault(offset, []).append(e)
        elif isinstance(e, CheckinEvent) and not isinstance(e.payload.diet_adherence_days, Unknown):
            weeks.add(offset // 7)
    levels = [_diet_day_level(meals) for _, meals in sorted(by_day.items())]
    return diet_precision_from_days(
        rulebook=rulebook, day_levels=levels, adherence_weeks=len(weeks)
    )


def recovery_precision(
    events: Iterable[CheckinEvent | EnergyCheckEvent],
    *,
    as_of: datetime,
    tz: tzinfo,
    rulebook: RuleBook,
) -> PrecisionResult:
    p = rulebook.get(RECOVERY).params_as(RecoveryParams)
    weeks: set[int] = set()
    energy_days: set[int] = set()
    for offset, e in _in_window(events, as_of, tz, p.window_days):
        if isinstance(e, EnergyCheckEvent):
            energy_days.add(offset)
        elif isinstance(e, CheckinEvent) and not (
            isinstance(e.payload.fatigue, Unknown) and isinstance(e.payload.sleep_quality, Unknown)
        ):
            weeks.add(offset // 7)
    return recovery_precision_from_counts(
        rulebook=rulebook, checkin_weeks=len(weeks), energy_days=len(energy_days)
    )
