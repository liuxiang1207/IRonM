"""指标组装（spec §20）：把趋势、精度、判断窗口、新鲜度、日均热量组装成指标契约。

只需要最近一段记录：回看天数由规则表中的各窗口推算（lookback_days），调用方据此取数。
"""

import math
from collections.abc import Iterable
from datetime import datetime, timedelta, tzinfo
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.common import Domain, Precision
from ironm.contracts.events import (
    CheckinEvent,
    EnergyCheckEvent,
    Event,
    MealEvent,
    SetLogEvent,
    WeightEvent,
    WorkoutCheckEvent,
)
from ironm.contracts.metrics import ConfidenceInterval, MetricResult, MetricStatus, Need
from ironm.contracts.unknown import Unknown, not_recorded
from ironm.engine.completeness import weight_freshness
from ironm.engine.data_quality import (
    WEIGHT_JUMP,
    WeightJumpParams,
    daily_kcal,
    effective_weights,
    pending_kcal_days,
    pending_weights,
)
from ironm.engine.effective import resolve
from ironm.engine.precision import (
    BODY,
    DIET,
    RECOVERY,
    TRAINING,
    BodyParams,
    DietParams,
    RecoveryParams,
    TrainingParams,
    body_precision,
    diet_precision,
    recovery_precision,
    training_precision,
)
from ironm.engine.stats import mean_known
from ironm.engine.trend import weight_trend
from ironm.engine.window import (
    JUDGEMENT_WINDOW,
    WindowParams,
    extra_days_for_target,
    judgement_window,
)
from ironm.rules.loader import RuleBook

DIET_MIN_DATA = "R-DQ-DIET-ADJ-MIN-DATA"


class DietMinDataParams(BaseModel):
    window_days: Annotated[int, Field(gt=0)]
    min_diet_days: Annotated[int, Field(ge=1)]
    min_weigh_ins: Annotated[int, Field(ge=0)]


class MetricsReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    computed_at: datetime
    precision: dict[Domain, Precision]
    metrics: tuple[MetricResult, ...]


def lookback_days(rulebook: RuleBook) -> int:
    """计算所需的最长回看天数：各窗口的最大值，加上“待确认”判定往前比较的天数。"""
    windows = [
        rulebook.get(JUDGEMENT_WINDOW).params_as(WindowParams).max_window_days,
        rulebook.get(BODY).params_as(BodyParams).window_days,
        rulebook.get(TRAINING).params_as(TrainingParams).window_days,
        rulebook.get(DIET).params_as(DietParams).window_days,
        rulebook.get(RECOVERY).params_as(RecoveryParams).window_days,
        rulebook.get(DIET_MIN_DATA).params_as(DietMinDataParams).window_days,
    ]
    return max(windows) + rulebook.get(WEIGHT_JUMP).params_as(WeightJumpParams).window_days


def build_metrics(
    events: Iterable[Event],
    *,
    as_of: datetime,
    tz: tzinfo,
    rulebook: RuleBook,
    earlier_weights: Iterable[WeightEvent] = (),
) -> MetricsReport:
    """earlier_weights：回看范围之外最近的几次称重，只用于新鲜度（区分“已过期”与“没有数据”）。"""
    items = list(events)
    weights = [e for e in items if isinstance(e, WeightEvent)]
    meals = [e for e in items if isinstance(e, MealEvent)]
    checkins = [e for e in items if isinstance(e, CheckinEvent)]
    energy = [e for e in items if isinstance(e, EnergyCheckEvent)]
    training = [e for e in items if isinstance(e, WorkoutCheckEvent | SetLogEvent)]
    precision = {
        Domain.BODY: body_precision(weights, as_of=as_of, tz=tz, rulebook=rulebook).precision,
        Domain.TRAINING: training_precision(
            training, as_of=as_of, tz=tz, rulebook=rulebook
        ).precision,
        Domain.DIET: diet_precision(
            [*meals, *checkins], as_of=as_of, tz=tz, rulebook=rulebook
        ).precision,
        Domain.RECOVERY: recovery_precision(
            [*checkins, *energy], as_of=as_of, tz=tz, rulebook=rulebook
        ).precision,
    }
    metrics = (
        _weight_trend_metric(
            weights, list(earlier_weights), as_of, tz, rulebook, precision[Domain.BODY]
        ),
        _diet_kcal_metric(meals, as_of, tz, rulebook, precision[Domain.DIET]),
    )
    return MetricsReport(computed_at=as_of, precision=precision, metrics=metrics)


def _weight_trend_metric(
    weights: list[WeightEvent],
    earlier: list[WeightEvent],
    as_of: datetime,
    tz: tzinfo,
    rulebook: RuleBook,
    precision: Precision,
) -> MetricResult:
    window = judgement_window(weights, as_of=as_of, tz=tz, rulebook=rulebook)
    max_days = rulebook.get(JUDGEMENT_WINDOW).params_as(WindowParams).max_window_days
    required = window.required_days if isinstance(window.required_days, int) else max_days
    # 已有数据超过所需天数时窗口随覆盖增长（不超过上限）：“还差 N 天”才能兑现，状态不会来回跳
    days = min(max_days, max(required, window.covered_days))
    fit = weight_trend(weights, as_of=as_of, window_days=days, rulebook=rulebook)
    fresh = weight_freshness([*earlier, *weights], as_of=as_of, rulebook=rulebook)
    start = as_of - timedelta(days=days)
    effective = effective_weights(weights, as_of=as_of)
    pending = pending_weights(effective, rulebook)
    pending_in_window = any(e.event_id in pending and e.occurred_at >= start for e in effective)

    needed: tuple[Need, ...] = fit.needed or window.needed
    # “正常”以实际可信范围为准：模型按均匀称重估算的天数可能不够
    target = rulebook.get(JUDGEMENT_WINDOW).params_as(WindowParams).target_half_width_kg_per_week
    too_wide = False
    if isinstance(fit.ci_low, float) and isinstance(fit.ci_high, float):
        half = (fit.ci_high - fit.ci_low) / 2
        if half > target and window.status == "ok":
            too_wide = True
            if days < max_days:
                needed = (Need(item="days", count=extra_days_for_target(days, half, target)),)
            elif window.readings_per_week < 7:
                count = min(7, math.floor(window.readings_per_week) + 1)
                needed = (Need(item="weigh_ins_per_week", count=count),)
            else:  # 窗口已到上限且每天都称：再多称也没用
                needed = (Need(item="consistent_weighing_time", count=1),)
    if fresh.status == "stale":
        status = MetricStatus.STALE
    elif pending_in_window:
        status = MetricStatus.PENDING_CONFIRM
    elif fit.status == "insufficient" or window.status == "insufficient" or too_wide:
        status = MetricStatus.INSUFFICIENT
    else:
        status = MetricStatus.OK
    if status is MetricStatus.INSUFFICIENT and not needed:
        needed = fresh.needed
    ci = None
    if isinstance(fit.ci_low, float) and isinstance(fit.ci_high, float):
        ci = ConfidenceInterval(low=fit.ci_low, high=fit.ci_high, level=fit.ci_level)
    return MetricResult(
        name="weight_trend",
        domain=Domain.BODY,
        value=fit.slope_kg_per_week,
        unit="kg/week",
        ci=ci,
        precision=precision,
        n_records=fit.n,
        window_days=days,
        status=status,
        needed=needed,
        last_record_at=fresh.last_record_at,
        computed_at=as_of,
        formula_id="F-WEIGHT-TREND-OLS-1",
    )


def _diet_kcal_metric(
    meals: list[MealEvent], as_of: datetime, tz: tzinfo, rulebook: RuleBook, precision: Precision
) -> MetricResult:
    p = rulebook.get(DIET_MIN_DATA).params_as(DietMinDataParams)
    today = as_of.astimezone(tz).date()
    # 不含今天：今天尚未结束，部分记录会拉低均值并误报异常
    days = [today - timedelta(days=i) for i in range(1, p.window_days + 1)]
    totals = daily_kcal(meals, days=days, tz=tz, as_of=as_of)
    pending = pending_kcal_days(totals, rulebook)
    trusted = {d: t for d, t in totals.items() if d not in pending}
    known = sum(1 for t in trusted.values() if not isinstance(t, Unknown))
    value = mean_known(trusted.values())
    if pending:
        status = MetricStatus.PENDING_CONFIRM
    elif known < p.min_diet_days:
        status = MetricStatus.INSUFFICIENT
    else:
        status = MetricStatus.OK
    needed = (
        (Need(item="diet_days", count=p.min_diet_days - known),)
        if status is MetricStatus.INSUFFICIENT
        else ()
    )
    recent = resolve(meals, as_of=as_of)
    last: datetime | Unknown = recent[-1].occurred_at if recent else not_recorded()
    return MetricResult(
        name="diet_kcal_mean_7d",
        domain=Domain.DIET,
        value=value,
        unit="kcal/day",
        precision=precision,
        n_records=known,
        window_days=p.window_days,
        status=status,
        needed=needed,
        last_record_at=last,
        computed_at=as_of,
        formula_id="F-DIET-KCAL-MEAN-1",
    )
