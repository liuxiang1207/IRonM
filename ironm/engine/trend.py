"""体重趋势：最小二乘线性回归、斜率标准误、按 t 分布的可信范围。参数取自规则表。"""

import math
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.events import WeightEvent
from ironm.contracts.metrics import Need
from ironm.contracts.unknown import Unknown, UnknownReason
from ironm.engine.data_quality import effective_weights, pending_weights
from ironm.engine.stats import t_quantile
from ironm.rules.loader import RuleBook

WEIGHT_TREND = "R-MT-WEIGHT-TREND"
_INSUFFICIENT = Unknown(unknown=UnknownReason.INSUFFICIENT_DATA)


class TrendParams(BaseModel):
    ci_level: Annotated[float, Field(gt=0, lt=1)]
    min_readings: Annotated[int, Field(ge=3)]  # 少于 3 次无法估计残差
    min_span_days: Annotated[float, Field(ge=0)]


class TrendFit(BaseModel):
    """回归结果。数据不足时数值字段为“未知（数据不足）”，needed 说明还差什么。"""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "insufficient"]
    needed: tuple[Need, ...]
    n: int
    span_days: float
    ci_level: float
    slope_kg_per_week: float | Unknown
    intercept_kg: float | Unknown  # 第一次读数时刻的拟合体重
    se_kg_per_week: float | Unknown
    ci_low: float | Unknown
    ci_high: float | Unknown
    residual_sd_kg: float | Unknown


def fit_trend(
    points: Sequence[tuple[float, float]],
    *,
    ci_level: float,
    min_readings: int,
    min_span_days: float,
) -> TrendFit:
    """points：(天数, 千克)。天数的起点无关紧要，截距取第一次读数时刻的拟合值。"""
    n = len(points)
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    span = max(xs) - min(xs) if xs else 0.0
    needed: list[Need] = []
    if n < min_readings:
        needed.append(Need(item="weight_record", count=min_readings - n))
    if span < min_span_days:
        needed.append(Need(item="span_days", count=max(1, math.ceil(min_span_days - span))))
    if needed:
        return TrendFit(
            status="insufficient",
            needed=tuple(needed),
            n=n,
            span_days=span,
            ci_level=ci_level,
            slope_kg_per_week=_INSUFFICIENT,
            intercept_kg=_INSUFFICIENT,
            se_kg_per_week=_INSUFFICIENT,
            ci_low=_INSUFFICIENT,
            ci_high=_INSUFFICIENT,
            residual_sd_kg=_INSUFFICIENT,
        )
    x_mean = sum(xs) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    y_mean = sum(ys) / n
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = y_mean - slope * x_mean
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    residual_var = sse / (n - 2)
    se = math.sqrt(residual_var / sxx)
    half = t_quantile(0.5 + ci_level / 2, n - 2) * se
    week = 7.0
    return TrendFit(
        status="ok",
        needed=(),
        n=n,
        span_days=span,
        ci_level=ci_level,
        slope_kg_per_week=slope * week,
        intercept_kg=intercept + slope * min(xs),
        se_kg_per_week=se * week,
        ci_low=(slope - half) * week,
        ci_high=(slope + half) * week,
        residual_sd_kg=math.sqrt(residual_var),
    )


def weight_trend(
    events: Iterable[WeightEvent], *, as_of: datetime, window_days: int, rulebook: RuleBook
) -> TrendFit:
    """窗口 [as_of - window_days, as_of] 内的趋势。

    只用有效、非待确认、与最近一次读数同一数据流的读数（不同的秤不混算）。
    """
    p = rulebook.get(WEIGHT_TREND).params_as(TrendParams)
    items = effective_weights(events, as_of=as_of)
    pending = pending_weights(items, rulebook)
    eligible = [e for e in items if e.event_id not in pending]
    start = as_of - timedelta(days=window_days)
    in_window = [e for e in eligible if start <= e.occurred_at <= as_of]
    if in_window:
        stream = in_window[-1].stream_key  # effective_weights 已按时间排序
        in_window = [e for e in in_window if e.stream_key == stream]
    points = [
        ((e.occurred_at - start).total_seconds() / 86400, e.payload.weight_kg) for e in in_window
    ]
    return fit_trend(
        points, ci_level=p.ci_level, min_readings=p.min_readings, min_span_days=p.min_span_days
    )
