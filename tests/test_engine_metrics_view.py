"""指标组装（spec §20）。"""

import json
import random
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from ironm.contracts.common import Domain
from ironm.contracts.events import EVENT_ADAPTER
from ironm.contracts.metrics import MetricResult, MetricStatus
from ironm.engine.metrics_view import MetricsReport, build_metrics, lookback_days
from ironm.rules.loader import RULES_DIR, load_rules

BOOK = load_rules(RULES_DIR)
SH = ZoneInfo("Asia/Shanghai")
AS_OF = datetime(2026, 10, 30, 15, 0, tzinfo=UTC)  # 上海时间 10 月 30 日 23:00
_n = 0


def _ev(type_: str, days_ago: float, payload: dict[str, Any], hour: int = 7, **extra: Any) -> Any:
    global _n
    _n += 1
    local = datetime(2026, 10, 30, hour, 30, tzinfo=SH) - timedelta(days=days_ago)
    data: dict[str, Any] = {
        "event_id": f"00000000-0000-4000-8000-{_n:012d}",
        "user_id": "u1",
        "type": type_,
        "occurred_at": local.isoformat(),
        "recorded_at": local.isoformat(),
        "source": "user",
        "confirmed": True,
        "payload": payload,
        **extra,
    }
    return EVENT_ADAPTER.validate_json(json.dumps(data))


def _daily_weights(days: int, start_ago: int = 0) -> list[Any]:
    rng = random.Random(3)
    return [
        _ev(
            "weight",
            start_ago + d,
            {"weight_kg": round(82 - 0.07 * (days - d) + rng.gauss(0, 0.3), 1)},
        )
        for d in range(days)
    ]


def _metric(report: MetricsReport, name: str) -> MetricResult:
    [m] = [m for m in report.metrics if m.name == name]
    # 能通过指标契约校验（往返）
    assert MetricResult.model_validate_json(m.model_dump_json()) == m
    return m


def test_weight_trend_ok() -> None:
    report = build_metrics(_daily_weights(30), as_of=AS_OF, tz=SH, rulebook=BOOK)
    m = _metric(report, "weight_trend")
    assert m.status is MetricStatus.OK
    assert isinstance(m.value, float) and m.ci is not None
    assert m.domain is Domain.BODY and m.unit == "kg/week"
    assert m.precision == report.precision[Domain.BODY] == "P3"
    assert m.formula_id == "F-WEIGHT-TREND-OLS-1"
    assert m.n_records >= 10 and m.window_days > 0


def test_weight_trend_stale() -> None:
    report = build_metrics(_daily_weights(20, start_ago=12), as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert _metric(report, "weight_trend").status is MetricStatus.STALE


def test_weight_trend_pending() -> None:
    events = [*_daily_weights(30), _ev("weight", 0.5, {"weight_kg": 90.0}, hour=20)]
    report = build_metrics(events, as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert _metric(report, "weight_trend").status is MetricStatus.PENDING_CONFIRM


def test_weight_trend_insufficient_with_provisional_value() -> None:
    report = build_metrics(_daily_weights(10), as_of=AS_OF, tz=SH, rulebook=BOOK)
    m = _metric(report, "weight_trend")
    assert m.status is MetricStatus.INSUFFICIENT
    assert m.needed and isinstance(m.value, float)  # 暂定数值
    assert m.ci is not None  # 暂定的可信范围也给出


def test_weight_trend_too_few() -> None:
    events = [_ev("weight", 0, {"weight_kg": 80.0}), _ev("weight", 3, {"weight_kg": 80.2})]
    m = _metric(build_metrics(events, as_of=AS_OF, tz=SH, rulebook=BOOK), "weight_trend")
    assert m.status is MetricStatus.INSUFFICIENT
    assert m.needed


def _meal(days_ago: int, kcal: float | None) -> Any:
    payload: dict[str, Any] = {"slot": "lunch"}
    if kcal is not None:
        payload["kcal"] = kcal
    return _ev("meal", days_ago, payload, hour=12)


def test_diet_average_known_days_only() -> None:
    meals = [_meal(d, 1800 + 10 * d) for d in range(4)]
    meals += [_meal(4, 1500), _meal(4, None)]  # 第 5 天有一餐没记热量 → 该天不计
    m = _metric(build_metrics(meals, as_of=AS_OF, tz=SH, rulebook=BOOK), "diet_kcal_mean_7d")
    assert m.status is MetricStatus.OK
    assert m.value == (1800 + 1810 + 1820 + 1830) / 4
    assert m.n_records == 4 and m.window_days == 7


def test_diet_insufficient() -> None:
    m = _metric(
        build_metrics([_meal(d, 1800) for d in range(3)], as_of=AS_OF, tz=SH, rulebook=BOOK),
        "diet_kcal_mean_7d",
    )
    assert m.status is MetricStatus.INSUFFICIENT
    assert [(n.item, n.count) for n in m.needed] == [("diet_days", 1)]


def test_empty_user() -> None:
    report = build_metrics([], as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert set(report.precision.values()) == {"P0"}
    assert set(report.precision) == set(Domain)
    assert _metric(report, "weight_trend").status is MetricStatus.INSUFFICIENT


def test_lookback_derived_and_old_history_irrelevant() -> None:
    days = lookback_days(BOOK)
    assert days >= 56 + 7
    recent = _daily_weights(30)
    ancient = [_ev("weight", days + 1 + i, {"weight_kg": 95.0}) for i in range(5000)]
    in_range = [e for e in [*recent, *ancient] if AS_OF - e.occurred_at <= timedelta(days=days)]
    start = time.perf_counter()
    a = build_metrics(in_range, as_of=AS_OF, tz=SH, rulebook=BOOK)
    elapsed = time.perf_counter() - start
    assert a == build_metrics(recent, as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert elapsed < 1.0


def test_diet_pending_day_excluded_from_average() -> None:
    meals = [_meal(d, 1800) for d in range(4)] + [_meal(4, 450)]  # 450 千卡 → 待确认
    m = _metric(build_metrics(meals, as_of=AS_OF, tz=SH, rulebook=BOOK), "diet_kcal_mean_7d")
    assert m.status is MetricStatus.PENDING_CONFIRM
    assert m.value == 1800
