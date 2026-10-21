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
    # 截至周五晚：1–3 天前为周四至周二，6 天前为周六（不含今天，spec §20 澄清）
    meals = [_meal(d, 1800 + 10 * d) for d in (1, 2, 3, 6)]
    meals += [_meal(5, 1500), _meal(5, None)]  # 有一餐没记热量 → 该天不计
    m = _metric(build_metrics(meals, as_of=AS_OF, tz=SH, rulebook=BOOK), "diet_kcal_mean_7d")
    assert m.status is MetricStatus.OK
    assert m.value == (1810 + 1820 + 1830 + 1860) / 4
    assert m.n_records == 4 and m.window_days == 7


def test_diet_needs_a_weekend_day() -> None:
    """已知 4 天都是工作日 → 数据不足，还差 1 个周末日（spec §26）。"""
    meals = [_meal(d, 1800) for d in range(1, 5)]  # 周四至周一
    m = _metric(build_metrics(meals, as_of=AS_OF, tz=SH, rulebook=BOOK), "diet_kcal_mean_7d")
    assert m.status is MetricStatus.INSUFFICIENT
    assert [(n.item, n.count) for n in m.needed] == [("weekend_diet_days", 1)]
    assert m.value == 1800  # 数值仍给出（暂定）


def test_diet_insufficient() -> None:
    m = _metric(
        build_metrics([_meal(d, 1800) for d in range(1, 4)], as_of=AS_OF, tz=SH, rulebook=BOOK),
        "diet_kcal_mean_7d",
    )
    assert m.status is MetricStatus.INSUFFICIENT
    assert [(n.item, n.count) for n in m.needed] == [("diet_days", 1), ("weekend_diet_days", 1)]


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
    meals = [_meal(d, 1800) for d in range(1, 5)] + [_meal(5, 450)]  # 450 千卡 → 待确认
    m = _metric(build_metrics(meals, as_of=AS_OF, tz=SH, rulebook=BOOK), "diet_kcal_mean_7d")
    assert m.status is MetricStatus.PENDING_CONFIRM
    assert m.value == 1800


# ---- 冲刺 1 末审查后补充 ----


def test_status_does_not_flip_once_ok() -> None:
    rng = random.Random(7)
    start = datetime(2026, 9, 1, 7, 30, tzinfo=SH)
    weights = []
    for d in range(60):
        at = (start + timedelta(days=d)).isoformat()
        weights.append(
            EVENT_ADAPTER.validate_json(
                json.dumps(
                    {
                        "event_id": f"00000000-0000-4000-9000-{d:012d}",
                        "user_id": "u1",
                        "type": "weight",
                        "occurred_at": at,
                        "recorded_at": at,
                        "source": "user",
                        "confirmed": True,
                        "payload": {"weight_kg": round(85 - 0.07 * d + rng.gauss(0, 0.45), 1)},
                    }
                )
            )
        )
    statuses = []
    for d in range(25, 60):
        as_of = (datetime(2026, 9, 1, 23, 0, tzinfo=SH) + timedelta(days=d)).astimezone(UTC)
        report = build_metrics(
            [e for e in weights if e.occurred_at <= as_of], as_of=as_of, tz=SH, rulebook=BOOK
        )
        statuses.append(_metric(report, "weight_trend").status)
    first_ok = statuses.index(MetricStatus.OK)
    assert all(s is MetricStatus.OK for s in statuses[first_ok:]), statuses


def test_today_excluded_from_diet_average() -> None:
    meals = [_meal(d, 2000) for d in range(1, 7)]
    meals.append(_ev("meal", 0, {"slot": "breakfast", "kcal": 300}, hour=8))
    noon = datetime(2026, 10, 30, 12, 0, tzinfo=SH).astimezone(UTC)
    m = _metric(build_metrics(meals, as_of=noon, tz=SH, rulebook=BOOK), "diet_kcal_mean_7d")
    assert m.status is MetricStatus.OK and m.value == 2000


def test_unconfirmed_estimate_not_counted_as_kcal() -> None:
    meals = [
        _ev("meal", d, {"slot": "lunch", "kcal": 2500}, hour=12, source="estimate", confirmed=False)
        for d in range(1, 8)
    ]
    m = _metric(build_metrics(meals, as_of=AS_OF, tz=SH, rulebook=BOOK), "diet_kcal_mean_7d")
    assert m.status is MetricStatus.INSUFFICIENT


def test_stale_with_only_older_history() -> None:
    old = [_ev("weight", 70 + d, {"weight_kg": 80.0}) for d in range(3)]
    report = build_metrics([], as_of=AS_OF, tz=SH, rulebook=BOOK, earlier_weights=old)
    m = _metric(report, "weight_trend")
    assert m.status is MetricStatus.STALE
    assert m.last_record_at == old[0].occurred_at


# ---- 审定后修订（spec §26） ----


def _weights(kgs: dict[int, float]) -> list[Any]:
    return [_ev("weight", ago, {"weight_kg": kg}) for ago, kg in kgs.items()]


def noise(d: int) -> float:
    return 0.3 if d % 2 else -0.3


def test_trend_window_follows_recent_change() -> None:
    """前 25 天每周掉约 1 千克、近 25 天持平：取达标的最短窗口，结论反映近期的持平。"""
    kgs = {ago: 75.0 + noise(ago) for ago in range(0, 25)}
    kgs |= {ago: 75.0 + 0.15 * (ago - 24) + noise(ago) for ago in range(25, 50)}
    m = _metric(build_metrics(_weights(kgs), as_of=AS_OF, tz=SH, rulebook=BOOK), "weight_trend")
    assert m.status is MetricStatus.OK
    assert m.window_days <= 28
    assert isinstance(m.value, float) and abs(m.value) < 0.25


def test_old_unconfirmed_reading_no_longer_blocks_status() -> None:
    """超过 7 天仍未确认的读数仍不参与计算，但不再让趋势停在“有待确认”。"""
    base = {ago: round(80 - 0.05 * (40 - ago) + (0.2 if ago % 2 else -0.2), 1) for ago in range(40)}
    old = dict(base)
    old[10] = base[10] + 5  # 10 天前的尖峰
    m = _metric(build_metrics(_weights(old), as_of=AS_OF, tz=SH, rulebook=BOOK), "weight_trend")
    assert m.status is MetricStatus.OK
    recent = dict(base)
    recent[3] = base[3] + 5  # 3 天前的尖峰
    r = _metric(build_metrics(_weights(recent), as_of=AS_OF, tz=SH, rulebook=BOOK), "weight_trend")
    assert r.status is MetricStatus.PENDING_CONFIRM
