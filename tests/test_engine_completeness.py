"""完成度、新鲜度、“未知”不计入平均、单日热量（spec [2026-10-20] §15）。"""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ironm.contracts.events import EVENT_ADAPTER, MealEvent, WeightEvent, WorkoutCheckEvent
from ironm.contracts.unknown import Unknown, UnknownReason
from ironm.engine.completeness import training_completion, weight_freshness
from ironm.engine.data_quality import daily_kcal, pending_kcal_days
from ironm.engine.stats import mean_known
from ironm.rules.loader import RULES_DIR, RuleBook, load_rules

BOOK = load_rules(RULES_DIR)
SH = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
U = Unknown(unknown=UnknownReason.NOT_RECORDED)
_n = 0


def _ev(type_: str, at: str, payload: dict[str, Any], **extra: Any) -> Any:
    global _n
    _n += 1
    data: dict[str, Any] = {
        "event_id": f"00000000-0000-4000-8000-{_n:012d}",
        "user_id": "u1",
        "type": type_,
        "occurred_at": at,
        "recorded_at": at,
        "source": "user",
        "confirmed": True,
        "payload": payload,
        **extra,
    }
    return EVENT_ADAPTER.validate_json(json.dumps(data))


# ---- “未知”不计入平均 ----

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)


@given(st.lists(finite, min_size=1, max_size=20), st.lists(st.integers(0, 40), max_size=10))
def test_inserting_unknown_does_not_change_mean(values: list[float], positions: list[int]) -> None:
    mixed: list[float | Unknown] = list(values)
    for pos in positions:
        mixed.insert(pos % (len(mixed) + 1), U)
    assert mean_known(mixed) == pytest.approx(mean_known(values))


@given(st.integers(0, 10))
def test_all_unknown_is_unknown(n: int) -> None:
    result = mean_known([U] * n)
    assert isinstance(result, Unknown)
    assert result.unknown is UnknownReason.INSUFFICIENT_DATA


# ---- 新鲜度 ----


def _w(at: datetime, kg: float, **extra: Any) -> WeightEvent:
    e = _ev("weight", at.isoformat(), {"weight_kg": kg}, **extra)
    assert isinstance(e, WeightEvent)
    return e


@pytest.mark.parametrize(
    ("age", "state"),
    [(timedelta(days=10), "ok"), (timedelta(days=10, minutes=1), "stale")],
)
def test_freshness_boundary(age: timedelta, state: str) -> None:
    f = weight_freshness([_w(NOW - age, 80.0)], as_of=NOW, rulebook=BOOK)
    assert f.status == state
    assert f.rule_id == "R-DQ-WEIGHT-STALE"
    assert f.last_record_at == NOW - age


def test_freshness_no_data() -> None:
    f = weight_freshness([], as_of=NOW, rulebook=BOOK)
    assert f.status == "insufficient"
    assert [(n.item, n.count) for n in f.needed] == [("weight_record", 1)]
    assert isinstance(f.last_record_at, Unknown)


def test_freshness_ignores_untrusted_readings() -> None:
    old = _w(NOW - timedelta(days=15), 80.0)
    base = _w(NOW - timedelta(days=11), 80.2)
    spike = _w(NOW - timedelta(days=4), 84.0)  # 与 7 天前的 base 比跳了 3.8 千克 → 待确认
    wrong = _w(NOW - timedelta(days=1), 60.0)
    fix = _w(NOW - timedelta(days=13), 80.1, corrects=str(wrong.event_id))  # 更正取代了 wrong
    f = weight_freshness([old, base, spike, wrong, fix], as_of=NOW, rulebook=BOOK)
    assert f.status == "stale"
    assert f.last_record_at == base.occurred_at


# ---- 训练完成度 ----

D1, D2, D3 = date(2026, 10, 19), date(2026, 10, 20), date(2026, 10, 21)


def _wo(at: str, status: str) -> WorkoutCheckEvent:
    e = _ev("workout_check", at, {"status": status})
    assert isinstance(e, WorkoutCheckEvent)
    return e


def test_completion_counts_known_days_only() -> None:
    events = [_wo("2026-10-19T20:00:00+08:00", "done"), _wo("2026-10-20T20:00:00+08:00", "half")]
    c = training_completion(events, planned_days=[D1, D2, D3], tz=SH, rulebook=BOOK)
    assert c.rate == pytest.approx(0.75)
    assert (c.recorded_days, c.planned_days) == (2, 3)
    assert c.coverage == pytest.approx(2 / 3)


def test_completion_all_missing_is_unknown() -> None:
    c = training_completion([], planned_days=[D1, D2], tz=SH, rulebook=BOOK)
    assert isinstance(c.rate, Unknown)
    assert c.coverage == 0


def test_completion_ignores_unplanned_days_and_uses_local_date() -> None:
    events = [
        _wo("2026-10-22T20:00:00+08:00", "done"),  # 非计划日
        _wo("2026-10-21T00:30:00+08:00", "skipped"),  # 上海时间 21 日 0:30，属于 21 日
    ]
    c = training_completion(events, planned_days=[D3], tz=SH, rulebook=BOOK)
    assert c.rate == 0.0
    assert c.recorded_days == 1


def test_completion_scores_from_rule_table(tmp_path: Path) -> None:
    d = tmp_path / "rules"
    d.mkdir()
    for f in RULES_DIR.glob("*.yaml"):
        (d / f.name).write_text(
            f.read_text(encoding="utf-8").replace("score_half: 0.5", "score_half: 0.4"),
            encoding="utf-8",
        )
    book: RuleBook = load_rules(d)
    events = [_wo("2026-10-20T20:00:00+08:00", "half")]
    c = training_completion(events, planned_days=[D2], tz=SH, rulebook=book)
    assert c.rate == pytest.approx(0.4)


# ---- 单日热量 ----


def _meal(at: str, slot: str, kcal: float | None) -> MealEvent:
    payload: dict[str, Any] = {"slot": slot}
    if kcal is not None:
        payload["kcal"] = kcal
    e = _ev("meal", at, payload)
    assert isinstance(e, MealEvent)
    return e


def test_daily_kcal() -> None:
    meals = [
        _meal("2026-10-19T12:30:00+08:00", "lunch", 650),
        _meal("2026-10-19T19:00:00+08:00", "dinner", 700),
        _meal("2026-10-20T12:30:00+08:00", "lunch", 600),
        _meal("2026-10-20T19:00:00+08:00", "dinner", None),  # 没记热量
    ]
    totals = daily_kcal(meals, days=[D1, D2, D3], tz=SH)
    assert totals[D1] == 1350
    assert isinstance(totals[D2], Unknown)
    assert isinstance(totals[D3], Unknown)  # 没有饮食记录


def test_pending_kcal_days_only_known_totals() -> None:
    totals: dict[date, float | Unknown] = {D1: 450.0, D2: U, D3: 1800.0}
    assert pending_kcal_days(totals, BOOK) == {D1: "R-DQ-KCAL-RANGE"}


def test_coverage_unknown_without_planned_days() -> None:
    c = training_completion([], planned_days=[], tz=SH, rulebook=BOOK)
    assert isinstance(c.coverage, Unknown)


def test_completion_insufficient_below_recorded_share() -> None:
    """有记录的计划日占比低于 75% → 数据不足，仍给出已记录日的完成度（spec §26）。"""
    two_of_three = training_completion(
        [_wo("2026-10-19T20:00:00+08:00", "done"), _wo("2026-10-20T20:00:00+08:00", "done")],
        planned_days=[D1, D2, D3],
        tz=SH,
        rulebook=BOOK,
    )
    assert two_of_three.status == "insufficient" and two_of_three.rate == 1.0
    all_three = training_completion(
        [
            _wo("2026-10-19T20:00:00+08:00", "done"),
            _wo("2026-10-20T20:00:00+08:00", "done"),
            _wo("2026-10-21T20:00:00+08:00", "half"),
        ],
        planned_days=[D1, D2, D3],
        tz=SH,
        rulebook=BOOK,
    )
    assert all_three.status == "ok"
    assert training_completion([], planned_days=[], tz=SH, rulebook=BOOK).status == "insufficient"
