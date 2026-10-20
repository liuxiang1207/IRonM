"""全局规则清单（spec §16）：每个读取记录的计算功能都处理更正、未知、时区、截止时间、顺序无关。

新增计算功能时必须在 REGISTERED 中登记，并在下方补对应用例；未登记即测试失败。
"""

import inspect
import json
import pkgutil
import random
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest

import ironm.engine
from ironm.contracts.events import EVENT_ADAPTER
from ironm.engine.completeness import training_completion, weight_freshness
from ironm.engine.data_quality import daily_kcal, pending_weights
from ironm.engine.effective import CorrectionForkError, resolve
from ironm.engine.energy import energy_today
from ironm.engine.precision import (
    body_precision,
    diet_precision,
    recovery_precision,
    training_precision,
)
from ironm.engine.trend import weight_trend
from ironm.rules.loader import RULES_DIR, load_rules

BOOK = load_rules(RULES_DIR)
SH = ZoneInfo("Asia/Shanghai")
DAY = date(2026, 10, 20)
_n = 0

# 读取记录做计算的公开函数（参数名为 events 或 meals）
REGISTERED = {
    "resolve",
    "effective_weights",
    "pending_weights",
    "daily_kcal",
    "training_completion",
    "weight_freshness",
    "weight_trend",
    "energy_today",
    "body_precision",
    "training_precision",
    "diet_precision",
    "recovery_precision",
}


def test_every_calculator_registered() -> None:
    found = set()
    for mod in pkgutil.iter_modules(ironm.engine.__path__, prefix="ironm.engine."):
        module = __import__(mod.name, fromlist=["_"])
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn.__module__ != mod.name or name.startswith("_"):
                continue
            params = inspect.signature(fn).parameters
            if "events" in params or "meals" in params:
                found.add(name)
    assert found == REGISTERED


def _ev(type_: str, at: str, payload: dict[str, Any], **extra: Any) -> Any:
    global _n
    _n += 1
    data: dict[str, Any] = {
        "event_id": f"00000000-0000-4000-8000-{_n:012d}",
        "user_id": "u1",
        "type": type_,
        "occurred_at": at,
        "recorded_at": extra.pop("recorded_at", at),
        "source": "user",
        "confirmed": True,
        "payload": payload,
        **extra,
    }
    return EVENT_ADAPTER.validate_json(json.dumps(data))


def _fix(original: Any, payload: dict[str, Any], at: str | None = None) -> Any:
    when = at or original.occurred_at.isoformat()
    return _ev(original.type, when, payload, corrects=str(original.event_id))


# ---- 有效记录层 ----


def test_resolve_chain_and_fork() -> None:
    a = _ev("meal", "2026-10-20T12:00:00+08:00", {"slot": "lunch", "kcal": 800})
    b = _fix(a, {"slot": "lunch", "kcal": 700})
    c = _fix(b, {"slot": "lunch", "kcal": 600})
    assert [e.event_id for e in resolve([c, a, b])] == [c.event_id]
    fork = _fix(a, {"slot": "lunch", "kcal": 500})
    with pytest.raises(CorrectionForkError):
        resolve([a, b, fork])


def test_resolve_correction_inherits_stream() -> None:
    w = _ev(
        "weight",
        "2026-10-20T07:00:00+08:00",
        {"weight_kg": 80.0},
        source="device",
        stream_key="scale:home",
    )
    confirm = _fix(w, {"weight_kg": 80.0})
    [effective] = resolve([w, confirm])
    assert effective.event_id == confirm.event_id
    assert effective.stream_key == "scale:home"


def test_resolve_as_of_excludes_later_records_and_corrections() -> None:
    a = _ev("meal", "2026-10-20T12:00:00+08:00", {"slot": "lunch", "kcal": 800})
    late_fix = _ev(
        "meal",
        "2026-10-20T12:00:00+08:00",
        {"slot": "lunch", "kcal": 600},
        corrects=str(a.event_id),
        recorded_at="2026-10-22T09:00:00+08:00",
    )
    as_of = datetime(2026, 10, 21, tzinfo=UTC)
    assert [e.event_id for e in resolve([a, late_fix], as_of=as_of)] == [a.event_id]


# ---- 各计算功能：更正 ----


def test_daily_kcal_uses_correction() -> None:
    a = _ev("meal", "2026-10-20T12:30:00+08:00", {"slot": "lunch", "kcal": 800})
    assert daily_kcal([a, _fix(a, {"slot": "lunch", "kcal": 600})], days=[DAY], tz=SH)[DAY] == 600


def test_completion_uses_correction_and_moved_day() -> None:
    a = _ev("workout_check", "2026-10-20T20:00:00+08:00", {"status": "skipped"})
    moved = _fix(a, {"status": "done"}, at="2026-10-21T20:00:00+08:00")
    c = training_completion(
        [a, moved], planned_days=[DAY, date(2026, 10, 21)], tz=SH, rulebook=BOOK
    )
    assert c.recorded_days == 1 and c.rate == 1.0  # 20 日不再算有记录


@pytest.mark.parametrize("fix_id_smaller", [True, False])
def test_energy_uses_correction_regardless_of_id_order(fix_id_smaller: bool) -> None:
    # 更正与原记录发生时间相同：若只按“同一时刻取标识最大的”，结果会取决于标识大小
    big, small = "ffffffff-0000-4000-8000-000000000000", "00000000-0000-4000-8000-0000000000ff"
    a = _ev("energy_check", "2026-10-20T08:00:00+08:00", {"level": 2})
    a = a.model_copy(update={"event_id": UUID(big if fix_id_smaller else small)})
    fix = _ev("energy_check", "2026-10-20T08:00:00+08:00", {"level": 4}, corrects=str(a.event_id))
    fix = fix.model_copy(update={"event_id": UUID(small if fix_id_smaller else big)})
    assert energy_today([a, fix], day=DAY, tz=SH, rulebook=BOOK).level == 4


def _weights(n: int, stream: str = "scale:home") -> list[Any]:
    start = datetime(2026, 10, 1, tzinfo=UTC)
    return [
        _ev(
            "weight",
            (start + timedelta(days=i)).isoformat(),
            {"weight_kg": round(80 - 0.1 * i, 1)},
            source="device",
            stream_key=stream,
        )
        for i in range(n)
    ]


def test_trend_keeps_device_readings_after_confirmation() -> None:
    ws = _weights(10)
    confirm = _fix(ws[-1], {"weight_kg": ws[-1].payload.weight_kg})
    as_of = datetime(2026, 10, 11, tzinfo=UTC)
    before = weight_trend(ws, as_of=as_of, window_days=14, rulebook=BOOK)
    after = weight_trend([*ws, confirm], as_of=as_of, window_days=14, rulebook=BOOK)
    assert before.n == after.n == 10


# ---- 截止时间 ----


def test_freshness_respects_as_of() -> None:
    ws = _weights(1)
    later = _ev("weight", "2026-10-20T07:00:00+08:00", {"weight_kg": 79.0})
    f = weight_freshness([*ws, later], as_of=datetime(2026, 10, 15, tzinfo=UTC), rulebook=BOOK)
    assert f.status == "stale"


def test_trend_respects_as_of_for_corrections() -> None:
    ws = _weights(6)
    late_fix = _ev(
        "weight",
        ws[2].occurred_at.isoformat(),
        {"weight_kg": 60.0},
        corrects=str(ws[2].event_id),
        recorded_at="2026-10-30T00:00:00+00:00",
    )
    as_of = datetime(2026, 10, 7, tzinfo=UTC)
    a = weight_trend(ws, as_of=as_of, window_days=14, rulebook=BOOK)
    b = weight_trend([*ws, late_fix], as_of=as_of, window_days=14, rulebook=BOOK)
    assert a.model_dump() == b.model_dump()


# ---- 顺序无关 ----


@pytest.mark.parametrize("seed", range(5))
def test_order_independent(seed: int) -> None:
    ws = _weights(8)
    shuffled = list(ws)
    random.Random(seed).shuffle(shuffled)
    as_of = datetime(2026, 10, 9, tzinfo=UTC)
    assert pending_weights(ws, BOOK) == pending_weights(shuffled, BOOK)
    assert (
        weight_trend(ws, as_of=as_of, window_days=14, rulebook=BOOK).model_dump()
        == weight_trend(shuffled, as_of=as_of, window_days=14, rulebook=BOOK).model_dump()
    )


@pytest.mark.parametrize("seed", range(3))
def test_precision_order_independent_and_uses_corrections(seed: int) -> None:
    ws = _weights(10)
    moved = _fix(
        ws[0], {"weight_kg": 80.0}, at="2026-10-05T00:00:00+00:00"
    )  # 1 日的读数更正到 5 日
    shuffled = [*ws, moved]
    random.Random(seed).shuffle(shuffled)
    as_of = datetime(2026, 10, 28, tzinfo=UTC)
    a = body_precision([*ws, moved], as_of=as_of, tz=SH, rulebook=BOOK)
    b = body_precision(shuffled, as_of=as_of, tz=SH, rulebook=BOOK)
    assert a == b
    assert a.basis["weigh_days"] == 9  # 1 日不再算
    t = training_precision([], as_of=as_of, tz=SH, rulebook=BOOK)
    assert t.precision == "P0"


@pytest.mark.parametrize("seed", range(3))
def test_diet_and_recovery_precision_order_independent(seed: int) -> None:
    meals = [
        _ev("meal", f"2026-10-{d:02d}T12:00:00+08:00", {"slot": "lunch", "kcal": 600})
        for d in range(1, 20)
    ]
    late = _ev("meal", "2026-10-29T12:00:00+08:00", {"slot": "lunch", "kcal": 600})  # 截止之后
    energy = [
        _ev("energy_check", f"2026-10-{d:02d}T08:00:00+08:00", {"level": 3}) for d in range(1, 18)
    ]
    events = [*meals, late, *energy]
    shuffled = list(events)
    random.Random(seed).shuffle(shuffled)
    as_of = datetime(2026, 10, 28, 15, tzinfo=UTC)
    d1 = diet_precision(events, as_of=as_of, tz=SH, rulebook=BOOK)
    assert d1 == diet_precision(shuffled, as_of=as_of, tz=SH, rulebook=BOOK)
    assert d1.basis["logged_days"] == 19 and d1.precision == "P2"
    r1 = recovery_precision(events, as_of=as_of, tz=SH, rulebook=BOOK)
    assert r1 == recovery_precision(shuffled, as_of=as_of, tz=SH, rulebook=BOOK)
    assert r1.precision == "P1"
