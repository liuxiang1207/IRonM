"""精度判定的记录汇总（spec §17）。各档边界见 tests/golden/precision_*.yaml。"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ironm.contracts.events import EVENT_ADAPTER
from ironm.engine.precision import body_precision, training_precision
from ironm.rules.loader import RULES_DIR, load_rules

BOOK = load_rules(RULES_DIR)
SH = ZoneInfo("Asia/Shanghai")
AS_OF = datetime(
    2026, 10, 28, 15, 0, tzinfo=UTC
)  # 上海时间 10 月 28 日 23:00；窗口为 10 月 1–28 日
_n = 0


def _ev(type_: str, local: str, payload: dict[str, Any], **extra: Any) -> Any:
    global _n
    _n += 1
    at = f"{local}+08:00"
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


def _w(day: int, kg: float = 80.0, time: str = "07:30:00", **extra: Any) -> Any:
    return _ev("weight", f"2026-10-{day:02d}T{time}", {"weight_kg": kg}, **extra)


def test_body_counts_days_not_readings() -> None:
    weights = [_w(d) for d in (1, 2, 3, 8, 9, 10, 15, 16, 17, 22, 23, 24)]
    weights += [_w(d, time="20:00:00") for d in (1, 2, 3)]  # 同一天第二次，不多算
    r = body_precision(weights, as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.precision == "P2"  # 12 天，未到 14 天
    assert r.basis["weigh_days"] == 12


def test_body_corrections_and_pending() -> None:
    days = (1, 2, 3, 4, 8, 9, 10, 11, 15, 16, 17, 22, 23, 24)
    weights = [_w(d) for d in days]
    weights.append(_w(25, kg=95.0))  # 待确认的跳变，仍算称过重
    wrong = _w(26, kg=81.0)
    moved = _ev("weight", "2026-10-24T07:30:00", {"weight_kg": 81.0}, corrects=str(wrong.event_id))
    r = body_precision([*weights, wrong, moved], as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.basis["weigh_days"] == 15  # 26 日被更正到 24 日（已有），25 日计入
    assert r.precision == "P3"


def test_body_local_date_and_window() -> None:
    early = _ev("weight", "2026-09-30T23:30:00", {"weight_kg": 80.0})  # 窗口前一天
    midnight = _ev("weight", "2026-10-01T00:30:00", {"weight_kg": 80.0})  # 上海 0:30，算 1 日
    after = _w(29)  # 截止之后
    r = body_precision([early, midnight, after], as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.basis["weigh_days"] == 1
    assert r.basis["weekly_days"] == [1, 0, 0, 0]


def _set(day: int, rir: int | None) -> Any:
    payload: dict[str, Any] = {"exercise_id": "squat", "set_index": 1, "reps": 8, "load_kg": 60}
    if rir is not None:
        payload["rir"] = rir
    return _ev("set_log", f"2026-10-{day:02d}T19:00:00", payload)


def _check(day: int, effort: int | None = None) -> Any:
    payload: dict[str, Any] = {"status": "done"}
    if effort is not None:
        payload["effort"] = effort
    return _ev("workout_check", f"2026-10-{day:02d}T20:00:00", payload)


def test_training_levels_from_records() -> None:
    events = [
        _set(1, 2),
        _set(1, 1),  # 第 1 天：每组有 RIR → 3
        _set(3, 2),
        _set(3, None),  # 第 3 天：一组缺 RIR → 2
        _check(5, effort=3),  # 第 5 天：打卡带用力 → 2
        _check(7),  # 第 7 天：只有打卡 → 1
    ]
    r = training_precision(events, as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.basis["day_levels"] == [3, 2, 2, 1]
    assert r.precision == "P2"  # 程度 ≥ 2 的占 3/4，恰好 75%


def test_training_share_from_rule_table(tmp_path: Path) -> None:
    d = tmp_path / "rules"
    d.mkdir()
    for f in RULES_DIR.glob("*.yaml"):
        (d / f.name).write_text(
            f.read_text(encoding="utf-8").replace("share_required: 0.75", "share_required: 0.5"),
            encoding="utf-8",
        )
    events = [_set(1, 2), _set(3, 2), _check(5), _check(7)]
    assert training_precision(events, as_of=AS_OF, tz=SH, rulebook=BOOK).precision == "P1"
    assert training_precision(events, as_of=AS_OF, tz=SH, rulebook=load_rules(d)).precision == "P3"


def test_training_window_excludes_old_and_future() -> None:
    old = _ev("workout_check", "2026-09-29T20:00:00", {"status": "done"})
    future = _ev("workout_check", "2026-10-29T20:00:00", {"status": "done"})
    r = training_precision([old, future, _check(2)], as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.basis["training_days"] == 1


def test_training_uses_local_date() -> None:
    # 上海时间 10 月 1 日 0:30 = UTC 9 月 30 日 16:30；按当地日期属于窗口第一天
    early = _ev("workout_check", "2026-10-01T00:30:00", {"status": "done"})
    r = training_precision([early], as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.basis["training_days"] == 1
