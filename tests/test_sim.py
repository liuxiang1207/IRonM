"""快进模拟（spec [2026-10-20] §9）。"""

import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest

from ironm.contracts.events import EVENT_ADAPTER, WeightEvent
from ironm.core.clock import FixedClock
from ironm.store.db import connect
from ironm.store.events import EventStore
from ironm.store.migrate import MIGRATIONS_DIR, migrate
from sim.__main__ import main, metrics_at_end, summary_row, true_slope
from sim.generate import Simulation, generate
from sim.persona import PERSONAS_DIR, AdherencePhase, Persona, load_personas
from sim.runner import run

PERSONAS = load_personas(PERSONAS_DIR)


def _weights(sim: Simulation) -> dict[int, list[WeightEvent]]:
    out: dict[int, list[WeightEvent]] = {}
    for day in sim.days:
        out[day.index] = [e for e in day.events if isinstance(e, WeightEvent)]
    return out


def _trap(p: Persona, kind: str) -> dict[str, object]:
    [t] = [t.model_dump() for t in p.traps if t.kind == kind]
    return t


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = connect(tmp_path / "sim.sqlite")
    migrate(c, MIGRATIONS_DIR, FixedClock(PERSONAS["adherent"].start_at()))
    yield c
    c.close()


def test_five_personas() -> None:
    assert set(PERSONAS) == {
        "adherent",
        "low_completion",
        "fatigue",
        "weekly_weigher",
        "habit_mode",
    }


@pytest.mark.parametrize("pid", sorted(PERSONAS))
def test_eight_weeks_within_range(pid: str) -> None:
    p = PERSONAS[pid]
    sim = generate(p)
    assert [d.index for d in sim.days] == list(range(1, 57))
    start = p.start_at()
    for day in sim.days:
        for e in day.events:
            assert start <= e.occurred_at < start + timedelta(days=56)
            assert e.user_id == p.user_id
            # 往返契约校验
            assert EVENT_ADAPTER.validate_json(EVENT_ADAPTER.dump_json(e)) == e
    assert len(sim.truth.true_weight_kg) == 56


def test_deterministic_and_seed_sensitive() -> None:
    p = PERSONAS["adherent"]
    assert generate(p) == generate(p)
    other = generate(p.model_copy(update={"seed": p.seed + 1}))
    assert other.days != generate(p).days


def test_removing_trap_only_changes_its_day() -> None:
    p = PERSONAS["adherent"]
    spike_day = int(_trap(p, "spike")["day"])  # type: ignore[call-overload]
    without = p.model_copy(update={"traps": tuple(t for t in p.traps if t.kind != "spike")})
    a, b = generate(p), generate(without)
    changed = [x.index for x, y in zip(a.days, b.days, strict=True) if x.events != y.events]
    assert changed == [spike_day]


def test_weekly_weigher_weighs_eight_times() -> None:
    p = PERSONAS["weekly_weigher"]
    no_traps = generate(p.model_copy(update={"traps": ()}))
    assert sum(len(ws) for ws in _weights(no_traps).values()) == 8
    # 带“第 45 天起停止记录”时，只有第 1、8、…、43 天这 7 个周一有称重
    with_stop = _weights(generate(p))
    assert sorted(i for i, ws in with_stop.items() if ws) == [1, 8, 15, 22, 29, 36, 43]


def test_schedules_follow_habits() -> None:
    p = PERSONAS["adherent"]
    sim = generate(p)
    trap_days = {t.day for t in p.traps if t.day is not None}
    for day in sim.days:
        weekday = day.date.strftime("%a").lower()
        kinds = Counter(e.type for e in day.events)
        if day.index not in trap_days:
            assert (kinds["weight"] > 0) == (weekday in p.habits.weigh_days)
        assert (kinds["workout_check"] > 0) == (weekday in p.habits.training_days)
        assert (kinds["checkin"] > 0) == (weekday == p.habits.checkin_day)


def test_spike_trap() -> None:
    p = PERSONAS["adherent"]
    t = _trap(p, "spike")
    sim = generate(p)
    day = int(t["day"])  # type: ignore[call-overload]
    [w] = _weights(sim)[day]
    truth = sim.truth.true_weight_kg[day - 1]
    assert w.payload.weight_kg - truth >= float(t["delta_kg"]) - 0.05  # type: ignore[arg-type]


def test_unit_error_trap() -> None:
    p = PERSONAS["fatigue"]
    day = int(_trap(p, "unit_error")["day"])  # type: ignore[call-overload]
    sim = generate(p)
    [w] = _weights(sim)[day]
    assert w.payload.weight_kg == pytest.approx(2 * sim.truth.true_weight_kg[day - 1], rel=0.03)


def test_device_switch_trap() -> None:
    p = PERSONAS["low_completion"]
    t = _trap(p, "device_switch")
    sim = generate(p)
    switch_day = int(t["day"])  # type: ignore[call-overload]
    before: list[float] = []
    after: list[float] = []
    for idx, ws in _weights(sim).items():
        for w in ws:
            expected = t["stream"] if idx >= switch_day else p.habits.scale_stream
            assert w.stream_key == expected
            diff = w.payload.weight_kg - sim.truth.true_weight_kg[idx - 1]
            (after if idx >= switch_day else before).append(diff)
    # 测量噪声标准差 0.3，每段约 8 次称重，平均差的误差远小于 0.35
    assert abs(sum(before) / len(before)) < 0.35
    assert abs(sum(after) / len(after) - float(t["offset_kg"])) < 0.35  # type: ignore[arg-type]


def test_stop_logging_and_gap() -> None:
    weekly = PERSONAS["weekly_weigher"]
    stop = int(_trap(weekly, "stop_logging")["from_day"])  # type: ignore[call-overload]
    assert all(not d.events for d in generate(weekly).days if d.index >= stop)
    habit = PERSONAS["habit_mode"]
    gap = _trap(habit, "gap")
    lo, hi = int(gap["from_day"]), int(gap["to_day"])  # type: ignore[call-overload]
    sim = generate(habit)
    assert all(not d.events for d in sim.days if lo <= d.index <= hi)
    assert any(d.events for d in sim.days if d.index == hi + 1)


def test_trap_events_registered() -> None:
    for p in PERSONAS.values():
        sim = generate(p)
        ids = {e.event_id for d in sim.days for e in d.events}
        kinds = {t.kind for t in p.traps}
        assert {t.kind for t in sim.truth.traps} == kinds
        for t in sim.truth.traps:
            assert set(t.event_ids) <= ids
            if t.kind in {"spike", "unit_error", "device_switch"}:
                assert t.event_ids


def test_habit_mode_logs_no_kcal() -> None:
    sim = generate(PERSONAS["habit_mode"])
    meals = [e for d in sim.days for e in d.events if e.type == "meal"]
    assert meals
    assert all(not isinstance(m.payload.kcal, float) for m in meals)


def test_run_writes_store_and_fast_forwards(conn: sqlite3.Connection) -> None:
    p = PERSONAS["adherent"]
    sim = generate(p)
    clock = FixedClock(p.start_at())
    store = EventStore(conn, clock)
    days: list[int] = []
    weeks: list[int] = []
    run(sim, store, clock, on_day_end=days.append, on_week_end=weeks.append)
    assert days == list(range(1, 57))
    assert weeks == list(range(1, 9))
    assert clock.now() == p.start_at() + timedelta(days=56)
    stored = store.list_for_user(p.user_id)
    assert len(stored) == sum(len(d.events) for d in sim.days)
    for day in sim.days:
        for e in day.events:
            assert store.received_at(e.event_id) == day.closes_at


def test_runner_with_all_personas_in_one_db(conn: sqlite3.Connection) -> None:
    clock = FixedClock(PERSONAS["adherent"].start_at())
    store = EventStore(conn, clock)
    total = 0
    for p in PERSONAS.values():
        sim = generate(p)
        clock.set(p.start_at())
        run(sim, store, clock)
        total += sum(len(d.events) for d in sim.days)
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == total


def test_truth_not_in_events() -> None:
    sim = generate(PERSONAS["adherent"])
    for d in sim.days:
        for e in d.events:
            text = e.model_dump_json()
            for hidden_key in ("true_weight", "kg_per_week", "p_done", "weight_noise", "hidden"):
                assert hidden_key not in text


def test_command_runs_all_personas(tmp_path: Path) -> None:
    db = tmp_path / "sim.sqlite"
    report = main(db)
    for pid in PERSONAS:
        assert pid in report
    for column in ("称重", "饮食", "训练", "签到", "陷阱记录", "真实体重", "测得体重"):
        assert column in report
    with closing(connect(db)) as c:
        users = {r[0] for r in c.execute("SELECT DISTINCT user_id FROM events")}
    assert users == {p.user_id for p in PERSONAS.values()}


def test_command_reports_metrics_at_end(tmp_path: Path) -> None:
    report = main(tmp_path / "sim.sqlite")
    for column in ("精度（体重/训练/饮食/恢复）", "体重趋势 kg/周", "真实趋势", "状态", "还差什么"):
        assert column in report
    rows = {
        line.split("|")[1].strip(): line for line in report.splitlines() if line.startswith("| ")
    }
    assert "已过期" in rows["weekly_weigher（指标）"]  # 第 45 天起不再记录


def test_metric_ci_covers_truth_for_adherent() -> None:
    p = PERSONAS["adherent"]
    sim = generate(p)
    m = metrics_at_end(sim, p)
    assert m.ci is not None and isinstance(m.value, float)
    truth = true_slope(sim, m.window_days)
    assert m.ci.low <= truth <= m.ci.high


# ---- 独立审查后补充 ----


@pytest.mark.parametrize(
    "traps",
    [
        ({"kind": "spike", "day": 300, "delta_kg": 2},),
        ({"kind": "spike", "day": 10, "delta_kg": 2}, {"kind": "unit_error", "day": 10}),
        (
            {"kind": "spike", "day": 29, "delta_kg": 2},
            {"kind": "device_switch", "day": 29, "stream": "s:2", "offset_kg": 1},
        ),
        ({"kind": "gap", "from_day": 20, "to_day": 10},),
        ({"kind": "stop_logging", "from_day": 99},),
    ],
)
def test_invalid_traps_rejected(traps: tuple[dict[str, object], ...]) -> None:
    data = PERSONAS["adherent"].model_dump(mode="json") | {"traps": list(traps)}
    with pytest.raises(ValueError):
        Persona.model_validate(data)


def test_zero_adherence_generates_valid_records() -> None:
    p = PERSONAS["low_completion"]
    hidden = p.hidden.model_copy(
        update={"adherence_phases": (AdherencePhase(from_week=1, p_done=0.0),)}
    )
    sim = generate(p.model_copy(update={"hidden": hidden}))
    assert len(sim.days) == 56


def test_summary_counts_device_switch_readings() -> None:
    p = PERSONAS["low_completion"]
    sim = generate(p)
    last_weight = [e for d in sim.days for e in d.events if isinstance(e, WeightEvent)][-1]
    assert f"→ {last_weight.payload.weight_kg:.1f} |" in summary_row(sim)


@pytest.mark.parametrize("pid", sorted(PERSONAS))
def test_ok_trend_meets_certainty_target(pid: str) -> None:
    # “正常”必须以实际算出的可信范围为准：半宽不超过规则表的目标（spec §20 澄清）
    p = PERSONAS[pid]
    m = metrics_at_end(generate(p), p)
    if m.status == "ok":
        assert m.ci is not None
        assert (m.ci.high - m.ci.low) / 2 <= 0.25
