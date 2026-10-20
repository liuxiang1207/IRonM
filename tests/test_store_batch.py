"""批量入库与顺序无关性（spec [2026-10-20] §11）。"""

import json
import sqlite3
import tempfile
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from ironm.contracts.events import EVENT_ADAPTER, Event
from ironm.core.clock import FixedClock
from ironm.store.db import connect
from ironm.store.events import EventStore
from ironm.store.migrate import MIGRATIONS_DIR, migrate

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def _ev(
    n: int, user: str = "u1", day: int = 1, kg: float = 80.0, corrects: int | None = None
) -> Event:
    data: dict[str, Any] = {
        "event_id": f"00000000-0000-4000-8000-{n:012d}",
        "user_id": user,
        "type": "weight",
        "occurred_at": f"2026-10-{day:02d}T07:30:00+08:00",
        "recorded_at": f"2026-10-{day:02d}T07:31:00+08:00",
        "source": "user",
        "confirmed": True,
        "payload": {"weight_kg": kg},
    }
    if corrects is not None:
        data["corrects"] = f"00000000-0000-4000-8000-{corrects:012d}"
    return EVENT_ADAPTER.validate_json(json.dumps(data))


# 一组记录：同一天多条（检验同一发生时间的排序）、两条更正、另一个用户
EVENTS: list[Event] = [
    _ev(1, day=1, kg=80.0),
    _ev(2, day=3, kg=79.6),
    _ev(3, day=3, kg=79.8),
    _ev(4, day=5, kg=79.1),
    _ev(5, day=1, kg=81.0, corrects=1),
    _ev(6, day=5, kg=79.0, corrects=4),
    _ev(7, user="u2", day=2, kg=60.0),
    _ev(8, user="u2", day=2, kg=60.2),
]
TARGET = {
    i: next(j for j, t in enumerate(EVENTS) if t.event_id == e.corrects)
    for i, e in enumerate(EVENTS)
    if e.corrects is not None
}


def _rows(conn: sqlite3.Connection) -> list[tuple[object, ...]]:
    cols = "event_id, user_id, type, occurred_at, recorded_at, received_at, corrects, payload"
    return [tuple(r) for r in conn.execute(f"SELECT {cols} FROM events ORDER BY event_id")]


def _submit(
    batches: Sequence[Sequence[Event]],
) -> tuple[list[tuple[object, ...]], dict[str, list[Event]]]:
    with tempfile.TemporaryDirectory() as d, closing(connect(Path(d) / "db.sqlite")) as conn:
        clock = FixedClock(NOW)
        migrate(conn, MIGRATIONS_DIR, clock)
        store = EventStore(conn, clock)
        for batch in batches:
            store.append_many(batch)
        return _rows(conn), {u: store.list_for_user(u) for u in ("u1", "u2")}


REFERENCE = _submit([EVENTS])


def test_results_aligned_with_input_order(tmp_path: Path) -> None:
    with closing(connect(tmp_path / "db.sqlite")) as conn:
        clock = FixedClock(NOW)
        migrate(conn, MIGRATIONS_DIR, clock)
        store = EventStore(conn, clock)
        batch = [EVENTS[4], EVENTS[0], EVENTS[0]]  # 更正排在原记录之前，且原记录重复
        results = store.append_many(batch)
    assert [r.event_id for r in results] == [e.event_id for e in batch]
    assert [r.status for r in results] == ["created", "created", "duplicate"]


def test_conflict_and_invalid_reported_per_item(tmp_path: Path) -> None:
    with closing(connect(tmp_path / "db.sqlite")) as conn:
        clock = FixedClock(NOW)
        migrate(conn, MIGRATIONS_DIR, clock)
        store = EventStore(conn, clock)
        changed = EVENTS[0].model_copy(update={"raw": "改过"})
        orphan = _ev(9, corrects=99)
        results = store.append_many([EVENTS[0], changed, orphan, EVENTS[1]])
    # 同一批中同一标识内容不一致：两条都判冲突（与先后无关，spec §11 澄清）
    assert [r.status for r in results] == ["conflict", "conflict", "invalid", "created"]
    assert results[2].reason is not None


@st.composite
def submissions(draw: st.DrawFn) -> list[list[Event]]:
    order = draw(st.permutations(range(len(EVENTS))))
    extra = draw(st.lists(st.integers(0, len(EVENTS) - 1), max_size=6))
    seq = list(order) + extra
    seq = draw(st.permutations(seq))
    cuts = sorted(draw(st.sets(st.integers(1, max(1, len(seq) - 1)), max_size=4)))
    batches = [
        list(seq[a:b]) for a, b in zip([0, *cuts], [*cuts, len(seq)], strict=True) if seq[a:b]
    ]
    # 跨批次时更正不早于其原记录：把过早出现的更正挪到原记录首次出现的那一批
    first = {i: next(k for k, b in enumerate(batches) if i in b) for i in range(len(EVENTS))}
    for corr, target in TARGET.items():
        if first[corr] < first[target]:
            for b in batches:
                b[:] = [x for x in b if x != corr]
            batches[first[target]].append(corr)
            first = {
                i: next(k for k, b in enumerate(batches) if i in b) for i in range(len(EVENTS))
            }
    return [[EVENTS[i] for i in b] for b in batches if b]


@settings(max_examples=60, deadline=None)
@given(submissions())
def test_order_and_duplicates_do_not_change_result(batches: list[list[Event]]) -> None:
    assert _submit(batches) == REFERENCE


def test_same_time_sorted_by_id_not_arrival() -> None:
    _, listed = _submit([[EVENTS[2]], [EVENTS[1]]])  # 第 3 天两条，倒序到达
    assert [e.event_id for e in listed["u1"]] == [EVENTS[1].event_id, EVENTS[2].event_id]
