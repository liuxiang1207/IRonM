"""记录仓储与全文检索（spec [2026-10-20] §5）。"""

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ironm.contracts.events import EVENT_ADAPTER, Event
from ironm.core.clock import FixedClock
from ironm.store.db import connect
from ironm.store.events import (
    EventConflictError,
    EventStore,
    InvalidCorrectionError,
    SearchQueryError,
)
from ironm.store.migrate import MIGRATIONS_DIR, migrate
from tests.test_contracts_events import PAYLOADS

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
_counter = 0


def _event(type_: str = "weight", **overrides: Any) -> Event:
    global _counter
    _counter += 1
    data: dict[str, Any] = {
        "event_id": f"00000000-0000-4000-8000-{_counter:012d}",
        "user_id": "u1",
        "type": type_,
        "occurred_at": "2026-10-20T07:30:00+08:00",
        "recorded_at": "2026-10-20T07:31:00+08:00",
        "source": "user",
        "confirmed": True,
        "payload": PAYLOADS[type_],
    }
    data.update(overrides)
    return EVENT_ADAPTER.validate_json(json.dumps(data))


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = connect(tmp_path / "db.sqlite")
    migrate(c, MIGRATIONS_DIR, FixedClock(NOW))
    yield c
    c.close()


@pytest.fixture
def store(conn: sqlite3.Connection) -> EventStore:
    return EventStore(conn, FixedClock(NOW))


@pytest.mark.parametrize("type_", sorted(PAYLOADS))
def test_round_trip(store: EventStore, type_: str) -> None:
    event = _event(type_, raw="原话", stream_key="s:1", precision_hint="P2")
    assert store.append(event) is True
    assert store.get(event.event_id) == event


def test_get_missing(store: EventStore) -> None:
    assert store.get(_event().event_id) is None


def test_received_at_from_clock(store: EventStore) -> None:
    event = _event()
    store.append(event)
    assert store.received_at(event.event_id) == NOW


def test_list_filters_and_order(store: EventStore) -> None:
    late = _event(occurred_at="2026-10-18T09:00:00+00:00", recorded_at="2026-10-19T00:00:00+00:00")
    early = _event(occurred_at="2026-10-15T09:00:00+00:00", recorded_at="2026-10-19T00:00:00+00:00")
    meal = _event("meal", occurred_at="2026-10-16T04:00:00+00:00")
    other = _event(user_id="u2")
    for e in (late, early, meal, other):
        store.append(e)
    assert store.list_for_user("u1") == [early, meal, late]
    assert store.list_for_user("u1", types={"weight"}) == [early, late]
    since = datetime(2026, 10, 16, tzinfo=UTC)
    until = datetime(2026, 10, 18, 9, tzinfo=UTC)
    assert store.list_for_user("u1", since=since, until=until) == [meal]
    assert store.list_for_user("u2") == [other]


def test_duplicate_identical_stored_once(store: EventStore, conn: sqlite3.Connection) -> None:
    event = _event()
    assert store.append(event) is True
    assert store.append(event) is False
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1


def test_duplicate_conflicting_rejected(store: EventStore) -> None:
    event = _event()
    store.append(event)
    changed = event.model_copy(update={"raw": "改过的原话"})
    with pytest.raises(EventConflictError):
        store.append(changed)
    assert store.get(event.event_id) == event


@pytest.mark.parametrize("sql", ["UPDATE events SET user_id = 'u9'", "DELETE FROM events"])
def test_append_only(store: EventStore, conn: sqlite3.Connection, sql: str) -> None:
    store.append(_event())
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(sql)
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 1


def test_valid_correction(store: EventStore) -> None:
    original = _event()
    store.append(original)
    fix = _event(corrects=str(original.event_id), payload={"weight_kg": 81.0})
    assert store.append(fix) is True


@pytest.mark.parametrize("problem", ["missing", "other_user", "other_type"])
def test_invalid_correction(store: EventStore, problem: str) -> None:
    original = _event(user_id="u2") if problem == "other_user" else _event()
    if problem != "missing":
        store.append(original)
    fix_type = "meal" if problem == "other_type" else "weight"
    fix = _event(fix_type, corrects=str(original.event_id))
    with pytest.raises(InvalidCorrectionError):
        store.append(fix)
    assert store.get(fix.event_id) is None


def test_search_chinese_substring(store: EventStore) -> None:
    mine = _event("symptom")  # 内容为“右膝训练后有点酸”
    by_raw = _event(raw="今天训练后很累")
    theirs = _event("symptom", user_id="u2")
    unrelated = _event(raw="早上空腹称重")
    for e in (mine, by_raw, theirs, unrelated):
        store.append(e)
    assert store.search("u1", "训练后") == [mine, by_raw]


def test_search_too_short(store: EventStore) -> None:
    with pytest.raises(SearchQueryError):
        store.search("u1", "训练")


@pytest.mark.parametrize("query", ['训练后"', '"训练后" OR 酸', "训练后 AND", "NEAR(训练后)"])
def test_search_special_characters_are_text(store: EventStore, query: str) -> None:
    store.append(_event("symptom"))
    store.search("u1", query)  # 不抛语法错误即可
