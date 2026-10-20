"""记录仓储与全文检索（spec [2026-10-20] §5）。"""

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ironm.contracts.events import EVENT_ADAPTER, Event
from ironm.core.clock import FixedClock
from ironm.store.db import connect
from ironm.store.events import (
    ConcurrentUseError,
    EventConflictError,
    EventStore,
    FutureEventError,
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


# ---- 独立审查后补充 ----


def test_insert_or_replace_cannot_overwrite(store: EventStore, conn: sqlite3.Connection) -> None:
    event = _event()
    store.append(event)
    row = conn.execute("SELECT * FROM events").fetchone()
    values = list(row)
    values[-1] = '{"weight_kg": 99.0}'
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(
            f"INSERT OR REPLACE INTO events VALUES ({', '.join('?' * len(values))})", values
        )
    assert store.get(event.event_id) == event


@pytest.mark.parametrize(("minutes", "accepted"), [(4, True), (6, False)])
def test_recorded_at_not_in_future(store: EventStore, minutes: int, accepted: bool) -> None:
    at = (NOW + timedelta(minutes=minutes)).isoformat()
    event = _event(occurred_at=at, recorded_at=at)
    if accepted:
        assert store.append(event) is True
    else:
        with pytest.raises(FutureEventError):
            store.append(event)
        assert store.get(event.event_id) is None


def test_types_must_be_a_collection(store: EventStore) -> None:
    with pytest.raises(TypeError):
        store.list_for_user("u1", types="weight")


def test_connection_usable_from_another_thread(store: EventStore) -> None:
    event = _event()
    store.append(event)
    result: list[object] = []
    t = threading.Thread(target=lambda: result.append(store.get(event.event_id)))
    t.start()
    t.join()
    assert result == [event]


def test_concurrent_writers_queue(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    with closing(connect(db)) as c:
        migrate(c, MIGRATIONS_DIR, FixedClock(NOW))
    errors: list[BaseException] = []

    def writer(user: str) -> None:
        with closing(connect(db)) as c:
            s = EventStore(c, FixedClock(NOW))
            try:
                for _ in range(50):
                    s.append(_event(user_id=user))
            except BaseException as e:
                errors.append(e)

    threads = [threading.Thread(target=writer, args=(u,)) for u in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with closing(connect(db)) as c:
        assert c.execute("SELECT count(*) FROM events").fetchone()[0] == 100


def test_quotes_are_matched_as_text(store: EventStore) -> None:
    quoted = _event(raw='注意"训练后"别太累')
    plain = _event(raw="今天训练后很累")
    store.append(quoted)
    store.append(plain)
    assert store.search("u1", '"训练后"') == [quoted]
    assert store.search("u1", '"训练后" OR 酸') == []


class _BlockingClock(FixedClock):
    """now() 被调用时停住，直到放行；用来让一个线程停在写入过程中。"""

    def __init__(self) -> None:
        super().__init__(NOW)
        self.entered = threading.Event()
        self.release = threading.Event()

    def now(self) -> datetime:
        self.entered.set()
        self.release.wait(5)
        return super().now()


def test_concurrent_use_of_one_connection_is_an_error(conn: sqlite3.Connection) -> None:
    clock = _BlockingClock()
    store = EventStore(conn, clock)
    writer = threading.Thread(target=store.append, args=(_event(),))
    writer.start()
    assert clock.entered.wait(5)
    try:
        with pytest.raises(ConcurrentUseError):
            store.list_for_user("u1")
    finally:
        clock.release.set()
        writer.join()
    assert len(store.list_for_user("u1")) == 1


def test_correction_fork_rejected(store: EventStore) -> None:
    original = _event()
    store.append(original)
    first = _event(corrects=str(original.event_id), payload={"weight_kg": 81.0})
    store.append(first)
    fork = _event(corrects=str(original.event_id), payload={"weight_kg": 82.0})
    with pytest.raises(InvalidCorrectionError, match="已被更正"):
        store.append(fork)
    chained = _event(corrects=str(first.event_id), payload={"weight_kg": 82.0})
    assert store.append(chained) is True
