"""安全标记与模型调用记录的存储（spec [2026-10-21] §2、§3）：只追加、读回一致。"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ironm.contracts.model_call import ModelCall, ModelStage
from ironm.contracts.screening import SafetyFlag
from ironm.core.clock import FixedClock
from ironm.core.ids import SeededIdGenerator
from ironm.store.db import connect
from ironm.store.migrate import MIGRATIONS_DIR, migrate
from ironm.store.safety import ModelCallStore, SafetyFlagStore

NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    with closing(connect(tmp_path / "db.sqlite")) as c:
        migrate(c, MIGRATIONS_DIR, FixedClock(NOW))
        yield c


def _call(call_id: str, **kw: object) -> ModelCall:
    data: dict[str, object] = {
        "call_id": call_id,
        "request_id": "req-1",
        "stage": ModelStage.ANSWER,
        "provider": "deepseek",
        "model": "deepseek-flash",
        "prompt_template": "answer@1.0.0",
        "started_at": NOW,
        "duration_ms": 120,
        "status": "ok",
        "fallback_used": False,
    }
    data.update(kw)
    return ModelCall.model_validate(data)


def test_flags_round_trip_per_user(conn: sqlite3.Connection) -> None:
    store = SafetyFlagStore(conn, SeededIdGenerator(1))
    flag = SafetyFlag(at=NOW, level="red_flag", rule_ids=("RF-CHEST",), categories=("cardio",))
    store.add("u1", "req-1", flag)
    assert store.for_user("u1") == [flag]
    assert store.for_user("u2") == []


def test_model_calls_round_trip_with_unknown_tokens(conn: sqlite3.Connection) -> None:
    store = ModelCallStore(conn)
    calls = [
        _call("c1", input_tokens=10, output_tokens=3),
        _call("c2", status="error", error_type="ConnectionError", fallback_used=True),
    ]
    for c in calls:
        store.add(c)
    assert store.for_request("req-1") == calls
    assert store.for_request("other") == []


@pytest.mark.parametrize("table", ["safety_flags", "model_calls"])
def test_tables_are_append_only(conn: sqlite3.Connection, table: str) -> None:
    SafetyFlagStore(conn, SeededIdGenerator(1)).add(
        "u1", "r", SafetyFlag(at=NOW, level="emergency", rule_ids=("EM-COLLAPSE",))
    )
    ModelCallStore(conn).add(_call("c1"))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(f"DELETE FROM {table}")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(f"UPDATE {table} SET request_id = 'x'")
