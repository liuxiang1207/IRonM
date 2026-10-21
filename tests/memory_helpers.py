"""测试用的临时对话记忆：每个测试结束时由 conftest 关闭连接。"""

import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ironm.core.clock import FixedClock
from ironm.store.db import connect
from ironm.store.memory import MemoryStore
from ironm.store.migrate import MIGRATIONS_DIR, migrate

_OPEN: list[sqlite3.Connection] = []


def memory_db() -> MemoryStore:
    """临时数据库（已迁移）上的对话记忆。"""
    conn = connect(Path(tempfile.mkdtemp()) / "db.sqlite")
    migrate(conn, MIGRATIONS_DIR, FixedClock(datetime(2026, 10, 21, tzinfo=UTC)))
    _OPEN.append(conn)
    return MemoryStore(conn)


def close_all() -> None:
    while _OPEN:
        _OPEN.pop().close()
