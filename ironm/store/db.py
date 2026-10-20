"""SQLite 连接与事务。连接一律处于自动提交模式，事务由 transaction() 显式开启。"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from ironm.core.clock import ensure_utc

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, autocommit=True)
    conn.row_factory = sqlite3.Row
    mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if mode != "wal":
        raise RuntimeError(f"无法启用 WAL 模式：{path}（当前 {mode}）")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def to_db_time(dt: datetime) -> str:
    """固定格式的 UTC 字符串，字典序即时间序。"""
    return ensure_utc(dt).strftime(_TIME_FORMAT)


def from_db_time(text: str) -> datetime:
    return ensure_utc(datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f%z"))
