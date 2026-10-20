"""按编号执行迁移脚本。用法：python -m ironm.store.migrate（数据库路径取自 IRONM_DB_PATH）。

脚本命名为 NNNN_说明.sql，编号从 0001 起连续；每个脚本在一个事务内执行，出错整体回滚。
脚本内不要自带 BEGIN / COMMIT。
"""

import hashlib
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from ironm.core.clock import Clock, SystemClock
from ironm.core.config import load_settings
from ironm.core.log import configure_logging, get_logger
from ironm.store.db import connect, to_db_time, transaction

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"
_NAME_RE = re.compile(r"(\d{4})_[a-z0-9_]+\.sql")


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str


def load_migrations(migrations_dir: Path) -> list[Migration]:
    found: dict[int, Migration] = {}
    for path in sorted(migrations_dir.glob("*.sql")):
        match = _NAME_RE.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"迁移文件名不合规（应为 NNNN_说明.sql）：{path.name}")
        version = int(match.group(1))
        if version in found:
            raise MigrationError(f"迁移编号 {version:04d} 重复：{found[version].name}、{path.name}")
        sql = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode()).hexdigest()
        found[version] = Migration(version, path.name, sql, checksum)
    for expected, version in enumerate(sorted(found), start=1):
        if version != expected:
            raise MigrationError(f"迁移编号不连续：缺少 {expected:04d}")
    return [found[v] for v in sorted(found)]


def migrate(conn: sqlite3.Connection, migrations_dir: Path, clock: Clock) -> list[int]:
    """执行尚未执行的迁移，返回本次执行的编号。"""
    migrations = load_migrations(migrations_dir)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version INTEGER PRIMARY KEY, name TEXT NOT NULL,"
        " checksum TEXT NOT NULL, applied_at TEXT NOT NULL) STRICT"
    )
    applied = {
        row["version"]: row["checksum"]
        for row in conn.execute("SELECT version, checksum FROM schema_migrations")
    }
    local = {m.version: m for m in migrations}
    for version, checksum in sorted(applied.items()):
        if version not in local:
            raise MigrationError(f"数据库已执行迁移 {version:04d}，本地却没有该文件")
        if local[version].checksum != checksum:
            raise MigrationError(f"迁移 {version:04d} 已执行过，但文件内容被改动")

    done: list[int] = []
    for m in migrations:
        if m.version in applied:
            continue
        try:
            with transaction(conn):
                conn.executescript(m.sql)
                conn.execute(
                    "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                    (m.version, m.name, m.checksum, to_db_time(clock.now())),
                )
        except sqlite3.Error as e:
            raise MigrationError(f"迁移 {m.version:04d} 执行失败，已回滚：{e}") from e
        done.append(m.version)
    return done


if __name__ == "__main__":
    settings = load_settings(os.environ)
    clock = SystemClock()
    configure_logging(settings, clock, sys.stderr)
    applied_now = migrate(connect(settings.db_path), MIGRATIONS_DIR, clock)
    get_logger("ironm.store.migrate").info("store.migrated", count=len(applied_now))
