"""数据库与迁移（spec [2026-10-20] §5）。"""

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ironm.core.clock import FixedClock
from ironm.store.db import connect
from ironm.store.migrate import MIGRATIONS_DIR, MigrationError, load_migrations, migrate

# 仓库里现有的迁移编号；测试自造的迁移从下一个编号开始
REAL = [m.version for m in load_migrations(MIGRATIONS_DIR)]
NEXT = len(REAL) + 1
CLOCK = FixedClock(datetime(2026, 10, 20, tzinfo=UTC))


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r[0] for r in rows}


def _applied(conn: sqlite3.Connection) -> list[int]:
    return [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = connect(tmp_path / "db.sqlite")
    yield c
    c.close()


@pytest.fixture
def mig_dir(tmp_path: Path) -> Path:
    d = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, d)
    return d


def test_fresh_database(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "ironm.sqlite"
    with closing(connect(db)) as conn:
        assert migrate(conn, MIGRATIONS_DIR, CLOCK) == REAL
        assert db.exists()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert "events" in _tables(conn)


def test_rerun_is_noop(conn: sqlite3.Connection) -> None:
    migrate(conn, MIGRATIONS_DIR, CLOCK)
    assert migrate(conn, MIGRATIONS_DIR, CLOCK) == []


def test_edited_applied_migration_fails(conn: sqlite3.Connection, mig_dir: Path) -> None:
    migrate(conn, mig_dir, CLOCK)
    with (mig_dir / "0001_init.sql").open("a", encoding="utf-8") as f:
        f.write("\n-- 改动\n")
    with pytest.raises(MigrationError, match="0001"):
        migrate(conn, mig_dir, CLOCK)


def test_gap_fails(conn: sqlite3.Connection, mig_dir: Path) -> None:
    (mig_dir / f"{NEXT + 1:04d}_later.sql").write_text("CREATE TABLE later (a INTEGER);")
    with pytest.raises(MigrationError, match=f"{NEXT:04d}"):
        migrate(conn, mig_dir, CLOCK)


def test_duplicate_number_fails(conn: sqlite3.Connection, mig_dir: Path) -> None:
    (mig_dir / "0001_other.sql").write_text("CREATE TABLE other (a INTEGER);", encoding="utf-8")
    with pytest.raises(MigrationError, match="0001"):
        migrate(conn, mig_dir, CLOCK)


def test_badly_named_file_fails(conn: sqlite3.Connection, mig_dir: Path) -> None:
    (mig_dir / "init.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(MigrationError, match=r"init\.sql"):
        migrate(conn, mig_dir, CLOCK)


def test_applied_migration_missing_locally_fails(conn: sqlite3.Connection, mig_dir: Path) -> None:
    extra = mig_dir / f"{NEXT:04d}_extra.sql"
    extra.write_text("CREATE TABLE extra (a INTEGER);", encoding="utf-8")
    assert migrate(conn, mig_dir, CLOCK) == [*REAL, NEXT]
    extra.unlink()
    with pytest.raises(MigrationError, match=f"{NEXT:04d}"):
        migrate(conn, mig_dir, CLOCK)


def test_failed_migration_rolls_back(conn: sqlite3.Connection, mig_dir: Path) -> None:
    (mig_dir / f"{NEXT:04d}_broken.sql").write_text(
        "CREATE TABLE half_done (a INTEGER);\nINSERT INTO no_such_table VALUES (1);",
        encoding="utf-8",
    )
    with pytest.raises(MigrationError, match=f"{NEXT:04d}"):
        migrate(conn, mig_dir, CLOCK)
    assert "half_done" not in _tables(conn)
    assert _applied(conn) == REAL


def test_command_line(tmp_path: Path) -> None:
    db = tmp_path / "cli.sqlite"
    env = {k: v for k, v in os.environ.items() if not k.startswith("IRONM_")}
    env["IRONM_DB_PATH"] = str(db)
    runs = [
        subprocess.run(
            [sys.executable, "-m", "ironm.store.migrate"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        for _ in range(2)
    ]
    for run in runs:
        assert run.returncode == 0, run.stderr
    counts = [json.loads(run.stderr.splitlines()[-1])["count"] for run in runs]
    assert counts == [len(REAL), 0]
    assert db.exists()


def test_missing_dir_fails(conn: sqlite3.Connection, tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="不存在"):
        migrate(conn, tmp_path / "nope", CLOCK)


@pytest.mark.parametrize("name", ["0002_x.SQL", "notes.txt", "0002_x.sql.bak"])
def test_unexpected_file_fails(conn: sqlite3.Connection, mig_dir: Path, name: str) -> None:
    (mig_dir / name).write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(MigrationError, match=re.escape(name)):
        migrate(conn, mig_dir, CLOCK)


def test_hidden_files_ignored(conn: sqlite3.Connection, mig_dir: Path) -> None:
    (mig_dir / ".gitkeep").write_text("", encoding="utf-8")
    assert migrate(conn, mig_dir, CLOCK) == REAL
