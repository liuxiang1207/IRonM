"""计划版本仓储（spec [2026-10-21] §15）：只追加；版本号按用户从 1 连续递增。"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ironm.store.db import from_db_time, to_db_time, transaction

__all__ = ["PlanStore", "StoredPlan", "VersionConflict"]


class VersionConflict(Exception):
    """写入时发现别的请求已经写了新版本（并发生成计划）。"""


@dataclass(frozen=True)
class StoredPlan:
    version: int
    record: dict[str, Any]
    reason: str
    at: datetime


class PlanStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def users(self) -> list[str]:
        """有计划的用户。"""
        rows = self._conn.execute(
            "SELECT DISTINCT user_id FROM plan_versions ORDER BY user_id"
        ).fetchall()
        return [r["user_id"] for r in rows]

    def latest(self, user_id: str) -> StoredPlan | None:
        row = self._conn.execute(
            "SELECT version, record, reason, at FROM plan_versions WHERE user_id = ?"
            " ORDER BY version DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return None if row is None else _stored(row)

    def history(self, user_id: str) -> list[StoredPlan]:
        rows = self._conn.execute(
            "SELECT version, record, reason, at FROM plan_versions WHERE user_id = ?"
            " ORDER BY version",
            (user_id,),
        ).fetchall()
        return [_stored(r) for r in rows]

    def add(
        self, user_id: str, expected: int, record: dict[str, Any], reason: str, at: datetime
    ) -> int:
        """写入 expected + 1 版；当前最新版本不是 expected 时抛 VersionConflict。"""
        with transaction(self._conn):
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM plan_versions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row["v"] != expected:
                raise VersionConflict(f"当前版本是 {row['v']}，不是 {expected}")
            self._conn.execute(
                "INSERT INTO plan_versions (user_id, version, record, reason, at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    user_id,
                    expected + 1,
                    json.dumps(record, ensure_ascii=False),
                    reason,
                    to_db_time(at),
                ),
            )
        return expected + 1


def _stored(row: sqlite3.Row) -> StoredPlan:
    return StoredPlan(
        row["version"], json.loads(row["record"]), row["reason"], from_db_time(row["at"])
    )
