"""决策日志与已发布的调整（spec [2026-10-21] §17）。决策日志只追加。"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from ironm.store.db import from_db_time, to_db_time, transaction

__all__ = ["Adjustment", "DecisionStore", "StoredDecision"]

type AdjustmentStatus = Literal["pending", "kept", "rolled_back", "superseded"]


@dataclass(frozen=True)
class StoredDecision:
    kind: str
    period: str
    record: dict[str, Any]
    at: datetime


@dataclass(frozen=True)
class Adjustment:
    adjustment_id: str
    user_id: str
    field: str
    from_value: Any
    to_value: Any
    direction: Literal["up", "down", "same"]
    plan_version: int
    review_date: date
    status: AdjustmentStatus


class DecisionStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self, user_id: str, kind: str, period: str, record: dict[str, Any], at: datetime
    ) -> None:
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO decision_logs (user_id, kind, period, record, at)"
                " VALUES (?, ?, ?, ?, ?)",
                (user_id, kind, period, json.dumps(record, ensure_ascii=False), to_db_time(at)),
            )

    def recent(
        self, user_id: str, kind: str | None = None, limit: int = 50
    ) -> list[StoredDecision]:
        sql = "SELECT kind, period, record, at FROM decision_logs WHERE user_id = ?"
        params: list[Any] = [user_id]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind)
        rows = self._conn.execute(sql + " ORDER BY seq DESC LIMIT ?", (*params, limit)).fetchall()
        return [
            StoredDecision(r["kind"], r["period"], json.loads(r["record"]), from_db_time(r["at"]))
            for r in rows
        ]

    def add_adjustment(self, a: Adjustment) -> None:
        with transaction(self._conn):
            # 同一字段之前还在等复核的调整被新的调整取代
            self._conn.execute(
                "UPDATE adjustments SET status = 'superseded' WHERE user_id = ? AND field = ?"
                " AND status = 'pending'",
                (a.user_id, a.field),
            )
            self._conn.execute(
                "INSERT INTO adjustments (adjustment_id, user_id, field, from_value, to_value,"
                " direction, plan_version, review_date, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    a.adjustment_id,
                    a.user_id,
                    a.field,
                    None if a.from_value is None else json.dumps(a.from_value, ensure_ascii=False),
                    json.dumps(a.to_value, ensure_ascii=False),
                    a.direction,
                    a.plan_version,
                    a.review_date.isoformat(),
                    a.status,
                ),
            )

    def due_adjustments(self, user_id: str, today: date) -> list[Adjustment]:
        rows = self._conn.execute(
            "SELECT * FROM adjustments WHERE user_id = ? AND status = 'pending'"
            " AND review_date <= ? ORDER BY review_date, adjustment_id",
            (user_id, today.isoformat()),
        ).fetchall()
        return [_adjustment(r) for r in rows]

    def adjustment(self, adjustment_id: str) -> Adjustment | None:
        row = self._conn.execute(
            "SELECT * FROM adjustments WHERE adjustment_id = ?", (adjustment_id,)
        ).fetchone()
        return None if row is None else _adjustment(row)

    def settle(
        self,
        adjustment_id: str,
        status: AdjustmentStatus,
        at: datetime,
        review_date: date | None = None,
    ) -> None:
        """记下复核结果；延后时传新的复核日并保持 pending。"""
        with transaction(self._conn):
            if review_date is not None:
                self._conn.execute(
                    "UPDATE adjustments SET review_date = ? WHERE adjustment_id = ?",
                    (review_date.isoformat(), adjustment_id),
                )
            else:
                self._conn.execute(
                    "UPDATE adjustments SET status = ?, decided_at = ? WHERE adjustment_id = ?",
                    (status, to_db_time(at), adjustment_id),
                )


def _adjustment(r: sqlite3.Row) -> Adjustment:
    return Adjustment(
        adjustment_id=r["adjustment_id"],
        user_id=r["user_id"],
        field=r["field"],
        from_value=None if r["from_value"] is None else json.loads(r["from_value"]),
        to_value=json.loads(r["to_value"]),
        direction=r["direction"],
        plan_version=r["plan_version"],
        review_date=date.fromisoformat(r["review_date"]),
        status=r["status"],
    )
