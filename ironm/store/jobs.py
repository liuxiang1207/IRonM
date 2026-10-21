"""定时任务表（spec [2026-10-21] §16）：同一用户、同一任务、同一周期只有一行。"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from ironm.store.db import from_db_time, to_db_time, transaction

__all__ = ["MAX_ATTEMPTS", "Job", "JobStore"]

MAX_ATTEMPTS = 3
RETRY_AFTER = timedelta(minutes=5)


@dataclass(frozen=True)
class Job:
    job_key: str
    kind: str
    user_id: str
    period: str
    due_at: datetime
    attempts: int


class JobStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def ensure(self, kind: str, user_id: str, period: str, due_at: datetime, now: datetime) -> bool:
        """登记任务；已经登记过（同一 job_key）的不再登记。返回是否新登记。"""
        key = f"{kind}:{user_id}:{period}"
        with transaction(self._conn):
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO jobs (job_key, kind, user_id, period, due_at, status,"
                " updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (key, kind, user_id, period, to_db_time(due_at), to_db_time(now)),
            )
        return cur.rowcount == 1

    def claim(self, now: datetime, limit: int = 50) -> list[Job]:
        """取出到期的任务（按到期时间先后，停机补跑时也按周期顺序），标为执行中。"""
        with transaction(self._conn):
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status = 'pending' AND due_at <= ?"
                " ORDER BY due_at, job_key LIMIT ?",
                (to_db_time(now), limit),
            ).fetchall()
            for r in rows:
                self._conn.execute(
                    "UPDATE jobs SET status = 'running', attempts = attempts + 1, updated_at = ?"
                    " WHERE job_key = ?",
                    (to_db_time(now), r["job_key"]),
                )
        return [
            Job(r["job_key"], r["kind"], r["user_id"], r["period"], from_db_time(r["due_at"]),
                r["attempts"] + 1)
            for r in rows
        ]  # fmt: skip

    def finish(self, job: Job, now: datetime) -> None:
        with transaction(self._conn):
            self._conn.execute(
                "UPDATE jobs SET status = 'done', updated_at = ? WHERE job_key = ?",
                (to_db_time(now), job.job_key),
            )

    def fail(self, job: Job, error_type: str, now: datetime) -> bool:
        """记下失败；未满 MAX_ATTEMPTS 次时 5 分钟后重试。返回是否已最终失败。"""
        final = job.attempts >= MAX_ATTEMPTS
        with transaction(self._conn):
            self._conn.execute(
                "UPDATE jobs SET status = ?, last_error = ?, due_at = ?, updated_at = ?"
                " WHERE job_key = ?",
                (
                    "failed" if final else "pending",
                    error_type,
                    to_db_time(now + RETRY_AFTER if not final else job.due_at),
                    to_db_time(now),
                    job.job_key,
                ),
            )
        return final

    def status(self, job_key: str) -> tuple[str, int] | None:
        row = self._conn.execute(
            "SELECT status, attempts FROM jobs WHERE job_key = ?", (job_key,)
        ).fetchone()
        return None if row is None else (row["status"], row["attempts"])

    def reset_running(self, now: datetime) -> None:
        """服务重启时，上次没执行完的任务重新排队。"""
        with transaction(self._conn):
            self._conn.execute(
                "UPDATE jobs SET status = 'pending', updated_at = ? WHERE status = 'running'",
                (to_db_time(now),),
            )
