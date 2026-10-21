"""对话记忆的存储（spec [2026-10-21] §8）：消息、摘要、画像版本与画像候选。

消息原文只存在数据库里，从不写入日志。画像只追加新版本；
候选的状态可以改（待确认 → 已确认 / 已拒绝）。
"""

import json
import sqlite3
from datetime import datetime
from typing import Any, Literal

from ironm.contracts.memory import Candidate, Role, Status, StoredMessage, Summary
from ironm.store.db import from_db_time, to_db_time, transaction


class MemoryStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ---- 消息与摘要 ----

    def add_message(
        self, user_id: str, request_id: str, role: Role, text: str, intent: str | None, at: datetime
    ) -> int:
        with transaction(self._conn):
            cur = self._conn.execute(
                "INSERT INTO chat_messages (user_id, request_id, role, text, intent, at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, request_id, role, text, intent, to_db_time(at)),
            )
        return int(cur.lastrowid or 0)

    def messages_after(self, user_id: str, after_seq: int) -> list[StoredMessage]:
        rows = self._conn.execute(
            "SELECT seq, role, text, intent FROM chat_messages WHERE user_id = ? AND seq > ?"
            " ORDER BY seq",
            (user_id, after_seq),
        ).fetchall()
        return [StoredMessage(r["seq"], r["role"], r["text"], r["intent"]) for r in rows]

    def latest_summary(self, user_id: str) -> Summary | None:
        row = self._conn.execute(
            "SELECT upto_seq, summary, source FROM chat_summaries WHERE user_id = ?"
            " ORDER BY seq DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return None if row is None else Summary(row["upto_seq"], row["summary"], row["source"])

    def add_summary(
        self, user_id: str, upto_seq: int, text: str, source: Literal["model", "rule"], at: datetime
    ) -> None:
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO chat_summaries (user_id, upto_seq, summary, source, at)"
                " VALUES (?, ?, ?, ?, ?)",
                (user_id, upto_seq, text, source, to_db_time(at)),
            )

    # ---- 画像 ----

    def profile(self, user_id: str) -> tuple[int, dict[str, Any]]:
        """(版本, 画像)；还没有画像时为 (0, {})。"""
        row = self._conn.execute(
            "SELECT version, data FROM profiles WHERE user_id = ? ORDER BY version DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return (0, {}) if row is None else (row["version"], json.loads(row["data"]))

    def save_profile(
        self, user_id: str, expected_version: int, data: dict[str, Any], reason: str, at: datetime
    ) -> int:
        """写入新版本；expected_version 不是当前版本时抛 ValueError（防止并发覆盖）。"""
        with transaction(self._conn):
            current, _ = self.profile(user_id)
            if current != expected_version:
                raise ValueError("画像已被更新，请重试")
            self._conn.execute(
                "INSERT INTO profiles (user_id, version, data, reason, at) VALUES (?, ?, ?, ?, ?)",
                (
                    user_id,
                    current + 1,
                    json.dumps(data, ensure_ascii=False),
                    reason,
                    to_db_time(at),
                ),
            )
        return current + 1

    # ---- 画像候选 ----

    def add_candidate(self, c: Candidate) -> None:
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO profile_candidates (candidate_id, user_id, request_id, field, op,"
                " value, immediate, status, at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    c.candidate_id,
                    c.user_id,
                    c.request_id,
                    c.field,
                    c.op,
                    json.dumps(c.value, ensure_ascii=False),
                    int(c.immediate),
                    c.status,
                    to_db_time(c.at),
                ),
            )

    def candidates(self, user_id: str, status: Status | None = None) -> list[Candidate]:
        sql = "SELECT * FROM profile_candidates WHERE user_id = ?"
        args: tuple[Any, ...] = (user_id,)
        if status is not None:
            sql, args = sql + " AND status = ?", (*args, status)
        rows = self._conn.execute(sql + " ORDER BY seq", args).fetchall()
        return [self._candidate(r) for r in rows]

    def candidate(self, candidate_id: str) -> Candidate | None:
        row = self._conn.execute(
            "SELECT * FROM profile_candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        return None if row is None else self._candidate(row)

    def decide(self, candidate_id: str, status: Status, at: datetime) -> None:
        with transaction(self._conn):
            self._conn.execute(
                "UPDATE profile_candidates SET status = ?, decided_at = ?"
                " WHERE candidate_id = ? AND status = 'pending'",
                (status, to_db_time(at), candidate_id),
            )

    @staticmethod
    def _candidate(r: sqlite3.Row) -> Candidate:
        return Candidate(
            candidate_id=r["candidate_id"],
            user_id=r["user_id"],
            request_id=r["request_id"],
            field=r["field"],
            op=r["op"],
            value=json.loads(r["value"]),
            immediate=bool(r["immediate"]),
            status=r["status"],
            at=from_db_time(r["at"]),
        )
