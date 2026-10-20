"""记录仓储：只追加、按标识去重、更正链校验、全文检索。"""

import json
import sqlite3
from collections.abc import Collection
from datetime import datetime, timedelta
from uuid import UUID

from ironm.contracts.events import EVENT_ADAPTER, Event
from ironm.core.clock import Clock
from ironm.store.db import from_db_time, to_db_time, transaction

_MIN_QUERY_CHARS = 3  # trigram 分词至少需要 3 个字才能匹配
_CLOCK_SKEW = timedelta(minutes=5)  # 允许的设备时钟误差


class EventConflictError(ValueError):
    """同一记录标识已存在，但内容不同。"""


class InvalidCorrectionError(ValueError):
    pass


class FutureEventError(ValueError):
    """记录时间晚于服务端接收时间（超出允许的时钟误差）。"""


class SearchQueryError(ValueError):
    pass


def _to_event(row: sqlite3.Row) -> Event:
    data = {
        "event_id": row["event_id"],
        "user_id": row["user_id"],
        "type": row["type"],
        "occurred_at": row["occurred_at"],
        "recorded_at": row["recorded_at"],
        "source": row["source"],
        "confirmed": bool(row["confirmed"]),
        "raw": row["raw"],
        "stream_key": row["stream_key"],
        "corrects": row["corrects"],
        "precision_hint": row["precision_hint"],
        "payload": json.loads(row["payload"]),
    }
    return EVENT_ADAPTER.validate_json(json.dumps(data))


class EventStore:
    def __init__(self, conn: sqlite3.Connection, clock: Clock) -> None:
        self._conn = conn
        self._clock = clock

    def append(self, event: Event) -> bool:
        """写入一条记录。返回 False 表示同一记录已存在（内容相同），未重复写入。"""
        received = self._clock.now()
        if event.recorded_at > received + _CLOCK_SKEW:
            raise FutureEventError(
                f"记录 {event.event_id} 的记录时间晚于接收时间 {received.isoformat()}"
            )
        with transaction(self._conn):
            existing = self.get(event.event_id)
            if existing is not None:
                if existing == event:
                    return False
                raise EventConflictError(f"记录 {event.event_id} 已存在且内容不同")
            if event.corrects is not None:
                self._check_correction(event)
            self._conn.execute(
                "INSERT INTO events (event_id, user_id, type, occurred_at, recorded_at,"
                " received_at, source, confirmed, raw, stream_key, corrects, precision_hint,"
                " payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(event.event_id),
                    event.user_id,
                    event.type,
                    to_db_time(event.occurred_at),
                    to_db_time(event.recorded_at),
                    to_db_time(received),
                    event.source,
                    int(event.confirmed),
                    event.raw,
                    event.stream_key,
                    None if event.corrects is None else str(event.corrects),
                    event.precision_hint,
                    event.payload.model_dump_json(),
                ),
            )
        return True

    def _check_correction(self, event: Event) -> None:
        target = self.get(event.corrects) if event.corrects is not None else None
        if target is None:
            raise InvalidCorrectionError(f"更正指向的记录不存在：{event.corrects}")
        if target.user_id != event.user_id:
            raise InvalidCorrectionError("更正记录必须与原记录属于同一用户")
        if target.type != event.type:
            raise InvalidCorrectionError(f"更正记录类型 {event.type} 与原记录 {target.type} 不同")

    def get(self, event_id: UUID) -> Event | None:
        row = self._conn.execute(
            "SELECT * FROM events WHERE event_id = ?", (str(event_id),)
        ).fetchone()
        return None if row is None else _to_event(row)

    def received_at(self, event_id: UUID) -> datetime | None:
        row = self._conn.execute(
            "SELECT received_at FROM events WHERE event_id = ?", (str(event_id),)
        ).fetchone()
        return None if row is None else from_db_time(row["received_at"])

    def list_for_user(
        self,
        user_id: str,
        *,
        types: Collection[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[Event]:
        """按发生时间排序；时间范围为 [since, until)。"""
        sql = "SELECT * FROM events WHERE user_id = ?"
        params: list[str] = [user_id]
        if isinstance(types, str):
            raise TypeError('types 必须是类型的集合，例如 {"weight"}，不能是单个字符串')
        if types is not None:
            sql += f" AND type IN ({', '.join('?' * len(types))})"
            params.extend(types)
        if since is not None:
            sql += " AND occurred_at >= ?"
            params.append(to_db_time(since))
        if until is not None:
            sql += " AND occurred_at < ?"
            params.append(to_db_time(until))
        sql += " ORDER BY occurred_at, seq"
        return [_to_event(r) for r in self._conn.execute(sql, params)]

    def search(self, user_id: str, text: str) -> list[Event]:
        """在原文与身体不适描述中按子串检索；检索词整体作为一个短语，符号按普通文字处理。"""
        text = text.strip()
        if len(text) < _MIN_QUERY_CHARS:
            raise SearchQueryError(f"检索词至少 {_MIN_QUERY_CHARS} 个字：{text!r}")
        phrase = '"' + text.replace('"', '""') + '"'
        rows = self._conn.execute(
            "SELECT e.* FROM events_fts f JOIN events e ON e.seq = f.rowid"
            " WHERE events_fts MATCH ? AND e.user_id = ? ORDER BY e.occurred_at, e.seq",
            (phrase, user_id),
        )
        return [_to_event(r) for r in rows]
