"""登录（spec [2026-10-21] §14）：白名单用户、一次性设备码、设备凭证。

- 设备码：10 位，去掉易混的 0/O、1/I/L，显示为 XXXXX-XXXXX；24 小时内有效，用一次即作废。
- 凭证：32 字节随机数；服务端只存 SHA-256 摘要。设备码与凭证都用系统安全随机数生成，
  不用可复现的标识生成器。
"""

import hashlib
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from ironm.core.clock import Clock
from ironm.store.db import from_db_time, to_db_time, transaction

__all__ = ["CODE_TTL", "AuthStore", "Device", "User", "normalize_code"]

CODE_TTL = timedelta(hours=24)
_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_CODE_LEN = 10


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def normalize_code(code: str) -> str:
    """大小写、空格与连字符不影响设备码。"""
    return "".join(ch for ch in code.upper() if ch not in " -\t")


@dataclass(frozen=True)
class Device:
    device_id: str
    name: str
    created_at: datetime


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    created_at: datetime
    disabled: bool


class AuthStore:
    def __init__(self, conn: sqlite3.Connection, clock: Clock) -> None:
        self._conn = conn
        self._clock = clock

    def add_user(self, name: str) -> str:
        user_id = "u_" + secrets.token_hex(6)
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
                (user_id, name, to_db_time(self._clock.now())),
            )
        return user_id

    def users(self) -> list[User]:
        rows = self._conn.execute(
            "SELECT user_id, name, created_at, disabled_at FROM users ORDER BY created_at, user_id"
        ).fetchall()
        return [
            User(
                r["user_id"], r["name"], from_db_time(r["created_at"]), r["disabled_at"] is not None
            )
            for r in rows
        ]

    def new_code(self, user_id: str) -> str:
        """给用户一张新的一次性设备码（显示用格式）。用户不存在或已停用时抛 KeyError。"""
        row = self._conn.execute(
            "SELECT disabled_at FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None or row["disabled_at"] is not None:
            raise KeyError(user_id)
        code = "".join(secrets.choice(_ALPHABET) for _ in range(_CODE_LEN))
        now = self._clock.now()
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO device_codes (code_hash, user_id, created_at, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (_digest(code), user_id, to_db_time(now), to_db_time(now + CODE_TTL)),
            )
        return f"{code[:5]}-{code[5:]}"

    def login(self, code: str, device: str) -> tuple[str, str] | None:
        """设备码换凭证：(用户, 凭证)；设备码无效、过期、用过或用户已停用时为 None。"""
        now = self._clock.now()
        code_hash = _digest(normalize_code(code))
        with transaction(self._conn):
            row = self._conn.execute(
                "SELECT c.user_id, c.expires_at, c.used_at, u.disabled_at FROM device_codes c"
                " JOIN users u ON u.user_id = c.user_id WHERE c.code_hash = ?",
                (code_hash,),
            ).fetchone()
            if (
                row is None
                or row["used_at"] is not None
                or row["disabled_at"] is not None
                or from_db_time(row["expires_at"]) <= now
            ):
                return None
            token = secrets.token_urlsafe(32)
            self._conn.execute(
                "UPDATE device_codes SET used_at = ? WHERE code_hash = ?",
                (to_db_time(now), code_hash),
            )
            self._conn.execute(
                "INSERT INTO devices (device_id, user_id, token_hash, name, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    "d_" + secrets.token_hex(6),
                    row["user_id"],
                    _digest(token),
                    device,
                    to_db_time(now),
                ),
            )
        return row["user_id"], token

    def user_for(self, token: str) -> tuple[str, str] | None:
        """凭证对应的 (用户, 设备)；凭证无效、设备已撤销或用户已停用时为 None。"""
        row = self._conn.execute(
            "SELECT d.user_id, d.device_id FROM devices d JOIN users u ON u.user_id = d.user_id"
            " WHERE d.token_hash = ? AND d.revoked_at IS NULL AND u.disabled_at IS NULL",
            (_digest(token),),
        ).fetchone()
        return None if row is None else (row["user_id"], row["device_id"])

    def devices(self, user_id: str) -> list[Device]:
        rows = self._conn.execute(
            "SELECT device_id, name, created_at FROM devices"
            " WHERE user_id = ? AND revoked_at IS NULL ORDER BY created_at, device_id",
            (user_id,),
        ).fetchall()
        return [Device(r["device_id"], r["name"], from_db_time(r["created_at"])) for r in rows]

    def revoke(self, user_id: str, device_id: str) -> bool:
        """撤销自己的一台设备；不是自己的或已撤销时返回 False。"""
        with transaction(self._conn):
            cur = self._conn.execute(
                "UPDATE devices SET revoked_at = ? WHERE device_id = ? AND user_id = ?"
                " AND revoked_at IS NULL",
                (to_db_time(self._clock.now()), device_id, user_id),
            )
        return cur.rowcount == 1

    def disable(self, user_id: str) -> bool:
        with transaction(self._conn):
            cur = self._conn.execute(
                "UPDATE users SET disabled_at = ? WHERE user_id = ? AND disabled_at IS NULL",
                (to_db_time(self._clock.now()), user_id),
            )
        return cur.rowcount == 1
