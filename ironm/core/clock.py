"""统一的时间来源。业务代码只通过注入的 Clock 取当前时间，测试与快进模拟用 FixedClock。"""

from datetime import UTC, datetime, timedelta
from typing import Protocol


class NaiveDatetimeError(ValueError):
    """不带时区的时间无法确定时刻，一律拒绝。"""


def ensure_utc(dt: datetime) -> datetime:
    """校验时间带时区并换算为 UTC；不带时区直接报错，不猜测。"""
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise NaiveDatetimeError(f"时间缺少时区：{dt.isoformat()}")
    return dt.astimezone(UTC)


class Clock(Protocol):
    def now(self) -> datetime:
        """当前时刻，带时区，UTC。"""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """固定在某一时刻，只在显式 set / advance 时变化。"""

    def __init__(self, at: datetime) -> None:
        self._now = ensure_utc(at)

    def now(self) -> datetime:
        return self._now

    def set(self, at: datetime) -> None:
        self._now = ensure_utc(at)

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise ValueError(f"只能向前拨动时钟：{delta}")
        self._now += delta
