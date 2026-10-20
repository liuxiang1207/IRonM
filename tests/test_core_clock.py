"""时间来源（spec [2026-10-20] §2）。"""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from ironm.core.clock import FixedClock, NaiveDatetimeError, SystemClock

T0 = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)


def test_fixed_clock_is_stable() -> None:
    clock = FixedClock(T0)
    assert clock.now() == clock.now() == T0


def test_fixed_clock_advance() -> None:
    clock = FixedClock(T0)
    clock.advance(timedelta(days=3))
    assert clock.now() - T0 == timedelta(days=3)


def test_fixed_clock_set() -> None:
    clock = FixedClock(T0)
    later = T0 + timedelta(hours=5)
    clock.set(later)
    assert clock.now() == later


def test_fixed_clock_normalizes_to_utc() -> None:
    beijing = timezone(timedelta(hours=8))
    clock = FixedClock(datetime(2026, 10, 20, 16, 0, tzinfo=beijing))
    assert clock.now() == T0
    assert clock.now().tzinfo is UTC


@pytest.mark.parametrize("bad", ["init", "set"])
def test_fixed_clock_rejects_naive(bad: str) -> None:
    naive = datetime(2026, 10, 20, 8, 0)
    with pytest.raises(NaiveDatetimeError):
        if bad == "init":
            FixedClock(naive)
        else:
            FixedClock(T0).set(naive)


def test_fixed_clock_cannot_go_backwards() -> None:
    with pytest.raises(ValueError):
        FixedClock(T0).advance(timedelta(seconds=-1))


def test_system_clock_is_utc_aware() -> None:
    now = SystemClock().now()
    assert now.tzinfo is UTC
