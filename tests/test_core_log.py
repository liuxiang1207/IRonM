"""结构化日志（spec [2026-10-20] §2）。"""

import io
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.core.ids import SeededIdGenerator
from ironm.core.log import LogFieldError, configure_logging, get_logger

T0 = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    buf = io.StringIO()
    configure_logging(load_settings({}), FixedClock(T0), buf)
    yield buf


def _lines(buf: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buf.getvalue().splitlines()]


def test_one_json_line_per_record(stream: io.StringIO) -> None:
    get_logger("ironm.test").info("event.recorded", event_id="e1", count=2)
    [rec] = _lines(stream)
    assert rec == {
        "ts": "2026-10-20T08:00:00+00:00",
        "level": "info",
        "logger": "ironm.test",
        "event": "event.recorded",
        "event_id": "e1",
        "count": 2,
    }


@pytest.mark.parametrize("event", ["user weighed 80kg", "称重", "Event", "", "a..b"])
def test_free_text_event_rejected(stream: io.StringIO, event: str) -> None:
    with pytest.raises(LogFieldError):
        get_logger("ironm.test").info(event)
    assert stream.getvalue() == ""


@pytest.mark.parametrize("field", ["weight", "message", "payload", "体重"])
def test_disallowed_field_rejected(stream: io.StringIO, field: str) -> None:
    with pytest.raises(LogFieldError, match=field):
        get_logger("ironm.test").info("event.recorded", **{field: 80})
    assert stream.getvalue() == ""


def test_non_scalar_value_rejected(stream: io.StringIO) -> None:
    with pytest.raises(LogFieldError, match="user_id"):
        get_logger("ironm.test").info("event.recorded", user_id={"weight": 80})  # type: ignore[arg-type]


def test_exception_logs_type_only(stream: io.StringIO) -> None:
    try:
        raise ValueError("体重 83.4 kg 超出范围")
    except ValueError as exc:
        get_logger("ironm.test").exception("event.rejected", exc, event_id="e1")
    out = stream.getvalue()
    assert "83.4" not in out
    [rec] = _lines(stream)
    assert rec["level"] == "error"
    assert rec["error_type"] == "ValueError"


def test_level_filter() -> None:
    buf = io.StringIO()
    configure_logging(load_settings({"IRONM_LOG_LEVEL": "warning"}), FixedClock(T0), buf)
    log = get_logger("ironm.test")
    log.info("event.recorded")
    log.warning("event.late")
    assert [r["event"] for r in _lines(buf)] == ["event.late"]


def test_timestamp_follows_clock() -> None:
    buf = io.StringIO()
    clock = FixedClock(T0)
    configure_logging(load_settings({}), clock, buf)
    clock.advance(timedelta(days=1))
    get_logger("ironm.test").info("tick")
    assert _lines(buf)[0]["ts"] == "2026-10-21T08:00:00+00:00"


def test_reconfigure_does_not_duplicate() -> None:
    first, second = io.StringIO(), io.StringIO()
    configure_logging(load_settings({}), FixedClock(T0), first)
    configure_logging(load_settings({}), FixedClock(T0), second)
    get_logger("ironm.test").info("tick")
    assert first.getvalue() == ""
    assert len(_lines(second)) == 1


def _run_flow(seed: int) -> str:
    buf = io.StringIO()
    clock = FixedClock(T0)
    configure_logging(load_settings({}), clock, buf)
    ids = SeededIdGenerator(seed)
    log = get_logger("ironm.flow")
    for _ in range(3):
        log.info("event.recorded", event_id=ids.new_id(), user_id="u1")
        clock.advance(timedelta(minutes=5))
    return buf.getvalue()


def test_flow_reproducible() -> None:
    assert _run_flow(42) == _run_flow(42)
    assert _run_flow(42) != _run_flow(43)


# ---- 独立审查后补充：取值受限、标准日志接口不泄露 ----


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("user_id", "今天胸口疼得厉害"),
        ("event_id", "has space"),
        ("status", "体重 92kg"),
        ("path", "/x?note=头晕"),
        ("error_type", "ValueError: 体重 80"),
        ("method", "DELETE everything"),
    ],
)
def test_free_text_values_rejected(stream: io.StringIO, field: str, value: str) -> None:
    with pytest.raises(LogFieldError, match=field):
        get_logger("ironm.test").info("event.recorded", **{field: value})
    assert stream.getvalue() == ""


def test_identifier_values_accepted(stream: io.StringIO) -> None:
    get_logger("ironm.test").info(
        "http.request",
        user_id="u_1",
        event_id="6f1c2a3e-1b2c-4d5e-8f90-123456789abc",
        status="ok",
        path="/events/batch",
        method="POST",
        error_type="sqlite3.IntegrityError",
        status_code=200,
        duration_ms=12.5,
    )
    assert len(_lines(stream)) == 1


def test_stdlib_logging_does_not_leak(stream: io.StringIO) -> None:
    logging.getLogger("ironm.y").warning("用户说胸口很痛 %s", "体重 80")
    out = stream.getvalue()
    assert "胸口" not in out
    assert "体重" not in out
    [rec] = _lines(stream)
    assert rec["event"] == "log.rejected_unstructured"
    assert rec["logger"] == "ironm.y"
