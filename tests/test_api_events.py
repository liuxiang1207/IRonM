"""记录提交接口（spec [2026-10-20] §10）。"""

import io
import json
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.core.clock import FixedClock
from ironm.core.config import Settings, load_settings
from ironm.store.migrate import MIGRATIONS_DIR, MigrationError
from tests.test_contracts_events import PAYLOADS

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
EID = "6f1c2a3e-1b2c-4d5e-8f90-123456789abc"


def _event(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "event_id": EID,
        "user_id": "u1",
        "type": "weight",
        "occurred_at": "2026-10-20T07:30:00+08:00",
        "recorded_at": "2026-10-20T07:31:00+08:00",
        "source": "user",
        "confirmed": True,
        "payload": PAYLOADS["weight"],
    }
    data.update(overrides)
    return data


def _settings(tmp_path: Path) -> Settings:
    return load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})


@pytest.fixture
def logs() -> io.StringIO:
    return io.StringIO()


@pytest.fixture
def client(tmp_path: Path, logs: io.StringIO) -> Iterator[TestClient]:
    app = create_app(_settings(tmp_path), FixedClock(NOW), log_stream=logs)
    with TestClient(app) as c:
        yield c


def _post(client: TestClient, data: dict[str, Any] | str) -> Any:
    body = data if isinstance(data, str) else json.dumps(data)
    return client.post("/events", content=body, headers={"Content-Type": "application/json"})


def _count(tmp_path: Path) -> int:
    with closing(sqlite3.connect(tmp_path / "db.sqlite")) as c:
        return int(c.execute("SELECT count(*) FROM events").fetchone()[0])


def test_created(client: TestClient, tmp_path: Path) -> None:
    r = _post(client, _event())
    assert r.status_code == 201
    assert r.json() == {"event_id": EID, "status": "created"}
    assert _count(tmp_path) == 1


def test_duplicate_stored_once(client: TestClient, tmp_path: Path) -> None:
    _post(client, _event())
    r = _post(client, _event())
    assert r.status_code == 200
    assert r.json() == {"event_id": EID, "status": "duplicate"}
    assert _count(tmp_path) == 1


def test_conflict(client: TestClient, tmp_path: Path) -> None:
    _post(client, _event())
    r = _post(client, _event(payload={"weight_kg": 90.0}))
    assert r.status_code == 409
    assert r.json()["status"] == "conflict"
    assert _count(tmp_path) == 1


@pytest.mark.parametrize(
    "body",
    [
        _event(payload={"weight_kg": 500}),
        {k: v for k, v in _event().items() if k != "user_id"},
        _event(mood="good"),
        "not json",
        "",
    ],
)
def test_invalid_returns_422_with_location(client: TestClient, body: dict[str, Any] | str) -> None:
    r = _post(client, body)
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert errors
    assert all("loc" in e and "msg" in e for e in errors)


def test_invalid_does_not_echo_content(client: TestClient) -> None:
    r = _post(client, _event(type="symptom", payload={"text": "胸口疼", "severity": 9}))
    assert r.status_code == 422
    assert "胸口疼" not in r.text


def test_unknown_key_not_echoed(client: TestClient) -> None:
    r = _post(client, _event(**{"胸口疼": 1}))
    assert r.status_code == 422
    assert "胸口疼" not in r.text


def test_future_recorded_at(client: TestClient) -> None:
    at = (NOW + timedelta(minutes=10)).isoformat()
    r = _post(client, _event(occurred_at=at, recorded_at=at))
    assert r.status_code == 422


def test_bad_correction(client: TestClient) -> None:
    r = _post(client, _event(corrects="7a2b3c4d-5e6f-4a1b-9c2d-3e4f5a6b7c8d"))
    assert r.status_code == 422


def test_concurrent_same_event(client: TestClient, tmp_path: Path) -> None:
    codes: list[int] = []

    def submit() -> None:
        codes.append(_post(client, _event()).status_code)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(codes) == [200, 201]
    assert _count(tmp_path) == 1


def test_startup_migrates_new_database(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), FixedClock(NOW), log_stream=io.StringIO())
    with TestClient(app):
        pass
    assert _count(tmp_path) == 0  # 表已建好


def test_startup_fails_on_bad_migrations(tmp_path: Path) -> None:
    bad = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, bad)
    (bad / "notes.txt").write_text("x")
    app = create_app(
        _settings(tmp_path), FixedClock(NOW), log_stream=io.StringIO(), migrations_dir=bad
    )
    with pytest.raises(MigrationError), TestClient(app):
        pass


def test_each_submission_logged_without_content(client: TestClient, logs: io.StringIO) -> None:
    _post(client, _event(raw="早上空腹称重"))
    _post(client, _event(raw="早上空腹称重"))
    _post(client, "not json")
    lines = [json.loads(line) for line in logs.getvalue().splitlines()]
    submitted = [x for x in lines if x["event"] == "events.submitted"]
    assert [x["status"] for x in submitted] == ["created", "duplicate", "invalid"]
    assert submitted[0]["event_id"] == EID
    assert all("duration_ms" in x for x in submitted)
    assert "空腹" not in logs.getvalue()
    assert "72.4" not in logs.getvalue() and "80.4" not in logs.getvalue()


@pytest.mark.parametrize(
    "overrides",
    [
        {"raw": ["胸口疼"]},  # 出错字段的取值本身就是原文
        {"type": "symptom", "payload": {"text": ["胸口疼得厉害"]}},
    ],
)
def test_invalid_value_itself_not_echoed(client: TestClient, overrides: dict[str, Any]) -> None:
    r = _post(client, _event(**overrides))
    assert r.status_code == 422
    assert "胸口疼" not in r.text
