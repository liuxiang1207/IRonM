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


# ---- 批量与离线补交（spec §11）----

E2 = "7a2b3c4d-5e6f-4a1b-9c2d-3e4f5a6b7c8d"
E3 = "8b3c4d5e-6f7a-4b2c-8d3e-4f5a6b7c8d9e"


def _batch(client: TestClient, events: list[Any] | Any) -> Any:
    body = json.dumps({"events": events} if isinstance(events, list) else events)
    return client.post("/events/batch", content=body, headers={"Content-Type": "application/json"})


def test_batch_all_created(client: TestClient, tmp_path: Path) -> None:
    r = _batch(client, [_event(), _event(event_id=E2), _event(event_id=E3)])
    assert r.status_code == 200
    assert [(x["index"], x["event_id"], x["status"]) for x in r.json()["results"]] == [
        (0, EID, "created"),
        (1, E2, "created"),
        (2, E3, "created"),
    ]
    assert _count(tmp_path) == 3


def test_batch_duplicates(client: TestClient, tmp_path: Path) -> None:
    _post(client, _event(event_id=E3))
    r = _batch(client, [_event(), _event(), _event(event_id=E3)])
    assert [x["status"] for x in r.json()["results"]] == ["created", "duplicate", "duplicate"]
    assert _count(tmp_path) == 2


def test_batch_one_invalid_others_stored(client: TestClient, tmp_path: Path) -> None:
    bad = _event(event_id=E2, raw=["胸口疼"])
    r = _batch(client, [_event(), bad, _event(event_id=E3)])
    results = r.json()["results"]
    assert [x["status"] for x in results] == ["created", "invalid", "created"]
    assert results[1]["errors"][0]["loc"]
    assert "胸口疼" not in r.text
    assert _count(tmp_path) == 2


def test_batch_correction_before_original(client: TestClient, tmp_path: Path) -> None:
    fix = _event(event_id=E2, corrects=EID, payload={"weight_kg": 81.0})
    r = _batch(client, [fix, _event()])
    assert [x["status"] for x in r.json()["results"]] == ["created", "created"]


@pytest.mark.parametrize(
    "body",
    [[], [_event()] * 501, ["not", "objects"], {"items": [_event()]}, {"events": "x"}],
)
def test_batch_rejected_as_a_whole(client: TestClient, tmp_path: Path, body: Any) -> None:
    if isinstance(body, list) and body and isinstance(body[0], str):
        r = client.post("/events/batch", content=json.dumps(body))
    else:
        r = _batch(client, body)
    assert r.status_code == 422
    assert _count(tmp_path) == 0


def test_offline_backfill(client: TestClient, tmp_path: Path) -> None:
    old = (NOW - timedelta(days=3)).isoformat()
    recent = _event(event_id=E2, occurred_at="2026-10-20T07:30:00+08:00")
    r = _batch(client, [recent, _event(occurred_at=old, recorded_at=old)])
    assert [x["status"] for x in r.json()["results"]] == ["created", "created"]
    with closing(sqlite3.connect(tmp_path / "db.sqlite")) as c:
        rows = c.execute("SELECT event_id, received_at FROM events ORDER BY occurred_at").fetchall()
    assert [row[0] for row in rows] == [EID, E2]
    assert {row[1] for row in rows} == {"2026-10-20T12:00:00.000000Z"}


def test_batch_logging(client: TestClient, logs: io.StringIO) -> None:
    _batch(client, [_event(raw="早上空腹"), _event(event_id=E2, raw=["胸口疼"])])
    lines = [json.loads(line) for line in logs.getvalue().splitlines()]
    per_item = [x for x in lines if x["event"] == "events.submitted"]
    batch = [x for x in lines if x["event"] == "events.batch_submitted"]
    assert [x["status"] for x in per_item] == ["created", "invalid"]
    assert len(batch) == 1 and batch[0]["count"] == 2 and "duration_ms" in batch[0]
    assert "空腹" not in logs.getvalue() and "胸口疼" not in logs.getvalue()
