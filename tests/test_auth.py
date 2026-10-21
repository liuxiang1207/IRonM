"""登录（白名单）（spec [2026-10-21] §14）：一次性设备码 → 设备凭证；撤销、停用、限流。"""

import io
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.api.cli import main as cli_main
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.store.auth import AuthStore
from ironm.store.db import connect

NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)


class _App:
    def __init__(self, tmp_path: Path, env: str) -> None:
        self.db = tmp_path / "db.sqlite"
        self.clock = FixedClock(NOW)
        self.log = io.StringIO()
        settings = load_settings({"IRONM_DB_PATH": str(self.db), "IRONM_ENV": env})
        self.app = create_app(settings, self.clock, log_stream=self.log, environ={})

    def store(self, conn: sqlite3.Connection) -> AuthStore:
        return AuthStore(conn, self.clock)

    def new_user(self, name: str = "me") -> tuple[str, str]:
        with closing(connect(self.db)) as conn:
            store = self.store(conn)
            user = store.add_user(name)
            return user, store.new_code(user)


@pytest.fixture
def prod(tmp_path: Path) -> Iterator[tuple[_App, TestClient]]:
    app = _App(tmp_path, "prod")
    with TestClient(app.app) as client:
        yield app, client


def _login(client: TestClient, code: str, device: str = "iPhone") -> object:
    return client.post("/auth/login", json={"code": code, "device": device})


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_code_login_then_token_works(prod: tuple[_App, TestClient]) -> None:
    app, client = prod
    user, code = app.new_user()
    resp = _login(client, code)
    assert resp.status_code == 200  # type: ignore[attr-defined]
    body = resp.json()  # type: ignore[attr-defined]
    assert body["user_id"] == user
    assert client.get("/profile", headers=_auth(body["token"])).status_code == 200


def test_code_accepts_lowercase_and_spaces(prod: tuple[_App, TestClient]) -> None:
    app, client = prod
    _, code = app.new_user()
    sloppy = " " + code.lower().replace("-", " ") + " "
    assert _login(client, sloppy).status_code == 200  # type: ignore[attr-defined]


def test_code_is_single_use_and_expires(prod: tuple[_App, TestClient]) -> None:
    app, client = prod
    _, code = app.new_user()
    assert _login(client, code).status_code == 200  # type: ignore[attr-defined]
    assert _login(client, code).status_code == 401  # type: ignore[attr-defined]
    _, late = app.new_user("other")
    app.clock.set(NOW + timedelta(hours=24, seconds=1))
    assert _login(client, late).status_code == 401  # type: ignore[attr-defined]


def test_revoke_one_device_and_disable_user(prod: tuple[_App, TestClient]) -> None:
    app, client = prod
    user, code = app.new_user()
    phone = _login(client, code, "iPhone").json()["token"]  # type: ignore[attr-defined]
    with closing(connect(app.db)) as conn:
        second = app.store(conn).new_code(user)
    pad = _login(client, second, "iPad").json()["token"]  # type: ignore[attr-defined]
    devices = client.get("/auth/devices", headers=_auth(phone)).json()
    assert sorted(d["name"] for d in devices) == ["iPad", "iPhone"]
    [ipad] = [d for d in devices if d["name"] == "iPad"]
    gone = client.delete(f"/auth/devices/{ipad['device_id']}", headers=_auth(phone))
    assert gone.status_code == 204
    assert client.get("/profile", headers=_auth(pad)).status_code == 401
    assert client.get("/profile", headers=_auth(phone)).status_code == 200
    with closing(connect(app.db)) as conn:
        app.store(conn).disable(user)
    assert client.get("/profile", headers=_auth(phone)).status_code == 401


def test_cannot_revoke_someone_elses_device(prod: tuple[_App, TestClient]) -> None:
    app, client = prod
    _, code_a = app.new_user("a")
    _, code_b = app.new_user("b")
    token_a = _login(client, code_a).json()["token"]  # type: ignore[attr-defined]
    token_b = _login(client, code_b).json()["token"]  # type: ignore[attr-defined]
    [device_b] = client.get("/auth/devices", headers=_auth(token_b)).json()
    resp = client.delete(f"/auth/devices/{device_b['device_id']}", headers=_auth(token_a))
    assert resp.status_code == 404
    assert client.get("/profile", headers=_auth(token_b)).status_code == 200


def test_prod_rejects_user_header_dev_accepts(tmp_path: Path) -> None:
    prod_app = _App(tmp_path / "p", "prod")
    with TestClient(prod_app.app) as client:
        assert client.get("/profile", headers={"X-IronM-User": "u1"}).status_code == 401
        assert client.get("/profile").status_code == 401
        assert client.get("/profile", headers=_auth("nonsense")).status_code == 401
    dev_app = _App(tmp_path / "d", "dev")
    with TestClient(dev_app.app) as client:
        assert client.get("/profile", headers={"X-IronM-User": "u1"}).status_code == 200


def test_secrets_never_stored_or_logged(prod: tuple[_App, TestClient]) -> None:
    app, client = prod
    _, code = app.new_user()
    token = _login(client, code).json()["token"]  # type: ignore[attr-defined]
    client.get("/profile", headers=_auth(token))
    raw = code.replace("-", "")
    with closing(connect(app.db)) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        dump = "\n".join(
            str(tuple(row)) for t in tables for row in conn.execute(f"SELECT * FROM {t}")
        )
    assert raw not in dump and token not in dump
    assert raw not in app.log.getvalue() and token not in app.log.getvalue()


def test_too_many_wrong_codes_locks_login_for_ten_minutes(prod: tuple[_App, TestClient]) -> None:
    app, client = prod
    _, code = app.new_user()
    for _ in range(10):
        assert _login(client, "WRONG-CODE1").status_code == 401  # type: ignore[attr-defined]
    assert _login(client, code).status_code == 429  # type: ignore[attr-defined]
    app.clock.set(NOW + timedelta(minutes=10, seconds=1))
    assert _login(client, code).status_code == 200  # type: ignore[attr-defined]


def test_cli_adds_user_and_prints_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "db.sqlite"  # 与 _App 用同一个库
    env = {"IRONM_DB_PATH": str(db), "IRONM_ENV": "prod"}
    assert cli_main(["add-user", "我"], environ=env, clock=FixedClock(NOW)) == 0
    out = capsys.readouterr().out
    code = out.strip().splitlines()[-1].strip()  # 设备码单独一行
    app = _App(tmp_path, "prod")
    with TestClient(app.app) as client:
        assert _login(client, code).status_code == 200  # type: ignore[attr-defined]
    assert cli_main(["users"], environ=env, clock=FixedClock(NOW)) == 0
    assert "我" in capsys.readouterr().out
