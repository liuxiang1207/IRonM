"""网页版（spec [2026-10-21] §19）：服务端直接提供网页；网页只调用已有接口，不引用外部资源。"""

import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import WEB_DIR, create_app
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings

NOW = datetime(2026, 10, 21, 1, tzinfo=UTC)


def _client(tmp_path: Path, env: str = "dev") -> TestClient:
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite"), "IRONM_ENV": env})
    return TestClient(create_app(settings, FixedClock(NOW), environ={}))


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _client(tmp_path) as c:
        yield c


def test_home_redirects_to_web_page(client: TestClient) -> None:
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307) and resp.headers["location"] == "/web/"
    page = client.get("/web/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "<title>IronM</title>" in page.text
    for asset in ("app.js", "style.css"):
        assert client.get(f"/web/{asset}").status_code == 200


def test_dev_user_login_only_outside_production(tmp_path: Path) -> None:
    with _client(tmp_path / "dev") as dev:
        assert dev.get("/auth/options").json() == {"dev_user_login": True}
    with _client(tmp_path / "prod", env="prod") as prod:
        assert prod.get("/auth/options").json() == {"dev_user_login": False}


def _api_paths() -> set[str]:
    """网页脚本里调用的接口路径（去掉查询参数，路径参数记为 {}）。"""
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    found = set()
    for raw in re.findall(r"api\(\s*[`'\"]([^`'\"]+)[`'\"]", script):
        path = raw.split("?")[0]
        found.add(re.sub(r"\$\{[^}]+\}", "{}", path))
    return found


def test_web_calls_only_existing_endpoints(client: TestClient) -> None:
    used = _api_paths()
    assert {"/auth/login", "/onboarding", "/plan/today", "/chat", "/metrics", "/plan"} <= used
    routes = {re.sub(r"\{[^}]+\}", "{}", getattr(r, "path", "")) for r in client.app.routes}  # type: ignore[attr-defined]
    assert sorted(used - routes) == []


def test_web_page_loads_nothing_from_other_sites() -> None:
    for f in WEB_DIR.iterdir():
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"https?://|//cdn|src=\"//", text), f.name


def test_onboarding_never_defaults_health_conditions_to_none() -> None:
    """健康状况必须由用户自己勾选（含“都没有”），网页不能替用户默认填“没有”。"""
    script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"conditions\s*:\s*\[\s*\]", script)
    assert '"都没有"' in script


def test_onboarding_explains_why_no_plan(client: TestClient) -> None:
    """不能给计划时，建档结果带上审定过的说明文字，网页照着显示（不只是代号）。"""
    minor = {"birth_date": "2015-01-01", "sex": "male", "height_cm": 150, "weight_kg": 45,
             "conditions": [], "goal": "fitness"}  # fmt: skip
    r = client.post("/onboarding", json=minor, headers={"X-IronM-User": "kid"}).json()
    assert r["status"] == "not_allowed"
    assert any("未满 18 岁" in t for t in r["notice_texts"])
