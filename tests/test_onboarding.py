"""建档、计划版本与今日任务（spec [2026-10-21] §15）。"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.store.db import connect
from ironm.store.plans import PlanStore, VersionConflict

NOW = datetime(2026, 10, 21, 1, tzinfo=UTC)  # 北京时间 10 月 21 日（周三）上午 9 点

ADULT: dict[str, Any] = {
    "birth_date": "1992-03-01",
    "sex": "male",
    "height_cm": 178.0,
    "weight_kg": 80.0,
    "waist_cm": 86.0,
    "conditions": [],
    "goal": "fitness",
    "environment": "gym",
    "available_days": 5,
    "training_level": "intermediate",
    "activity": "moderate",
    "restrictions": [],
}


class _Env:
    def __init__(self, tmp_path: Path) -> None:
        self.db = tmp_path / "db.sqlite"
        self.clock = FixedClock(NOW)
        settings = load_settings({"IRONM_DB_PATH": str(self.db)})
        self.app = create_app(settings, self.clock, environ={})


@pytest.fixture
def env(tmp_path: Path) -> Iterator[tuple[_Env, TestClient]]:
    e = _Env(tmp_path)
    with TestClient(e.app, headers={"X-IronM-User": "u1"}) as client:
        yield e, client


def _onboard(client: TestClient, **changes: Any) -> Any:
    body = {**ADULT, **changes}
    body = {k: v for k, v in body.items() if v is not None}
    return client.post("/onboarding", json=body)


def test_fitness_goal_gets_training_plan_without_calorie_target(
    env: tuple[_Env, TestClient],
) -> None:
    _, client = env
    resp = _onboard(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "plan" and body["tier"] in ("a", "a_prime")
    fields = body["plan"]["fields"]
    assert {"training_days_per_week", "session_slots", "exercise_selection", "diet_mode"} <= set(
        fields
    )
    assert "daily_kcal_target" not in fields
    assert "training_schedule" in fields  # 每周哪几天练


def test_fat_loss_goal_gets_calorie_target(env: tuple[_Env, TestClient]) -> None:
    _, client = env
    fields = _onboard(client, goal="fat_loss").json()["plan"]["fields"]
    assert isinstance(fields["daily_kcal_target"], int)


def test_missing_answer_asks_then_plans(env: tuple[_Env, TestClient]) -> None:
    _, client = env
    first = _onboard(client, birth_date=None).json()
    assert first["status"] == "questions"
    asked = {q["question_id"]: q for q in first["questions"]}
    assert "profile.birth_date" in asked and asked["profile.birth_date"]["text"]
    second = _onboard(client, birth_date=None, answers={"profile.birth_date": "1992-03-01"}).json()
    assert second["status"] == "plan"


def test_pregnant_cannot_get_plan(env: tuple[_Env, TestClient]) -> None:
    _, client = env
    body = _onboard(client, sex="female", conditions=["pregnant"]).json()
    assert body["status"] == "not_allowed" and body["notices"] and body.get("plan") is None
    assert client.get("/plan").json() == {"status": "none"}


def test_plan_saved_with_cards_and_history(env: tuple[_Env, TestClient]) -> None:
    _, client = env
    made = _onboard(client).json()["plan"]
    got = client.get("/plan").json()
    assert got["status"] == "ok" and got["plan"]["version"] == 1
    assert got["plan"]["fields"] == made["fields"]
    carded = {c["field"] for c in got["plan"]["cards"]}
    assert set(got["plan"]["fields"]) <= carded  # 每一项都有依据卡
    [v1] = client.get("/plan/history").json()
    assert v1["version"] == 1 and v1["reason"] == "onboarding"
    assert {c["field"] for c in v1["changes"]} == set(made["fields"])


def test_reonboarding_makes_new_version_with_diff(env: tuple[_Env, TestClient]) -> None:
    _, client = env
    _onboard(client)
    again = _onboard(client, available_days=2).json()["plan"]
    assert again["version"] == 2
    history = client.get("/plan/history").json()
    assert [h["version"] for h in history] == [1, 2]
    changed = {c["field"] for c in history[1]["changes"]}
    assert "training_days_per_week" in changed or "training_schedule" in changed
    assert all(c["from"] != c["to"] for c in history[1]["changes"])


def _local_weekday(when: datetime, tz: str) -> int:
    from zoneinfo import ZoneInfo

    return when.astimezone(ZoneInfo(tz)).weekday()


def test_today_follows_schedule_and_timezone(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    plan = _onboard(client).json()["plan"]["fields"]
    schedule = set(plan["training_schedule"])
    for offset in range(7):
        e.clock.set(NOW + timedelta(days=offset))
        today = client.get("/plan/today", params={"tz": "Asia/Shanghai"}).json()
        weekday = _local_weekday(NOW + timedelta(days=offset), "Asia/Shanghai")
        assert (
            today["date"]
            == (NOW + timedelta(days=offset))
            .astimezone(__import__("zoneinfo").ZoneInfo("Asia/Shanghai"))
            .date()
            .isoformat()
        )
        training = today["training"]
        assert training["scheduled"] is (weekday in schedule)
        if training["scheduled"]:
            assert training["items"] and all(i["sets"] >= 1 for i in training["items"])
            assert all(i["exercise"] and i["key_points"] for i in training["items"])
        else:
            assert training["items"] == []
        assert today["weigh_in_today"] is (weekday in plan["weigh_in_days"])
        assert today["checkin_today"] is (weekday == plan["checkin_day"])


def test_today_uses_users_timezone(env: tuple[_Env, TestClient]) -> None:
    """同一时刻，北京已经是周三，洛杉矶还是周二。"""
    _, client = env
    _onboard(client)
    shanghai = client.get("/plan/today", params={"tz": "Asia/Shanghai"}).json()["date"]
    la = client.get("/plan/today", params={"tz": "America/Los_Angeles"}).json()["date"]
    assert date.fromisoformat(shanghai) - date.fromisoformat(la) == timedelta(days=1)


def test_today_without_plan(env: tuple[_Env, TestClient]) -> None:
    _, client = env
    assert client.get("/plan/today", params={"tz": "Asia/Shanghai"}).json()["status"] == "none"


def test_concurrent_writes_keep_one_new_version(tmp_path: Path) -> None:
    e = _Env(tmp_path)
    with TestClient(e.app):
        pass
    with closing(connect(e.db)) as a, closing(connect(e.db)) as b:
        assert PlanStore(a).add("u1", 0, {"x": 1}, "onboarding", NOW) == 1
        with pytest.raises(VersionConflict):
            PlanStore(b).add("u1", 0, {"x": 2}, "onboarding", NOW)
        assert [p.version for p in PlanStore(b).history("u1")] == [1]


def test_plan_versions_append_only(tmp_path: Path) -> None:
    e = _Env(tmp_path)
    with TestClient(e.app):
        pass
    with closing(connect(e.db)) as conn:
        PlanStore(conn).add("u1", 0, {"x": 1}, "onboarding", NOW)
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE plan_versions SET reason = 'x'")


def test_conflict_reported_to_client(
    env: tuple[_Env, TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """建档写入时发现别的请求刚写了新版本：返回 409，不覆盖。"""
    _, client = env

    def conflict(*args: object, **kwargs: object) -> int:
        raise VersionConflict("当前版本是 1，不是 0")

    monkeypatch.setattr(PlanStore, "add", conflict)
    resp = _onboard(client)
    assert resp.status_code == 409 and resp.json() == {"status": "conflict"}
