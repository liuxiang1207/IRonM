"""定时任务、每周复盘与计划调整（spec [2026-10-21] §16、§17）：端到端。"""

import io
import json
import uuid
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.store.db import connect
from ironm.store.jobs import JobStore
from ironm.store.reviews import Adjustment, DecisionStore

SH = ZoneInfo("Asia/Shanghai")
START = datetime(2026, 10, 21, 1, tzinfo=UTC)  # 北京时间 10 月 21 日（周三）9 点建档
W1 = date(2026, 10, 26)  # 建档后第一个完整周的周一

ADULT: dict[str, Any] = {
    "birth_date": "1992-03-01", "sex": "male", "height_cm": 178.0, "weight_kg": 80.0,
    "waist_cm": 86.0, "conditions": [], "goal": "fitness", "environment": "gym",
    "available_days": 5, "training_level": "intermediate", "activity": "moderate",
    "restrictions": [],
}  # fmt: skip


class _Env:
    def __init__(self, tmp_path: Path) -> None:
        self.db = tmp_path / "db.sqlite"
        self.clock = FixedClock(START)
        self.log = io.StringIO()
        settings = load_settings({"IRONM_DB_PATH": str(self.db)})
        self.app = create_app(settings, self.clock, log_stream=self.log, environ={})

    def at(self, day: date, hour: int = 20, minute: int = 0) -> datetime:
        return datetime(day.year, day.month, day.day, hour, minute, tzinfo=SH)

    def tick(self, when: datetime) -> Any:
        self.clock.set(when.astimezone(UTC))
        return self.app.state.tick()


@pytest.fixture
def env(tmp_path: Path) -> Iterator[tuple[_Env, TestClient]]:
    e = _Env(tmp_path)
    with TestClient(e.app, headers={"X-IronM-User": "u1"}) as client:
        assert client.post("/onboarding", json=ADULT).json()["status"] == "plan"
        yield e, client


def _post(
    client: TestClient, kind: str, at: datetime, payload: dict[str, Any], **extra: Any
) -> None:
    body = {
        "event_id": str(uuid.uuid4()),
        "user_id": "u1",
        "type": kind,
        "occurred_at": at.isoformat(),
        "recorded_at": extra.pop("recorded_at", at).isoformat(),
        "source": "user",
        "confirmed": True,
        "payload": payload,
        **extra,
    }
    resp = client.post("/events", content=json.dumps(body))
    assert resp.status_code in (200, 201), resp.text


def _train_week(
    e: _Env,
    client: TestClient,
    monday: date,
    *,
    status: str = "done",
    rpe: float | None = 6,
    checkin: dict[str, Any] | None = None,
) -> None:
    """按计划（周一、三、五）打卡并记组；周日签到。"""
    for offset in (0, 2, 4):
        day = monday + timedelta(days=offset)
        e.clock.set(e.at(day, 21).astimezone(UTC))
        _post(client, "workout_check", e.at(day), {"status": status})
        if rpe is not None:
            payload = {"exercise_id": "sq_goblet", "set_index": 1, "reps": 10, "load_kg": 16,
                       "rpe": rpe}  # fmt: skip
            _post(client, "set_log", e.at(day, 19), payload)
    if checkin is not None:
        sunday = monday + timedelta(days=6)
        e.clock.set(e.at(sunday, 21).astimezone(UTC))
        _post(client, "checkin", e.at(sunday), checkin)


def _sets(plan: dict[str, Any]) -> list[int]:
    return [s["sets"] for s in plan["fields"]["session_slots"]]


GOOD_CHECKIN = {"fatigue": 2, "sleep_quality": 4, "record_completeness": "full"}


# ---- 复盘：进阶、恢复、数据不足 ----


def test_good_week_progresses_after_checkin(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    before = client.get("/plan").json()["plan"]
    _train_week(e, client, W1, checkin=GOOD_CHECKIN)
    e.tick(e.at(W1 + timedelta(days=6), 21, 30))  # 签到之后
    after = client.get("/plan").json()["plan"]
    assert after["version"] == before["version"] + 1
    assert _sets(after) == [3] * 5  # 先加组
    assert (
        after["fields"]["weekly_cardio_minutes"] == before["fields"]["weekly_cardio_minutes"] + 30
    )
    [latest, *_] = client.get("/plan/decisions").json()
    assert latest["kind"] == "weekly" and latest["period"] == W1.isoformat()
    assert latest["decision"] == "adjusted" and latest["summary"].startswith("这次调整了")
    changed = {c["field"] for c in latest["changes"]}
    assert changed == {"session_slots", "weekly_cardio_minutes"}


@pytest.mark.parametrize(
    ("plant", "stage"),
    [
        ({"sets": 10}, "inconsistent"),  # 每个动作 10 组：整合关拦下
        ({"rule_id": "R-NOT-THERE"}, "evidence"),  # 引用不存在的规则：盲审关拦下
    ],
)
def test_planted_bad_change_blocked_good_one_published_and_logged(
    env: tuple[_Env, TestClient],
    monkeypatch: pytest.MonkeyPatch,
    plant: dict[str, Any],
    stage: str,
) -> None:
    """复盘时植入一项错误的修改：被拦下、不发布；另一项照常发布；决策日志写明没发布的修改与原因。"""
    from ironm.contracts.agents import ProposalOutput
    from ironm.contracts.roles import AgentId
    from ironm.orchestrator import reviews

    real_run = reviews._run

    def planted(agent: AgentId, *args: Any) -> Any:
        outputs = real_run(agent, *args)
        if agent is not AgentId.TRAINING_PLANNER:
            return outputs
        bad = []
        for o in outputs:
            if isinstance(o, ProposalOutput):
                items = tuple(
                    _plant(p, plant) if p.field.value == "session_slots" else p for p in o.items
                )
                o = o.model_copy(update={"items": items})
            bad.append(o)
        return tuple(bad)

    monkeypatch.setattr(reviews, "_run", planted)
    e, client = env
    before = client.get("/plan").json()["plan"]
    _train_week(e, client, W1, checkin=GOOD_CHECKIN)
    e.tick(e.at(W1 + timedelta(days=6), 21, 30))
    after = client.get("/plan").json()["plan"]
    assert after["version"] == before["version"] + 1
    assert _sets(after) == _sets(before)  # 错误的修改没有发布
    assert after["fields"]["weekly_cardio_minutes"] > before["fields"]["weekly_cardio_minutes"]
    [latest, *_] = client.get("/plan/decisions").json()
    assert {c["field"] for c in latest["changes"]} == {"weekly_cardio_minutes"}
    assert "not_published" not in latest  # 给用户看的版本只说改了什么、为什么、何时再看
    with closing(connect(e.db)) as conn:  # 完整的决策日志记下没发布的修改与原因
        [full] = DecisionStore(conn).recent("u1", kind="weekly", limit=1)
    blocked = {n["field"]: n["reason"] for n in full.record["not_published"]}
    assert blocked.get("session_slots", "").startswith(stage)


def _plant(p: Any, plant: dict[str, Any]) -> Any:
    if "sets" in plant:
        return p.model_copy(update={"to_value": [{**s, **plant} for s in p.to_value]})
    return p.model_copy(update=plant)


def test_diet_review_every_other_week_only_for_fat_loss(env: tuple[_Env, TestClient]) -> None:
    """减脂用户每满两周做一次饮食复盘（与当周训练复盘合在一起）；非减脂用户没有饮食复盘。"""
    e, client = env
    fat_loss = {**ADULT, "goal": "fat_loss"}
    resp = client.post("/onboarding", json=fat_loss, headers={"X-IronM-User": "u2"})
    assert resp.json()["status"] == "plan"
    e.tick(e.at(W1 + timedelta(days=14), 0, 30))  # 建档后两个完整周都结束
    with closing(connect(e.db)) as conn:
        store = DecisionStore(conn)
        loser = {d.period: d.record["diet_review"] for d in store.recent("u2", kind="weekly")}
        fit = {d.period: d.record["diet_review"] for d in store.recent("u1", kind="weekly")}
    second, third = W1.isoformat(), (W1 + timedelta(days=7)).isoformat()
    assert loser[second] is True and loser[third] is False  # 建档那周算第 1 周
    assert fit and not any(fit.values())
    # 决策日志写全饮食复盘用到的事实，复盘可回放、可核对
    with closing(connect(e.db)) as conn:
        [record] = [
            d.record for d in DecisionStore(conn).recent("u2", kind="weekly") if d.period == second
        ]
    assert set(record["diet_facts"]) == {
        "red_flag", "meal_days_min", "weigh_ins_min", "loss_kg_per_week", "ci_half_width",
        "completion", "adherence_days", "prev_slow", "days_since_start", "prev_loss_kg_per_week",
    }  # fmt: skip


def test_checkin_submitted_minutes_later_still_counts(env: tuple[_Env, TestClient]) -> None:
    """签到 21:00 填写、21:02 才提交：复盘要用上这次签到（很累、睡得差 → 恢复性下调）。
    快进模拟发现：此前以签到的发生时刻为截止，签到自己被排除，“疲劳高或睡眠差”永远不会命中。"""
    e, client = env
    before = client.get("/plan").json()["plan"]
    _train_week(e, client, W1)
    sunday = W1 + timedelta(days=6)
    e.clock.set(e.at(sunday, 21, 2).astimezone(UTC))
    _post(client, "checkin", e.at(sunday, 21), {"fatigue": 5, "sleep_quality": 1},
          recorded_at=e.at(sunday, 21, 2))  # fmt: skip
    e.tick(e.at(sunday, 21, 30))
    with closing(connect(e.db)) as conn:
        [record] = [
            d.record
            for d in DecisionStore(conn).recent("u1", kind="weekly")
            if d.period == W1.isoformat()
        ]
    this = record["facts"]["this_week"]
    assert (this["fatigue"], this["sleep_quality"]) == (5, 1)
    after = client.get("/plan").json()["plan"]
    assert after["fields"]["weekly_cardio_minutes"] < before["fields"]["weekly_cardio_minutes"]


def test_three_tired_weeks_in_a_row_stop_cutting_and_suggest_rest(
    env: tuple[_Env, TestClient],
) -> None:
    """连续 3 周睡得差：前两周恢复性下调，第 3 周不再减量，提示先休息、持续不缓解请就医。"""
    e, client = env
    cardio = []
    for k in range(3):
        monday = W1 + timedelta(days=7 * k)
        _train_week(e, client, monday, checkin={"fatigue": 4, "sleep_quality": 1})
        e.tick(e.at(monday + timedelta(days=6), 21, 30))
        cardio.append(client.get("/plan").json()["plan"]["fields"]["weekly_cardio_minutes"])
    assert cardio[0] > cardio[1] and cardio[2] == cardio[1]  # 第 3 周不再减
    latest = next(d for d in client.get("/plan/decisions").json() if d["kind"] == "weekly")
    assert latest["period"] == (W1 + timedelta(days=14)).isoformat()
    assert latest["changes"] == [] and "就医" in latest["summary"]


def test_tired_week_cuts_load(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    for k, fatigue in enumerate((2, 2)):  # 前两次签到的疲劳都是 2
        _train_week(e, client, W1 + timedelta(days=7 * k), status="half", rpe=7,
                    checkin={"fatigue": fatigue, "sleep_quality": 4})  # fmt: skip
    tired = W1 + timedelta(days=14)
    _train_week(e, client, tired, status="half", rpe=7, checkin={"fatigue": 5, "sleep_quality": 3})
    e.tick(e.at(tired + timedelta(days=7), 0, 30))
    plan = client.get("/plan").json()["plan"]
    assert _sets(plan) == [1] * 5
    decisions = client.get("/plan/decisions").json()
    assert decisions[0]["period"] == tired.isoformat() and decisions[0]["decision"] == "adjusted"


def test_sparse_records_mean_no_change(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    e.clock.set(e.at(W1, 21).astimezone(UTC))
    _post(client, "workout_check", e.at(W1), {"status": "done"})  # 3 个训练日只记了 1 天
    e.tick(e.at(W1 + timedelta(days=7), 0, 30))
    assert client.get("/plan").json()["plan"]["version"] == 1
    week = next(d for d in client.get("/plan/decisions").json() if d["period"] == W1.isoformat())
    assert week["decision"] == "insufficient" and "记录太少" in week["summary"]


# ---- 定时任务：只执行一次、补跑顺序与截止时间、失败重试 ----


def test_same_period_runs_once(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    _train_week(e, client, W1, checkin=GOOD_CHECKIN)
    first = e.tick(e.at(W1 + timedelta(days=7), 1))
    again = e.tick(e.at(W1 + timedelta(days=7), 2))
    assert first.done >= 1 and again.done == 0
    assert client.get("/plan").json()["plan"]["version"] == 2
    weekly = [d for d in client.get("/plan/decisions").json() if d["period"] == W1.isoformat()]
    assert len(weekly) == 1


def test_catch_up_in_order_with_cutoff(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    for k in range(3):
        _train_week(e, client, W1 + timedelta(days=7 * k), checkin=GOOD_CHECKIN)
    # 第一周的一条“没练”是停机期间才补记的：补跑第一周复盘时不能算进去
    late = e.at(W1 + timedelta(days=24), 10)
    e.clock.set(late.astimezone(UTC))
    _post(client, "workout_check", e.at(W1 + timedelta(days=1)), {"status": "skipped"},
          recorded_at=late)  # fmt: skip
    e.tick(e.at(W1 + timedelta(days=25)))  # 停机三周后重启
    periods = [d["period"] for d in reversed(client.get("/plan/decisions").json())]
    weekly = [p for p in periods if p >= W1.isoformat()]
    assert weekly == [(W1 + timedelta(days=7 * k)).isoformat() for k in range(3)]
    history = client.get("/plan/history").json()
    assert [h["reason"] for h in history].count("weekly_review") >= 1
    first = next(d for d in client.get("/plan/decisions").json() if d["period"] == W1.isoformat())
    assert first["decision"] == "adjusted"  # 补记的“没练”在截止之后，第一周仍算全勤


def test_catch_up_ignores_records_made_after_period_end(env: tuple[_Env, TestClient]) -> None:
    """第一周按时只记了 1 天；停机期间才补记另外 2 天。补跑第一周复盘时数据截止到周期结束：
    补记的不算，仍是“记录太少”（算进去就成了全勤）。"""
    e, client = env
    e.clock.set(e.at(W1, 21).astimezone(UTC))
    _post(client, "workout_check", e.at(W1), {"status": "done"})
    late = e.at(W1 + timedelta(days=24), 10)
    e.clock.set(late.astimezone(UTC))
    for offset in (2, 4):
        _post(client, "workout_check", e.at(W1 + timedelta(days=offset)), {"status": "done"},
              recorded_at=late)  # fmt: skip
    e.tick(e.at(W1 + timedelta(days=25)))  # 停机三周后重启
    first = next(d for d in client.get("/plan/decisions").json() if d["period"] == W1.isoformat())
    assert first["decision"] == "insufficient"


def test_failing_job_retries_then_fails_without_blocking_others(
    env: tuple[_Env, TestClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    e, client = env
    _train_week(e, client, W1, checkin=GOOD_CHECKIN)
    from ironm.orchestrator.reviews import run_review as real

    def flaky(user_id: str, week: date, **kw: Any) -> Any:
        if week == W1:
            raise RuntimeError("boom")
        return real(user_id, week, **kw)

    monkeypatch.setattr("ironm.orchestrator.jobs.run_review", flaky)
    when = e.at(W1 + timedelta(days=7), 1)
    results = [e.tick(when + timedelta(minutes=6 * k)) for k in range(3)]
    assert [r.retrying for r in results] == [1, 1, 0] and results[2].failed == 1
    assert results[0].done >= 1  # 别的周期（建档那一周）照常执行
    with closing(connect(e.db)) as conn:
        assert JobStore(conn).status(f"review.weekly:u1:{W1.isoformat()}") == ("failed", 3)
    assert '"event": "scheduler.job_failed"' in e.log.getvalue()
    assert "boom" not in e.log.getvalue()  # 只记异常类型


# ---- 到期复核 ----


def _adjust(e: _Env, field: str, before: Any, after: Any, direction: str, due: date) -> str:
    with closing(connect(e.db)) as conn:
        version = conn.execute("SELECT MAX(version) AS v FROM plan_versions").fetchone()["v"]
        adj_id = f"u1:{version}:{field}:{uuid.uuid4().hex[:6]}"  # 手工插入，避免与复盘生成的重名
        DecisionStore(conn).add_adjustment(
            Adjustment(adj_id, "u1", field, before, after, direction, version, due, "pending")  # type: ignore[arg-type]
        )
    return adj_id


def test_recheck_rolls_back_progression_that_did_not_work(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    _train_week(e, client, W1, checkin=GOOD_CHECKIN)
    e.tick(e.at(W1 + timedelta(days=7), 0, 30))
    progressed = client.get("/plan").json()["plan"]
    assert _sets(progressed) == [3] * 5
    for k in (1, 2):  # 进阶后两周都没跟上
        _train_week(e, client, W1 + timedelta(days=7 * k), status="skipped", rpe=None,
                    checkin={"fatigue": 2, "sleep_quality": 4})  # fmt: skip
    e.tick(e.at(W1 + timedelta(days=21), 7))  # 复核日（调整后 14 天）
    plan = client.get("/plan").json()["plan"]
    assert _sets(plan) == [2] * 5  # 撤回到原来的组数
    kinds = [d["kind"] for d in client.get("/plan/decisions").json()]
    assert "recheck" in kinds


def test_recheck_keeps_working_change_and_postpones_without_data(
    env: tuple[_Env, TestClient],
) -> None:
    e, client = env
    current = client.get("/plan").json()["plan"]["fields"]["weekly_cardio_minutes"]
    kept = _adjust(e, "weekly_cardio_minutes", current - 30, current, "up", W1 + timedelta(days=14))
    for k in (0, 1):
        _train_week(e, client, W1 + timedelta(days=7 * k), checkin=GOOD_CHECKIN)
    e.tick(e.at(W1 + timedelta(days=14), 7))
    with closing(connect(e.db)) as conn:
        assert DecisionStore(conn).adjustment(kept).status in ("kept", "superseded")  # type: ignore[union-attr]
    quiet = _adjust(e, "session_slots", None, client.get("/plan").json()["plan"]["fields"]
                    ["session_slots"], "up", W1 + timedelta(days=42))  # fmt: skip
    e.tick(e.at(W1 + timedelta(days=42), 7))  # 这之前两周没有任何记录：延后一周
    with closing(connect(e.db)) as conn:
        adj = DecisionStore(conn).adjustment(quiet)
    assert adj is not None and adj.status == "pending"
    assert adj.review_date == W1 + timedelta(days=49)


# ---- 今日任务：中断回归、出差与节假日 ----


def _today(client: TestClient) -> dict[str, Any]:
    return client.get("/plan/today", params={"tz": "Asia/Shanghai"}).json()  # type: ignore[no-any-return]


def test_welcome_back_after_ten_days(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    e.clock.set(e.at(W1, 21).astimezone(UTC))
    _post(client, "workout_check", e.at(W1), {"status": "done"})
    back = W1 + timedelta(days=11)  # 周五：训练日
    e.clock.set(e.at(back, 8).astimezone(UTC))
    today = _today(client)
    assert today["greeting"] == "欢迎回来！" and "天没练" not in json.dumps(
        today, ensure_ascii=False
    )
    assert all(i["sets"] == 1 for i in today["training"]["items"])  # 2 组减半为 1 组


def test_long_break_asks_for_baseline_first(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    e.clock.set(e.at(W1, 21).astimezone(UTC))
    _post(client, "workout_check", e.at(W1), {"status": "done"})
    e.clock.set(e.at(W1 + timedelta(days=32), 8).astimezone(UTC))
    today = _today(client)
    assert today["greeting"] == "欢迎回来！" and today["training"]["items"] == []
    assert "基线测试" in today["next_step"]


def test_travel_mode_uses_hotel_version_until_end(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    friday = W1 + timedelta(days=4)
    e.clock.set(e.at(W1, 8).astimezone(UTC))
    resp = client.post("/plan/mode", json={"mode": "travel", "until": friday.isoformat()})
    assert resp.json()["mode"] == {"mode": "travel", "until": friday.isoformat()}
    hotel = _today(client)
    assert hotel["mode"]["mode"] == "travel"
    assert {i["exercise_id"] for i in hotel["training"]["items"]} >= {"sq_bodyweight"}
    e.clock.set(e.at(W1 + timedelta(days=7), 8).astimezone(UTC))  # 结束日之后自动恢复
    normal = _today(client)
    assert normal["mode"] is None and "sq_bodyweight" not in {
        i["exercise_id"] for i in normal["training"]["items"]
    }


def test_holiday_keeps_only_minimum_task(env: tuple[_Env, TestClient]) -> None:
    e, client = env
    e.clock.set(e.at(W1, 8).astimezone(UTC))
    client.post(
        "/plan/mode", json={"mode": "holiday", "until": (W1 + timedelta(days=3)).isoformat()}
    )
    today = _today(client)
    assert today["training"] == {"scheduled": False, "items": []}
    assert today["minimum_task"] and today["minimum_task"][0]["exercise"]
    assert "不设缺口" in today["diet_note"]


def test_mode_requires_end_date(env: tuple[_Env, TestClient]) -> None:
    _, client = env
    assert client.post("/plan/mode", json={"mode": "travel"}).status_code == 422
    assert client.post("/plan/mode", json={"mode": "normal"}).json()["mode"] is None
