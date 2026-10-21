"""快进模拟：每周复盘与计划调整（spec [2026-10-21] §17，计划 D40）。

5 类模拟用户各 8 周：建档 → 每天提交当天的记录 → 当天结束时执行一轮定时任务（复盘、到期复核）。
跑完后用独立的核对器（sim/review_check.py，按 spec 的决策表另写的判断）逐次核对：每次调整都要
符合决策表。

- 模拟用户按计划安排的训练日训练：生成器里每周的训练按先后顺序挪到计划安排的训练日上（计划只看
  安排的训练日，挪过去才能算完成率）。
- 用户目标（Claude 的决定）：按计划执行、疲劳累积、每周只称一次三类为减脂（会做饮食复盘），
  习惯模式、完成度低两类为健身。

`make sim-review` 输出每类用户的复盘与调整，以及不符合决策表之处（应为 0）。
"""

import json
import sys
from contextlib import closing
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.contracts.events import Event, SetLogEvent, WorkoutCheckEvent
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.orchestrator.onboarding import DEFAULT_SCHEDULES
from ironm.store.db import connect
from ironm.store.plans import PlanStore
from ironm.store.reviews import DecisionStore
from sim.generate import Simulation, generate
from sim.persona import PERSONAS_DIR, Persona, load_personas
from sim.review_check import SPEC_DIET, SPEC_WEEKLY, check_review

__all__ = ["GOALS", "PersonaReport", "run_all", "run_persona"]

GOALS = {
    "adherent": "fat_loss",
    "fatigue": "fat_loss",
    "weekly_weigher": "fat_loss",
    "habit_mode": "fitness",
    "low_completion": "fitness",
}
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True)
class PersonaReport:
    persona: str
    goal: str
    reviews: int  # 每周复盘次数
    rows: tuple[str, ...]  # 每次复盘命中的训练决策表行（按周）
    diet_rows: tuple[str, ...]  # 做了饮食复盘的那几周命中的饮食决策表行
    adjusted: int  # 发布了新计划版本的复盘次数
    changes: tuple[str, ...]  # 每次调整的一行说明
    violations: tuple[str, ...]  # 不符合决策表之处（应为空）


def _intake(p: Persona) -> dict[str, Any]:
    born = p.start_date.replace(year=p.start_date.year - p.profile.age)
    return {
        "birth_date": born.isoformat(),
        "sex": p.profile.sex,
        "height_cm": float(p.profile.height_cm),
        "weight_kg": p.profile.start_weight_kg,
        "conditions": [],
        "goal": GOALS[p.id],
        "environment": "gym",
        "available_days": len(p.habits.training_days),
        "training_level": "beginner",
        "activity": "moderate",
        "restrictions": [],
    }


def _follow_schedule(
    sim: Simulation, persona: Persona, schedule: list[int]
) -> dict[date, list[Event]]:
    """把每周的训练（打卡与记组）按先后顺序挪到计划安排的训练日；其余记录不动。

    计划安排的天数少于用户自己练的天数时，多出来的那几次留在原来那天（算额外多练，不计入
    完成率）；安排的天数更多时，多出来的安排日没有记录。
    """
    own = sorted(_WEEKDAYS.index(d) for d in persona.habits.training_days)
    moved = dict(zip(own, sorted(schedule), strict=False))
    by_day: dict[date, list[Event]] = {}
    for day in sim.days:
        for e in day.events:
            target = day.date
            weekday = day.date.weekday()
            if isinstance(e, WorkoutCheckEvent | SetLogEvent) and weekday in moved:
                shift = timedelta(days=moved[weekday] - weekday)
                e = e.model_copy(
                    update={
                        "occurred_at": e.occurred_at + shift,
                        "recorded_at": e.recorded_at + shift,
                    }
                )
                target = day.date + shift
            by_day.setdefault(target, []).append(e)
    return by_day


def _post(client: TestClient, user: str, events: list[Event]) -> None:
    for start in range(0, len(events), 500):
        chunk = [json.loads(e.model_dump_json()) for e in events[start : start + 500]]
        resp = client.post("/events/batch", json={"events": chunk}, headers={"X-IronM-User": user})
        resp.raise_for_status()
        bad = [r for r in resp.json()["results"] if r["status"] == "invalid"]
        if bad:
            raise RuntimeError(f"模拟记录不合规：{bad[:2]}")


def _describe(week: str, changes: list[dict[str, Any]]) -> str:
    parts = []
    for c in changes:
        if c["field"] == "weekly_cardio_minutes":
            parts.append(f"有氧 {c['from']}→{c['to']} 分钟")
        elif c["field"] == "session_slots":
            sets = [f"{b['sets']}→{a['sets']}" for b, a in zip(c["from"], c["to"], strict=False)]
            reps = [
                f"{b['reps'][1]}→{a['reps'][1]}" for b, a in zip(c["from"], c["to"], strict=False)
            ]
            parts.append(f"组数 {sets[0]}，次数上限 {reps[0]}（共 {len(sets)} 个动作）")
        elif c["field"] == "daily_kcal_target":
            parts.append(f"每日热量 {c['from']}→{c['to']} 千卡")
        else:
            parts.append(c["field"])
    return f"{week}：" + "；".join(parts)


def run_persona(persona: Persona, workdir: Path) -> PersonaReport:
    db = workdir / f"{persona.id}.sqlite"
    db.unlink(missing_ok=True)
    clock = FixedClock(persona.start_at() + timedelta(hours=8))  # 第 1 天早上 8 点建档
    settings = load_settings({"IRONM_DB_PATH": str(db)})
    app = create_app(settings, clock, environ={})
    user = persona.user_id
    with TestClient(app, headers={"X-IronM-User": user}) as client:
        onboarded = client.post("/onboarding", json=_intake(persona)).json()
        if onboarded.get("status") != "plan":
            raise RuntimeError(f"{persona.id} 建档没有得到计划：{onboarded.get('status')}")
        fields = client.get("/plan").json()["plan"]["fields"]
        schedule = (
            fields.get("training_schedule") or DEFAULT_SCHEDULES[len(persona.habits.training_days)]
        )
        sim = generate(persona)
        by_day = _follow_schedule(sim, persona, list(schedule))
        for day in sim.days:
            clock.set(day.closes_at)
            if by_day.get(day.date):
                _post(client, user, by_day[day.date])
            app.state.tick()
    with closing(connect(db)) as conn:
        decisions = list(reversed(DecisionStore(conn).recent(user, kind="weekly", limit=100)))
        history = PlanStore(conn).history(user)
    first = history[0].record
    tier = (first.get("tier") or {}).get("tier")
    start_cardio = first["plan"]["fields"].get("weekly_cardio_minutes")
    rows: list[str] = []
    diet_rows, changes, violations = [], [], []
    for d in decisions:
        streak = 0
        while streak < len(rows) and rows[-1 - streak] == "recovery":
            streak += 1
        row, diet_seen, problems = check_review(
            d.record, weekly=SPEC_WEEKLY, diet=SPEC_DIET, goal=GOALS[persona.id], tier=tier,
            start_cardio=start_cardio, recovery_before=streak,
        )  # fmt: skip
        rows.append(row)
        if diet_seen is not None:
            diet_rows.append(diet_seen)
        if d.record.get("changes"):
            changes.append(_describe(d.period, d.record["changes"]))
        violations += [f"{persona.id} {d.period}：{p}" for p in problems]
    return PersonaReport(
        persona=persona.id,
        goal=GOALS[persona.id],
        reviews=len(decisions),
        rows=tuple(rows),
        diet_rows=tuple(diet_rows),
        adjusted=sum(1 for d in decisions if d.record.get("published_version")),
        changes=tuple(changes),
        violations=tuple(violations),
    )


def run_all(workdir: Path) -> list[PersonaReport]:
    workdir.mkdir(parents=True, exist_ok=True)
    return [run_persona(p, workdir) for p in load_personas(PERSONAS_DIR).values()]


def main() -> int:
    reports = run_all(Path("data/sim/reviews"))
    for r in reports:
        print(f"\n== {r.persona}（目标 {r.goal}）：复盘 {r.reviews} 次，调整 {r.adjusted} 次")
        print("   训练决策表：" + " / ".join(r.rows))
        if r.diet_rows:
            print("   饮食决策表：" + " / ".join(r.diet_rows))
        for line in r.changes:
            print("   " + line)
    violations = [v for r in reports for v in r.violations]
    print(f"\n不符合决策表之处：{len(violations)}")
    for v in violations:
        print("  " + v)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
