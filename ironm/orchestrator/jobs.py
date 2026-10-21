"""要登记的定时任务与每个任务怎么执行（spec [2026-10-21] §16、§17）。

- 每周训练复盘：用户签到后执行；没签到就在这一周结束时（用户时区周日 24 点）执行。停机补跑最多
  回溯 8 周，数据截止到周期结束（或签到）时刻。
- 到期复核：有调整到了复核日，当天执行一次。
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ironm.contracts.events import CheckinEvent
from ironm.core.clock import FixedClock
from ironm.core.ids import IdGenerator
from ironm.engine.review_facts import week_start
from ironm.orchestrator.reviews import ReviewDeps, run_recheck, run_review
from ironm.rules.loader import RuleBook
from ironm.skills.loader import LoadedSkill
from ironm.store.events import EventStore
from ironm.store.memory import MemoryStore
from ironm.store.plans import PlanStore
from ironm.store.reviews import DecisionStore
from ironm.store.safety import SafetyFlagStore

__all__ = ["CATCH_UP_WEEKS", "JobContext", "JobLike", "JobSpec", "plan_jobs", "run_job"]

CATCH_UP_WEEKS = 8


@dataclass(frozen=True)
class JobSpec:
    """要登记的任务：同一用户、同一任务类型、同一周期只登记一次（由调度层的任务表保证）。"""

    kind: str
    user_id: str
    period: str
    due_at: datetime


class JobLike(Protocol):
    @property
    def kind(self) -> str: ...
    @property
    def user_id(self) -> str: ...
    @property
    def period(self) -> str: ...
    @property
    def due_at(self) -> datetime: ...


DEFAULT_TZ = "Asia/Shanghai"


@dataclass(frozen=True)
class JobContext:
    rulebook: RuleBook
    skills: tuple[LoadedSkill, ...]
    ids: IdGenerator
    # 其他任务（提醒、到期删除、备份）的执行函数，按任务类型登记
    extra: dict[str, Callable[[sqlite3.Connection, JobLike, datetime], None]] = field(
        default_factory=dict
    )
    planners: tuple[Callable[[sqlite3.Connection, datetime], list[JobSpec]], ...] = ()


def _zone(name: object) -> ZoneInfo:
    try:
        return ZoneInfo(str(name)) if name else ZoneInfo(DEFAULT_TZ)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo(DEFAULT_TZ)


def _midnight(day: date, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, time(0), tzinfo=tz)


def plan_jobs(conn: sqlite3.Connection, now: datetime, ctx: JobContext) -> list[JobSpec]:
    plans, memory = PlanStore(conn), MemoryStore(conn)
    events, decisions = EventStore(conn, FixedClock(now)), DecisionStore(conn)
    specs: list[JobSpec] = []
    for user in plans.users():
        tz = _zone(memory.profile(user)[1].get("timezone"))
        today = now.astimezone(tz).date()
        started = week_start(plans.history(user)[0].at.astimezone(tz).date())
        first = max(started, week_start(today) - timedelta(weeks=CATCH_UP_WEEKS))
        checkins = [
            e
            for e in events.list_for_user(user, types={"checkin"}, since=_midnight(first, tz))
            if isinstance(e, CheckinEvent) and e.occurred_at <= now
        ]
        week = first
        while week <= week_start(today):
            end = _midnight(week + timedelta(days=7), tz)
            # 签到后执行：以签到提交的时刻为准（数据截止也在这一刻），签到本身才算得进这次复盘
            done = [
                max(c.occurred_at, c.recorded_at)
                for c in checkins
                if week <= c.occurred_at.astimezone(tz).date() < week + timedelta(days=7)
            ]
            due = min([end, *done])
            if due <= now:
                specs.append(JobSpec("review.weekly", user, week.isoformat(), due))
            week += timedelta(days=7)
        if decisions.due_adjustments(user, today):
            specs.append(JobSpec("recheck", user, today.isoformat(), now))
    for planner in ctx.planners:
        specs += planner(conn, now)
    return specs


def run_job(conn: sqlite3.Connection, job: JobLike, now: datetime, ctx: JobContext) -> None:
    if job.kind in ctx.extra:
        ctx.extra[job.kind](conn, job, now)
        return
    memory = MemoryStore(conn)
    profile = memory.profile(job.user_id)[1]
    tz = _zone(profile.get("timezone"))
    deps = ReviewDeps(PlanStore(conn), DecisionStore(conn), ctx.rulebook, ctx.skills)
    events = EventStore(conn, FixedClock(now)).list_for_user(job.user_id)
    flags = SafetyFlagStore(conn, ctx.ids).for_user(job.user_id)
    if job.kind == "review.weekly":
        run_review(
            job.user_id,
            date.fromisoformat(job.period),
            deps=deps,
            events=events,
            profile=profile,
            flags=flags,
            as_of=min(job.due_at, now),  # 补跑时数据截止到周期结束
            now=now,
            tz=tz,
        )
    elif job.kind == "recheck":
        run_recheck(
            job.user_id,
            date.fromisoformat(job.period),
            deps=deps,
            events=events,
            profile=profile,
            flags=flags,
            now=now,
            tz=tz,
        )
    else:
        raise ValueError(f"不认识的任务类型：{job.kind}")
