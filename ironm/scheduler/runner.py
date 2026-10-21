"""定时任务的执行循环（spec [2026-10-21] §16）。

调度层不直接访问数据存储：任务表、要登记哪些任务、每个任务怎么执行，都由上层注入。
- 每次轮询先登记到期的任务（同一用户、同一任务、同一周期只登记一次），再按到期时间先后执行；
  停机补跑时也就按周期顺序执行。
- 每个任务单独执行，出错只记异常类型；未满 3 次 5 分钟后重试，满 3 次记为失败并告警。
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ironm.core.clock import Clock
from ironm.core.log import get_logger

__all__ = ["JobLike", "JobRepo", "JobSpec", "Scheduler", "TickResult"]

_log = get_logger("ironm.scheduler")


class JobSpec(Protocol):
    """要登记的任务（由上层给出）。"""

    @property
    def kind(self) -> str: ...
    @property
    def user_id(self) -> str: ...
    @property
    def period(self) -> str: ...
    @property
    def due_at(self) -> datetime: ...


class JobLike(Protocol):
    @property
    def job_key(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def user_id(self) -> str: ...
    @property
    def period(self) -> str: ...
    @property
    def due_at(self) -> datetime: ...
    @property
    def attempts(self) -> int: ...


class JobRepo[J: JobLike](Protocol):
    def ensure(
        self, kind: str, user_id: str, period: str, due_at: datetime, now: datetime
    ) -> bool: ...
    def claim(self, now: datetime, limit: int = 50) -> Sequence[J]: ...
    def finish(self, job: J, now: datetime) -> None: ...
    def fail(self, job: J, error_type: str, now: datetime) -> bool: ...


@dataclass(frozen=True)
class TickResult:
    planned: int
    done: int
    retrying: int
    failed: int


class Scheduler[J: JobLike]:
    def __init__(
        self,
        *,
        repo: JobRepo[J],
        plan: Callable[[datetime], Iterable[JobSpec]],
        run: Callable[[J, datetime], None],
        clock: Clock,
    ) -> None:
        self._repo, self._plan, self._run, self._clock = repo, plan, run, clock

    def tick(self) -> TickResult:
        now = self._clock.now()
        planned = sum(
            self._repo.ensure(s.kind, s.user_id, s.period, s.due_at, now) for s in self._plan(now)
        )
        done = retrying = failed = 0
        for job in self._repo.claim(now):
            try:
                self._run(job, now)
            except Exception as e:  # 单个任务出错不影响其他任务
                final = self._repo.fail(job, type(e).__name__, now)
                if final:
                    failed += 1
                    _log.error(
                        "scheduler.job_failed",
                        job_id=job.job_key.replace(":", "_"),
                        status="failed",
                        count=job.attempts,
                        error_type=type(e).__name__,
                    )
                else:
                    retrying += 1
                    _log.info(
                        "scheduler.job_retry",
                        job_id=job.job_key.replace(":", "_"),
                        status="retry",
                        count=job.attempts,
                        error_type=type(e).__name__,
                    )
                continue
            self._repo.finish(job, now)
            done += 1
        return TickResult(planned, done, retrying, failed)
