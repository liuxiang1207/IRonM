"""数据新鲜度与训练完成度。规定天数与计分取自规则表。"""

from collections.abc import Iterable
from datetime import date, datetime, timedelta, tzinfo
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ironm.contracts.events import WeightEvent, WorkoutCheckEvent
from ironm.contracts.unknown import Unknown, not_recorded
from ironm.engine.data_quality import effective_weights, pending_weights
from ironm.engine.stats import mean_known
from ironm.rules.loader import RuleBook

WEIGHT_STALE = "R-DQ-WEIGHT-STALE"
TRAINING_COMPLETION = "R-MT-TRAINING-COMPLETION"


class _StaleParams(BaseModel):
    max_age_days: int


class _CompletionParams(BaseModel):
    score_done: float
    score_half: float
    score_skipped: float


class Freshness(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: Literal["fresh", "stale", "no_data"]
    last_record_at: datetime | Unknown
    rule_id: str


class Completion(BaseModel):
    model_config = ConfigDict(frozen=True)

    rate: float | Unknown  # 只计有打卡记录的计划日
    recorded_days: int
    planned_days: int
    coverage: float  # 有记录的计划日占比
    rule_id: str


def weight_freshness(
    events: Iterable[WeightEvent], *, as_of: datetime, rulebook: RuleBook
) -> Freshness:
    """最近一次可信称重（有效、非待确认）距今超过规定天数为“已过期”；恰好等于不算。"""
    p = rulebook.get(WEIGHT_STALE).params_as(_StaleParams)
    items = list(events)
    pending = pending_weights(items, rulebook)
    trusted = [e for e in effective_weights(items) if e.event_id not in pending]
    if not trusted:
        return Freshness(state="no_data", last_record_at=not_recorded(), rule_id=WEIGHT_STALE)
    last = trusted[-1].occurred_at  # 已按时间排序
    stale = as_of - last > timedelta(days=p.max_age_days)
    return Freshness(state="stale" if stale else "fresh", last_record_at=last, rule_id=WEIGHT_STALE)


def training_completion(
    events: Iterable[WorkoutCheckEvent],
    *,
    planned_days: Iterable[date],
    tz: tzinfo,
    rulebook: RuleBook,
) -> Completion:
    """对照计划训练日：同一天以最后一次打卡为准；无记录的计划日为“未知”，不计入完成度。"""
    p = rulebook.get(TRAINING_COMPLETION).params_as(_CompletionParams)
    score = {"done": p.score_done, "half": p.score_half, "skipped": p.score_skipped}
    latest: dict[date, WorkoutCheckEvent] = {}
    for e in sorted(events, key=lambda e: (e.occurred_at, str(e.event_id))):
        latest[e.occurred_at.astimezone(tz).date()] = e
    days = list(dict.fromkeys(planned_days))
    scores: list[float | Unknown] = [
        score[latest[d].payload.status] if d in latest else not_recorded() for d in days
    ]
    recorded = sum(1 for s in scores if not isinstance(s, Unknown))
    return Completion(
        rate=mean_known(scores),
        recorded_days=recorded,
        planned_days=len(days),
        coverage=recorded / len(days) if days else 0.0,
        rule_id=TRAINING_COMPLETION,
    )
