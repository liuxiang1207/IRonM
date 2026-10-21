"""数据新鲜度与训练完成度。规定天数与计分取自规则表。"""

from collections.abc import Iterable
from datetime import date, datetime, timedelta, tzinfo
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.events import WeightEvent, WorkoutCheckEvent
from ironm.contracts.metrics import Need
from ironm.contracts.unknown import Unknown, not_recorded
from ironm.engine.data_quality import pending_weights
from ironm.engine.effective import resolve
from ironm.engine.stats import mean_known
from ironm.rules.loader import RuleBook

WEIGHT_STALE = "R-DQ-WEIGHT-STALE"
TRAINING_COMPLETION = "R-MT-TRAINING-COMPLETION"


class StaleParams(BaseModel):
    max_age_days: int


class CompletionParams(BaseModel):
    score_done: float
    score_half: float
    score_skipped: float
    min_recorded_share: Annotated[float, Field(gt=0, le=1)]


class Freshness(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "stale", "insufficient"]
    needed: tuple[Need, ...]
    last_record_at: datetime | Unknown
    rule_id: str


class Completion(BaseModel):
    model_config = ConfigDict(frozen=True)

    # 有记录的计划日占比低于规则下限（或没有计划日）时为 insufficient：没练的日子更可能不打卡
    status: Literal["ok", "insufficient"]
    rate: float | Unknown  # 只计有打卡记录的计划日
    recorded_days: int
    planned_days: int
    coverage: float | Unknown  # 有记录的计划日占比；没有计划日时为“未知”
    rule_id: str


def weight_freshness(
    events: Iterable[WeightEvent], *, as_of: datetime, rulebook: RuleBook
) -> Freshness:
    """最近一次可信称重（有效、非待确认）距今超过规定天数为“已过期”；恰好等于不算。"""
    p = rulebook.get(WEIGHT_STALE).params_as(StaleParams)
    items = resolve(events, as_of=as_of)
    pending = pending_weights(items, rulebook)
    trusted = [e for e in items if e.event_id not in pending]
    if not trusted:
        return Freshness(
            status="insufficient",
            needed=(Need(item="weight_record", count=1),),
            last_record_at=not_recorded(),
            rule_id=WEIGHT_STALE,
        )
    last = trusted[-1].occurred_at  # 已按时间排序
    stale = as_of - last > timedelta(days=p.max_age_days)
    return Freshness(
        status="stale" if stale else "ok", needed=(), last_record_at=last, rule_id=WEIGHT_STALE
    )


def training_completion(
    events: Iterable[WorkoutCheckEvent],
    *,
    planned_days: Iterable[date],
    tz: tzinfo,
    rulebook: RuleBook,
) -> Completion:
    """对照计划训练日：同一天以最后一次打卡为准；无记录的计划日为“未知”，不计入完成度。"""
    p = rulebook.get(TRAINING_COMPLETION).params_as(CompletionParams)
    score = {"done": p.score_done, "half": p.score_half, "skipped": p.score_skipped}
    latest: dict[date, WorkoutCheckEvent] = {}
    for e in resolve(events):
        latest[e.occurred_at.astimezone(tz).date()] = e
    days = list(dict.fromkeys(planned_days))
    scores: list[float | Unknown] = [
        score[latest[d].payload.status] if d in latest else not_recorded() for d in days
    ]
    recorded = sum(1 for s in scores if not isinstance(s, Unknown))
    enough = bool(days) and recorded / len(days) >= p.min_recorded_share
    return Completion(
        status="ok" if enough else "insufficient",
        rate=mean_known(scores),
        recorded_days=recorded,
        planned_days=len(days),
        coverage=recorded / len(days) if days else not_recorded(),
        rule_id=TRAINING_COMPLETION,
    )
