"""数据质量：异常值“待确认”判定。阈值一律取自规则表（data_quality 规则集），代码里不写数字。

“待确认”不修改记录本身。用户确认或修正一条读数，是追加一条指向它的更正记录：
被更正的记录不再是有效读数，更正记录视为用户已确认。
"""

from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta, tzinfo
from statistics import median
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.events import MealEvent, Source, WeightEvent
from ironm.contracts.unknown import Unknown, UnknownReason, not_recorded
from ironm.engine.effective import resolve
from ironm.rules.loader import RuleBook

WEIGHT_JUMP = "R-DQ-WEIGHT-JUMP"
KCAL_RANGE = "R-DQ-KCAL-RANGE"


class WeightJumpParams(BaseModel):
    max_change_kg: Annotated[float, Field(gt=0)]
    max_change_pct: Annotated[float, Field(ge=0)]
    window_days: Annotated[int, Field(gt=0)]
    baseline_readings: Annotated[int, Field(ge=1)]
    rebaseline_after: Annotated[int, Field(ge=1)]
    rebaseline_tolerance_kg: Annotated[float, Field(gt=0)]
    pending_status_days: Annotated[int, Field(gt=0)]

    def threshold_kg(self, level_kg: float) -> float:
        """跳变阈值：固定千克数与近期水平的百分比，取较大者。"""
        return max(self.max_change_kg, self.max_change_pct / 100 * level_kg)


class KcalRangeParams(BaseModel):
    min_kcal: float
    max_kcal: float


class PendingCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    pending_confirm: bool
    rule_id: str


def weight_jump_pending(
    *,
    rulebook: RuleBook,
    previous_kg: float | None,
    previous_at: datetime | None,
    current_kg: float,
    current_at: datetime,
) -> PendingCheck:
    """规定天数内（含）体重变化超过阈值 → 待确认；没有可比读数时不标。

    阈值取固定千克数与近期水平百分比中较大者。
    """
    p = rulebook.get(WEIGHT_JUMP).params_as(WeightJumpParams)
    if previous_kg is None or previous_at is None:
        return PendingCheck(pending_confirm=False, rule_id=WEIGHT_JUMP)
    within = abs(current_at - previous_at) <= timedelta(days=p.window_days)
    # 四舍五入到微克级，避免 82.0 - 80.0 这类浮点误差让“恰好等于”变成“超过”
    jumped = round(abs(current_kg - previous_kg), 6) > round(p.threshold_kg(previous_kg), 6)
    return PendingCheck(pending_confirm=within and jumped, rule_id=WEIGHT_JUMP)


def kcal_out_of_range(*, rulebook: RuleBook, day_kcal: float) -> PendingCheck:
    """单日热量低于下限或高于上限 → 待确认。"""
    p = rulebook.get(KCAL_RANGE).params_as(KcalRangeParams)
    return PendingCheck(
        pending_confirm=day_kcal < p.min_kcal or day_kcal > p.max_kcal, rule_id=KCAL_RANGE
    )


def effective_weights(
    events: Iterable[WeightEvent], *, as_of: datetime | None = None
) -> list[WeightEvent]:
    """有效体重读数（见有效记录层），按发生时间、记录标识排序。"""
    return resolve(events, as_of=as_of)


def pending_weights(events: Iterable[WeightEvent], rulebook: RuleBook) -> dict[UUID, str]:
    """逐次判定，返回 {待确认的记录标识: 规则编号}。

    每次称重与“近期水平”比较：同一数据流中、规定天数内最近几次未待确认的有效读数的中位数。
    不和单一的上一次读数比：上一次读数带噪声时尖峰会漏判，漏判的尖峰又会让其后的正常读数被误标。
    基准只有 1 条（没有其他读数印证）、而其后几次读数彼此一致又与当前读数一致时，判定基准那条有误：
    改为标记它，以新读数为准。规定天数内没有可比读数时不标；不同数据流（换秤）之间不比较。
    更正记录视为用户已确认，不判定，并计入近期水平。
    """
    p = rulebook.get(WEIGHT_JUMP).params_as(WeightJumpParams)
    window = timedelta(days=p.window_days)
    items = effective_weights(events)
    pending: dict[UUID, str] = {}
    ok: dict[str | None, list[WeightEvent]] = {}
    for idx, e in enumerate(items):
        history = ok.setdefault(e.stream_key, [])
        recent = [r for r in history if e.occurred_at - r.occurred_at <= window]
        recent = recent[-p.baseline_readings :]
        if e.corrects is None and recent:
            check = weight_jump_pending(
                rulebook=rulebook,
                previous_kg=median(r.payload.weight_kg for r in recent),
                previous_at=recent[-1].occurred_at,
                current_kg=e.payload.weight_kg,
                current_at=e.occurred_at,
            )
            if check.pending_confirm:
                baseline_kg = recent[0].payload.weight_kg
                if len(recent) == 1 and _followers_agree(items[idx + 1 :], e, baseline_kg, p):
                    wrong = recent[0]
                    pending[wrong.event_id] = check.rule_id
                    history.remove(wrong)
                    history.append(e)
                else:
                    pending[e.event_id] = check.rule_id
                continue
        history.append(e)
    return pending


def _followers_agree(
    later: list[WeightEvent], e: WeightEvent, baseline_kg: float, p: WeightJumpParams
) -> bool:
    """e 之后同一数据流、规定天数内的若干次读数是否都与 e 一致（容差另设），且都与基准明显不同。

    只看“与 e 一致”不够：尖峰之后回到正常的读数，与 e 的差可能也在跳变阈值之内。
    """
    window = timedelta(days=p.window_days)
    same = [
        f for f in later if f.stream_key == e.stream_key and f.occurred_at - e.occurred_at <= window
    ]
    followers = same[: p.rebaseline_after]
    if len(followers) < p.rebaseline_after:
        return False
    kg = e.payload.weight_kg
    return all(
        abs(f.payload.weight_kg - kg) <= p.rebaseline_tolerance_kg
        and abs(f.payload.weight_kg - baseline_kg) > p.threshold_kg(baseline_kg)
        for f in followers
    )


def daily_kcal(
    meals: Iterable[MealEvent],
    *,
    days: Iterable[date],
    tz: tzinfo,
    as_of: datetime | None = None,
) -> dict[date, float | Unknown]:
    """按用户所在时区的当地日期汇总热量。

    当天任一餐热量未知、或当天没有饮食记录时，当天总量为“未知”：部分记录会让总量偏低，空白优于部分记录。
    """
    by_day: dict[date, list[MealEvent]] = {}
    for m in resolve(meals, as_of=as_of):
        by_day.setdefault(m.occurred_at.astimezone(tz).date(), []).append(m)
    totals: dict[date, float | Unknown] = {}
    for day in days:
        kcals = [
            m.payload.kcal if m.source is not Source.ESTIMATE or m.confirmed else not_recorded()
            for m in by_day.get(day, [])
        ]  # 未经确认的估算视为没有热量
        if not kcals:
            totals[day] = not_recorded()
        elif any(isinstance(k, Unknown) for k in kcals):
            totals[day] = Unknown(unknown=UnknownReason.NOT_RECORDED)
        else:
            totals[day] = sum(k for k in kcals if not isinstance(k, Unknown))
    return totals


def pending_kcal_days(
    totals: Mapping[date, float | Unknown], rulebook: RuleBook
) -> dict[date, str]:
    """已知的单日总量超出范围 → 待确认；未知的日子不判定。"""
    pending: dict[date, str] = {}
    for day, total in totals.items():
        if isinstance(total, Unknown):
            continue
        check = kcal_out_of_range(rulebook=rulebook, day_kcal=total)
        if check.pending_confirm:
            pending[day] = check.rule_id
    return pending
