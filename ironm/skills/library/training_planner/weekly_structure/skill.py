"""每周结构（spec §30）：第一版力量训练次数、全身槽位、用力程度与每周有氧分钟数。"""

from collections.abc import Iterable, Mapping
from datetime import timedelta
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from ironm.contracts.agents import AgentOutput, Constraint
from ironm.contracts.exercises import Pattern
from ironm.contracts.screening import Tier
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import Change, proposals
from ironm.skills.view import SnapshotView

RULE = "R-TR-START"
CONSERVATIVE = frozenset({Tier.B, Tier.COLLABORATIVE})


class StartParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    strength_days: int
    strength_days_max: int
    three_days_min_available: int
    slots: tuple[str, ...]  # 动作模式；规则表里是文字，这里按名称校验
    sets: int
    reps_low: int
    reps_high: int
    rpe_low: int
    rpe_high: int
    cardio_start_add: int
    cardio_start_min: int
    cardio_start_min_conservative: int
    review_days: int

    @model_validator(mode="after")
    def _check(self) -> Self:
        unknown = [s for s in self.slots if s not in set(Pattern)]
        if unknown:
            raise ValueError(f"不认识的动作模式：{unknown}")
        if not 1 <= self.strength_days <= self.strength_days_max <= 7:
            raise ValueError("力量训练次数应满足 1 ≤ 默认 ≤ 最多 ≤ 7")
        if not 0 < self.reps_low <= self.reps_high or not 1 <= self.rpe_low <= self.rpe_high <= 10:
            raise ValueError("次数与 RPE 范围不合理")
        return self


RULE_PARAMS = {RULE: StartParams}


def _max_limit(
    constraints: Iterable[Constraint], field: str, key: str | None = None
) -> float | None:
    """安全官对某字段给的最严上限；没有时为 None。"""
    limits: list[float] = []
    for c in constraints:
        if c.field != field or c.op != "max":
            continue
        raw: object = c.value
        value = raw.get(key) if key is not None and isinstance(raw, Mapping) else raw
        if isinstance(value, int | float):
            limits.append(float(value))
    return min(limits) if limits else None


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    p = context.rulebook.get(RULE).params_as(StartParams)
    tier = view.get("tier")
    constraints = view.get("constraints")
    rules: tuple[Constraint, ...] = constraints if isinstance(constraints, tuple) else ()
    available = view.get("profile.available_days")
    current = view.get("profile.current_cardio_minutes")

    enough_days = isinstance(available, int) and available >= p.three_days_min_available
    days = p.strength_days_max if enough_days else p.strength_days

    rpe_cap = _max_limit(rules, "intensity_target", "rpe")
    rpe_high = p.rpe_high if rpe_cap is None else min(p.rpe_high, int(rpe_cap))
    rpe_low = min(p.rpe_low, rpe_high)

    floor = p.cardio_start_min_conservative if tier in CONSERVATIVE else p.cardio_start_min
    base = current if isinstance(current, int) and current >= 0 else 0
    cardio = max(base + p.cardio_start_add, floor)
    cardio_cap = _max_limit(rules, "weekly_cardio_minutes")
    if cardio_cap is not None:
        cardio = min(cardio, int(cardio_cap))

    slots = [
        {
            "pattern": s,
            "sets": p.sets,
            "reps": [p.reps_low, p.reps_high],
            "rpe": [rpe_low, rpe_high],
        }
        for s in p.slots
    ]
    review = (view.as_of + timedelta(days=p.review_days)).date()
    rollback = "连续两周完成率低于一半，或训练后疼痛、过度疲劳时，回到更轻的安排并复核"
    changes = [
        Change(
            field="training_days_per_week",
            to_value=days,
            reason_facts=(f"可练天数：{available if isinstance(available, int) else '未知'}",),
            rule_id=RULE,
            expected_effect="每周至少 2 天力量训练，保持减脂期的肌肉量",
            rollback_condition=rollback,
        ),
        Change(
            field="session_slots",
            to_value=slots,
            reason_facts=("全身训练：下蹲、髋铰链、推、拉、核心",),
            rule_id=RULE,
            expected_effect="两周内能按要求完成全部组数",
            rollback_condition=rollback,
        ),
        Change(
            field="intensity_target",
            to_value={"rpe": [rpe_low, rpe_high]},
            reason_facts=(f"安全官给的 RPE 上限：{int(rpe_cap) if rpe_cap is not None else '无'}",),
            rule_id=RULE,
            expected_effect="每组结束时还能再做 3–4 次",
            rollback_condition=rollback,
        ),
        Change(
            field="weekly_cardio_minutes",
            to_value=cardio,
            reason_facts=(
                f"当前每周运动：{current if isinstance(current, int) else '未知'} 分钟",
                f"每周有氧上限：{int(cardio_cap) if cardio_cap is not None else '无'}",
            ),
            rule_id=RULE,
            expected_effect="逐步接近每周 250 分钟的减脂有氧目标",
            rollback_condition=rollback,
        ),
    ]
    return proposals(context, view, changes, review)
