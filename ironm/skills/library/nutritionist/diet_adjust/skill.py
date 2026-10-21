"""营养师 · 两周饮食复盘（spec [2026-10-21] §17）：按决策表 R-RV-DIET 调整每日热量目标。"""

from datetime import timedelta

from pydantic import BaseModel, ConfigDict

from ironm.contracts.agents import AgentOutput, Constraint
from ironm.contracts.screening import Tier
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import Change, current_value, no_opinion, proposals
from ironm.skills.view import SnapshotView

RULE = "R-RV-DIET"


class DietParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_meal_days_per_week: int
    min_weigh_ins_per_week: int
    max_ci_half_width: float
    low_completion: float
    low_adherence_days: int
    ok_completion: float
    ok_adherence_days: int
    step_kcal: int
    fast_loss_kg_per_week: float
    fast_loss_after_days: int
    target_low_a: float
    target_low_a_prime: float
    trend_window_days: int
    review_days: int


RULE_PARAMS = {RULE: DietParams}


def _num(view: SnapshotView, key: str) -> float | None:
    value = view.get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _floor(constraints: tuple[Constraint, ...]) -> float | None:
    mins = [
        float(c.value)
        for c in constraints
        if c.field == "daily_kcal_target" and c.op == "min" and isinstance(c.value, int | float)
    ]
    return max(mins) if mins else None


def slow_loss(loss: float, tier: object, p: DietParams) -> bool:
    """减重不到目标下限的一半（供上一周期的判断复用）。"""
    target = p.target_low_a_prime if tier == Tier.A_PRIME else p.target_low_a
    return loss < target / 2


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    p = context.rulebook.get(RULE).params_as(DietParams)
    if view.get("review.red_flag") is True:
        return no_opinion(context, "diet_red_flag")
    meal_days, weigh_ins = _num(view, "review.meal_days_min"), _num(view, "review.weigh_ins_min")
    ci_half, loss = _num(view, "review.ci_half_width"), _num(view, "review.loss_kg_per_week")
    insufficient = (
        meal_days is None
        or meal_days < p.min_meal_days_per_week
        or weigh_ins is None
        or weigh_ins < p.min_weigh_ins_per_week
        or ci_half is None
        or ci_half > p.max_ci_half_width
        or loss is None
    )
    if insufficient or loss is None:
        return no_opinion(context, "diet_insufficient")
    completion, adherence = _num(view, "review.completion"), _num(view, "review.adherence_days")
    if completion is None or adherence is None:
        return no_opinion(context, "diet_insufficient")
    if completion < p.low_completion or adherence < p.low_adherence_days:
        return no_opinion(context, "diet_simplify_first")
    kcal = current_value(view, "daily_kcal_target")
    if not isinstance(kcal, int):
        return no_opinion(context, "diet_no_kcal_target")
    constraints = view.get("constraints")
    rules: tuple[Constraint, ...] = constraints if isinstance(constraints, tuple) else ()
    floor = _floor(rules)
    tier = view.get("tier")
    flagged = view.get("review.partial_week") is True or view.get("review.menstruation") is True
    good = completion >= p.ok_completion and adherence >= p.ok_adherence_days
    target = None
    if not flagged and good and slow_loss(loss, tier, p) and view.get("review.prev_slow") is True:
        lowered = kcal - p.step_kcal
        if floor is not None and lowered < floor:
            return no_opinion(context, "diet_at_floor")
        target, why = lowered, "执行良好，但连续两个周期减重不到目标下限的一半"
    days = _num(view, "review.days_since_start")
    settled = days is not None and days >= p.fast_loss_after_days  # 前 4 周不判断减得太快
    if target is None and settled and loss > p.fast_loss_kg_per_week:
        target = kcal + p.step_kcal
        why = f"满 4 周后减重快于每周 {p.fast_loss_kg_per_week} 千克"
    if target is None:
        return no_opinion(context, "diet_keep")
    review = (view.as_of + timedelta(days=p.review_days)).date()
    change = Change(
        field="daily_kcal_target",
        to_value=target,
        reason_facts=(
            why,
            f"近两周减重：每周约 {loss:.2f} 千克（可信范围半宽 {ci_half:.2f}）",
            f"训练完成率：{completion:.0%}；按计划吃：每周 {adherence:.0f} 天",
            f"热量底线：{'无' if floor is None else f'{floor:.0f} 千卡'}",
        ),
        rule_id=RULE,
        expected_effect="接下来两周减重回到每周目标区间",
        rollback_condition="持续饥饿、疲劳、头晕，或减重快于每周 1 千克时，回到原来的热量",
    )
    return proposals(context, view, [change], review)
