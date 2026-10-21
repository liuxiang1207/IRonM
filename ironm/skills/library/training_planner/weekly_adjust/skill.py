"""训练规划师 · 每周训练复盘（spec [2026-10-21] §17）：按决策表 R-RV-WEEKLY 调整有氧与组数次数。"""

from datetime import timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict

from ironm.contracts.agents import AgentOutput, Constraint
from ironm.contracts.common import deep_thaw
from ironm.contracts.screening import Tier
from ironm.safety.tiers import TIER_LIMITS_RULE, TierLimitsParams
from ironm.skills.library.training_planner.weekly_structure.skill import (
    RULE as START_RULE,
)
from ironm.skills.library.training_planner.weekly_structure.skill import (
    StartParams,
    _max_limit,
)
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import Change, current_value, no_opinion, proposals
from ironm.skills.view import SnapshotView

RULE = "R-RV-WEEKLY"
CONSERVATIVE = frozenset({Tier.B, Tier.COLLABORATIVE})


class WeeklyParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fatigue_rise: int
    sleep_low: int
    rpe_high: float
    low_completion: float
    high_completion: float
    rpe_ok: float
    cardio_cut_pct: int
    max_sets: int
    max_reps_high: int
    review_days: int


RULE_PARAMS = {RULE: WeeklyParams, START_RULE: StartParams, TIER_LIMITS_RULE: TierLimitsParams}


def _num(view: SnapshotView, key: str) -> float | None:
    value = view.get(key)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _row(view: SnapshotView, p: WeeklyParams) -> str:
    """决策表中命中的那一行。"""
    if view.get("review.partial_week") is True:
        return "partial_week"
    completion = _num(view, "review.completion")
    if completion is None:
        return "insufficient"
    previous = _num(view, "review.prev_completion")
    rpe = _num(view, "review.avg_rpe")
    fatigue, baseline = _num(view, "review.fatigue"), _num(view, "review.fatigue_baseline")
    sleep = _num(view, "review.sleep_quality")
    down = view.get("review.menstruation") is not True  # 经期这一周不下调
    tired = fatigue is not None and baseline is not None and fatigue - baseline >= p.fatigue_rise
    if down and (tired or (sleep is not None and sleep <= p.sleep_low)):
        return "recovery"
    if down and rpe is not None and rpe >= p.rpe_high:
        return "rpe_high"
    low_twice = previous is not None and max(completion, previous) < p.low_completion
    if down and low_twice:
        return "low_completion"
    steady = previous is not None and previous >= p.high_completion
    if completion >= p.high_completion and (rpe <= p.rpe_ok if rpe is not None else steady):
        return "progress"
    return "keep"


def _slots(slots: list[dict[str, Any]], action: str, p: WeeklyParams) -> list[dict[str, Any]]:
    out = []
    for s in slots:
        s = dict(s)
        if action == "down":
            s["sets"] = max(1, s["sets"] - 1)
        elif action == "up":
            if s["sets"] < p.max_sets:
                s["sets"] += 1
            elif s["reps"][1] < p.max_reps_high:
                s["reps"] = [s["reps"][0], p.max_reps_high]
        out.append(s)
    return out


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    p = context.rulebook.get(RULE).params_as(WeeklyParams)
    start = context.rulebook.get(START_RULE).params_as(StartParams)
    limits = context.rulebook.get(TIER_LIMITS_RULE).params_as(TierLimitsParams)
    row = _row(view, p)
    if row in ("partial_week", "insufficient", "keep"):
        return no_opinion(context, f"weekly_{row}")
    cardio_now = current_value(view, "weekly_cardio_minutes")
    slots_now = current_value(view, "session_slots")
    if not isinstance(cardio_now, int) or not isinstance(slots_now, list | tuple):
        return no_opinion(context, "weekly_no_current_plan")
    thawed = deep_thaw(slots_now)
    slots_now = [dict(s) for s in thawed] if isinstance(thawed, list) else []
    conservative = view.get("tier") in CONSERVATIVE
    floor = start.cardio_start_min_conservative if conservative else start.cardio_start_min
    constraints = view.get("constraints")
    rules: tuple[Constraint, ...] = constraints if isinstance(constraints, tuple) else ()
    cap = _max_limit(rules, "weekly_cardio_minutes")
    ceiling = min(start.cardio_target_min, int(cap) if cap is not None else start.cardio_target_min)
    cut = round(cardio_now * (100 - p.cardio_cut_pct) / 100 / 5) * 5
    facts = _facts(view)
    cardio, strength = cardio_now, "same"
    if row == "recovery":
        cardio, strength = cut, "down"
    elif row == "rpe_high":
        strength = "down"
    elif row == "low_completion":
        cardio = max(cut, min(floor, cardio_now))
    elif row == "progress":
        step = limits.b_cardio_step_min if conservative else limits.a_cardio_step_min
        cardio = max(cardio_now, min(cardio_now + step, ceiling))
        strength = "up"
    review = (view.as_of + timedelta(days=p.review_days)).date()
    why = {
        "recovery": "疲劳明显升高或睡眠差：先恢复，暂停进阶",
        "rpe_high": "本周平均 RPE ≥9：负荷偏重",
        "low_completion": "连续两周完成率低于一半：先降难度",
        "progress": "完成率高且不吃力：按先加组、再加次数的顺序进阶",
    }[row]
    changes = []
    if cardio != cardio_now:
        changes.append(
            Change(
                field="weekly_cardio_minutes",
                to_value=cardio,
                reason_facts=(why, *facts, f"有氧上限：每周 {ceiling} 分钟"),
                rule_id=RULE,
                expected_effect="接下来两周能按计划完成，疲劳与睡眠回到平常水平",
                rollback_condition="两周内完成率低于一半或平均 RPE ≥9 时，回到原来的分钟数",
            )
        )
    new_slots = _slots(slots_now, strength, p)
    if new_slots != slots_now:
        changes.append(
            Change(
                field="session_slots",
                to_value=new_slots,
                reason_facts=(why, *facts),
                rule_id=RULE,
                expected_effect="接下来两周能完成全部组数，每组结束时还能再做 2–3 次",
                rollback_condition="两周内完成率低于一半或平均 RPE ≥9 时，回到原来的组数次数",
            )
        )
    if not changes:
        return no_opinion(context, f"weekly_{row}_at_limit")
    return proposals(context, view, changes, review)


def _facts(view: SnapshotView) -> tuple[str, ...]:
    def show(key: str, label: str, fmt: str = "{}") -> str:
        value = view.get(key)
        known = isinstance(value, int | float) and not isinstance(value, bool)
        return f"{label}：{fmt.format(value) if known else '没有记录'}"

    return (
        show("review.completion", "本周完成率", "{:.0%}"),
        show("review.prev_completion", "上周完成率", "{:.0%}"),
        show("review.avg_rpe", "本周平均 RPE"),
        show("review.fatigue", "本周疲劳（1–5）"),
        show("review.fatigue_baseline", "前两次签到的疲劳平均"),
        show("review.sleep_quality", "本周睡眠质量（1–5）"),
    )
