"""按 spec 的决策表核对每次复盘（spec [2026-10-21] §17，计划 D40）。

独立于复盘代码与规则表：只读决策日志里记下的事实与修改，按 spec 的训练、饮食两张决策表推出应命中
的那一行，再看实际修改的方向与幅度是否与这一行一致。表中的数值也直接取 spec 写明的值（不读规则表），
规则表与 spec 不一致时，模拟会报出来。
"""

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["SPEC_DIET", "SPEC_WEEKLY", "check_review", "diet_row", "training_row"]

CARDIO, SLOTS, KCAL = "weekly_cardio_minutes", "session_slots", "daily_kcal_target"
# 有氧每次最多加一档（A、A′ 30 分钟，B、协作 20 分钟），总量不超过每周 300 分钟（R-SF-TIER-LIMITS、
# R-TR-START）；这里按最宽的一档核对
STEP_MAX, CAP = 30, 300
# spec §17 两张决策表里写明的数值
SPEC_WEEKLY: dict[str, float] = {
    "fatigue_rise": 2,
    "sleep_low": 2,
    "rpe_high": 9,
    "low_completion": 0.5,
    "high_completion": 0.9,
    "rpe_ok": 7,
    "cardio_cut_pct": 20,
    "max_sets": 3,
    "max_reps_high": 15,
}
SPEC_DIET: dict[str, float] = {
    "min_meal_days_per_week": 4, "min_weigh_ins_per_week": 1, "max_ci_half_width": 0.25,
    "low_completion": 0.5, "low_adherence_days": 3, "ok_completion": 0.7, "ok_adherence_days": 4,
    "step_kcal": 150, "fast_loss_kg_per_week": 1.0, "fast_loss_after_days": 28,
    "target_low_a": 0.5, "target_low_a_prime": 0.25,
}  # fmt: skip
TRAINING_FIELDS = frozenset({CARDIO, SLOTS})
DIET_FIELDS = frozenset({KCAL})
# 每一行允许的修改方向（spec §17 训练决策表）
ALLOWED: dict[str, dict[str, set[str]]] = {
    "recovery": {CARDIO: {"down"}, SLOTS: {"down"}},
    "rpe_high": {SLOTS: {"down"}},
    "low_completion": {CARDIO: {"down"}},
    "progress": {CARDIO: {"up"}, SLOTS: {"up"}},
}


def training_row(this: Mapping[str, Any], last: Mapping[str, Any], p: Mapping[str, Any]) -> str:
    """按顺序取第一条命中的行；记录不全或数据不足时不调整。"""
    if this.get("partial_week"):
        return "partial_week"
    completion = this.get("completion")
    if completion is None:
        return "insufficient"
    fatigue, base = this.get("fatigue"), this.get("fatigue_baseline")
    sleep, rpe, prev = this.get("sleep_quality"), this.get("avg_rpe"), last.get("completion")
    if (fatigue is not None and base is not None and fatigue - base >= p["fatigue_rise"]) or (
        sleep is not None and sleep <= p["sleep_low"]
    ):
        return "recovery"
    if rpe is not None and rpe >= p["rpe_high"]:
        return "rpe_high"
    if completion < p["low_completion"] and prev is not None and prev < p["low_completion"]:
        return "low_completion"
    good_effort = (rpe is not None and rpe <= p["rpe_ok"]) or (
        rpe is None and prev is not None and prev >= p["high_completion"]
    )
    if completion >= p["high_completion"] and good_effort:
        return "progress"
    return "keep"


def diet_row(f: Mapping[str, Any], p: Mapping[str, Any], tier: str | None) -> str:
    """饮食决策表（只对减脂）。"""
    if f.get("red_flag"):
        return "red_flag"
    loss, half = f.get("loss_kg_per_week"), f.get("ci_half_width")
    if (
        f.get("meal_days_min", 0) < p["min_meal_days_per_week"]
        or f.get("weigh_ins_min", 0) < p["min_weigh_ins_per_week"]
        or loss is None
        or half is None
        or half > p["max_ci_half_width"]
    ):
        return "insufficient"
    completion, adherence = f.get("completion"), f.get("adherence_days")
    if (completion is not None and completion < p["low_completion"]) or (
        adherence is not None and adherence < p["low_adherence_days"]
    ):
        return "ease_plan"
    target_low = p["target_low_a_prime"] if tier == "a_prime" else p["target_low_a"]

    def slow(value: Any) -> bool:
        return isinstance(value, int | float) and value < target_low / 2

    if (
        completion is not None
        and completion >= p["ok_completion"]
        and adherence is not None
        and adherence >= p["ok_adherence_days"]
        and slow(loss)
        and slow(f.get("prev_loss_kg_per_week"))
    ):
        return "cut"
    if (
        f.get("days_since_start", 0) >= p["fast_loss_after_days"]
        and loss > p["fast_loss_kg_per_week"]
    ):
        return "raise"
    return "keep"


def _direction(before: float, after: float) -> str:
    return "up" if after > before else "down" if after < before else "same"


def _slots_direction(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> str:
    moves = {_direction(b["sets"], a["sets"]) for b, a in zip(before, after, strict=False)} | {
        _direction(b["reps"][1], a["reps"][1]) for b, a in zip(before, after, strict=False)
    }
    moves.discard("same")
    if len(before) != len(after) or len(moves) > 1:
        return "mixed"
    return moves.pop() if moves else "same"


def _slot_problems(
    row: str, before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]],
    p: Mapping[str, Any],
) -> list[str]:  # fmt: skip
    out = []
    for b, a in zip(before, after, strict=False):
        if abs(a["sets"] - b["sets"]) > 1:
            out.append(f"组数一次变了 {a['sets'] - b['sets']} 组")
        if a["sets"] < 1:
            out.append("组数低于 1")
        if row == "progress":
            if a["sets"] > p["max_sets"] or a["reps"][1] > p["max_reps_high"]:
                out.append("进阶超过上限（组数或次数）")
            if b["sets"] < p["max_sets"] and a["reps"][1] > b["reps"][1]:
                out.append("没先加组就加了次数")
    return out


def check_review(
    record: Mapping[str, Any],
    *,
    weekly: Mapping[str, Any],
    diet: Mapping[str, Any],
    goal: str,
    tier: str | None,
    start_cardio: float | None,
) -> tuple[str, str | None, list[str]]:
    """返回（训练行, 饮食行, 不符合决策表之处）。record 是一条每周复盘的完整决策日志。"""
    facts = record.get("facts", {})
    this, last = facts.get("this_week", {}), facts.get("last_week", {})
    row = training_row(this, last, weekly)
    allowed = {k: set(v) for k, v in ALLOWED.get(row, {}).items()}
    if this.get("menstruation"):
        allowed = {k: v - {"down"} for k, v in allowed.items()}  # 经期不下调
    problems: list[str] = []
    changes = {c["field"]: c for c in record.get("changes", [])}
    for name, c in changes.items():
        if name not in TRAINING_FIELDS:
            continue
        if name == CARDIO:
            move = _direction(c["from"], c["to"])
        else:
            move = _slots_direction(c["from"], c["to"])
            problems += _slot_problems(row, c["from"], c["to"], weekly)
        if move not in allowed.get(name, set()):
            problems.append(f"训练命中“{row}”，却改了 {name}（{move}）")
        if name == CARDIO and move == "down":
            # 减 20%，时长按 5 分钟取整：与准确值相差不超过 2.5 分钟；完成率低时不低于起点
            exact = c["from"] * (1 - weekly["cardio_cut_pct"] / 100)
            at_start = row == "low_completion" and c["to"] == start_cardio
            if abs(c["to"] - exact) > 2.5 and not at_start:
                problems.append(
                    f"有氧没有按 {weekly['cardio_cut_pct']}% 下调：{c['from']}→{c['to']}"
                )
            if row == "low_completion" and start_cardio is not None and c["to"] < start_cardio:
                problems.append("有氧减到了起点以下")
        if name == CARDIO and move == "up" and (c["to"] - c["from"] > STEP_MAX or c["to"] > CAP):
            problems.append(f"有氧加得太多：{c['from']}→{c['to']}")
    diet_seen = None
    if record.get("diet_review"):
        if goal != "fat_loss":
            problems.append("非减脂用户做了饮食复盘")
        diet_seen = diet_row(record.get("diet_facts", {}), diet, tier)
    kcal = changes.get(KCAL)
    if kcal is not None:
        step = kcal["to"] - kcal["from"]
        expected = {"cut": -diet["step_kcal"], "raise": diet["step_kcal"]}.get(diet_seen or "")
        if goal != "fat_loss" or tier not in ("a", "a_prime"):
            problems.append("不该调整热量的用户调整了热量")
        elif step != expected:
            problems.append(f"饮食命中“{diet_seen}”，热量却改了 {step:+}")
    unknown = sorted(set(changes) - TRAINING_FIELDS - DIET_FIELDS)
    if unknown:  # 两张决策表只调有氧、组数次数与每日热量；其余字段复盘不该动
        problems.append(f"决策表里没有这些项的修改：{unknown}")
    return row, diet_seen, problems
