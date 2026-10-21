"""行为教练 · 作息（spec [2026-10-21] §15）：称重日、签到日与提醒时间（R-HB-ROUTINE）。"""

import re
from datetime import timedelta
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from ironm.contracts.agents import AgentOutput, Constraint
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import Change, proposals
from ironm.skills.view import SnapshotView

RULE = "R-HB-ROUTINE"
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
WEEKDAY = "一二三四五六日"


class RoutineParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    weigh_in_days: tuple[int, ...]
    checkin_day: int
    weigh_in_time: str
    checkin_time: str
    followup_time: str
    review_days: int

    @model_validator(mode="after")
    def _check(self) -> Self:
        days = self.weigh_in_days
        if not days or len(set(days)) != len(days) or not all(0 <= d <= 6 for d in days):
            raise ValueError("称重日应是不重复的星期几（0–6）")
        if not 0 <= self.checkin_day <= 6:
            raise ValueError("签到日应是星期几（0–6）")
        times = (self.weigh_in_time, self.checkin_time, self.followup_time)
        if not all(_TIME_RE.fullmatch(t) for t in times):
            raise ValueError("提醒时间应是 HH:MM")
        return self


RULE_PARAMS = {RULE: RoutineParams}


def _weigh_cap(constraints: tuple[Constraint, ...]) -> int | None:
    """安全官给的每周称重次数上限（取最严的）。"""
    caps = [
        int(c.value)
        for c in constraints
        if c.field == "weigh_in_days" and c.op == "max" and isinstance(c.value, int | float)
    ]
    return min(caps) if caps else None


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    p = context.rulebook.get(RULE).params_as(RoutineParams)
    constraints = view.get("constraints")
    rules: tuple[Constraint, ...] = constraints if isinstance(constraints, tuple) else ()
    cap = _weigh_cap(rules)
    days = list(p.weigh_in_days) if cap is None else list(p.weigh_in_days)[: max(cap, 0)]
    named = "、".join(f"周{WEEKDAY[d]}" for d in days) or "不安排"
    review = (view.as_of + timedelta(days=p.review_days)).date()
    changes = [
        Change(
            field="weigh_in_days",
            to_value=days,
            reason_facts=(
                f"固定称重日：{named}早上空腹",
                f"安全官的称重次数上限：{'无' if cap is None else f'每周 {cap} 次'}",
            ),
            rule_id=RULE,
            expected_effect="固定时间称重，趋势判断更稳，也不用每天称",
            rollback_condition="称重让人焦虑或连续两周漏称时，减少次数或改到更方便的日子",
        ),
        Change(
            field="checkin_day",
            to_value=p.checkin_day,
            reason_facts=(f"每周签到：周{WEEKDAY[p.checkin_day]}晚上，约 1 分钟",),
            rule_id=RULE,
            expected_effect="每周回顾一次疲劳、睡眠与执行情况，复盘有据可依",
            rollback_condition="连续两周没签到时，换到用户更方便的日子",
        ),
        Change(
            field="reminder_schedule",
            to_value={
                "weigh_in": p.weigh_in_time,
                "checkin": p.checkin_time,
                "followup": p.followup_time,
                "max_per_day": 1,
            },
            reason_facts=(
                "只在称重日早上、签到日晚上、漏打卡或没签到的第二天早上提醒，每天最多 1 条",
            ),
            rule_id=RULE,
            expected_effect="提醒少而准，不打扰",
            rollback_condition="用户觉得提醒多了或时间不合适时，按用户的选择调整",
        ),
    ]
    return proposals(context, view, changes, review)
