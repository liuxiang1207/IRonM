"""饮食关系警示（spec §29）：有进食障碍史或信号时，不设热量缺口、每周最多称重 1 次。"""

from ironm.contracts.agents import AgentOutput
from ironm.contracts.screening import Condition
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import make_output, no_opinion
from ironm.skills.view import SnapshotView

RULE_ID = "R-SF-TIER"  # 进食障碍相关状况的归属取自分层规则
SIGNALS = frozenset({Condition.EATING_DISORDER_HISTORY, Condition.EATING_DISORDER_SIGNAL})


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    present: set[object] = set()
    for key in ("profile.conditions", "profile.derived_conditions"):  # 问卷答案与聊天推出的
        value = view.get(key)
        if isinstance(value, list | tuple):
            present |= set(value)
    if not present & SIGNALS:
        return no_opinion(context, "no_signal")
    items = [
        {
            "field": "daily_kcal_target",
            "op": "exclude",
            "value": "deficit",
            "rule_id": RULE_ID,
            "reason_code": "eating_relationship_no_deficit",
        },
        {
            "field": "weigh_in_days",
            "op": "max",
            "value": 1,
            "rule_id": RULE_ID,
            "reason_code": "eating_relationship_weigh_weekly_at_most",
        },
    ]
    return make_output(context, "constraint", items=items)
