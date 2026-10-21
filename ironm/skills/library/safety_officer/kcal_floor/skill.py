"""热量底线（spec §29）：每日热量目标的下限约束，取自 R-CAL-FLOOR。"""

from ironm.contracts.agents import AgentOutput
from ironm.engine.calories import FLOOR, FloorParams
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import make_output
from ironm.skills.view import SnapshotView


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    p = context.rulebook.get(FLOOR).params_as(FloorParams)
    sex = view.get("profile.sex")
    if sex == "male":
        floor, reason = p.male, "floor_male"
    elif sex == "female":
        floor, reason = p.female, "floor_female"
    else:
        floor, reason = max(p.male, p.female), "floor_sex_unknown_conservative"
    item = {
        "field": "daily_kcal_target",
        "op": "min",
        "value": max(floor, p.absolute),
        "rule_id": FLOOR,
        "reason_code": reason,
    }
    return make_output(context, "constraint", items=[item])
