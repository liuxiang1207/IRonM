"""建档追问（spec §29）：缺什么、为什么、没回答时怎么办；每次最多 3 项。"""

from ironm.contracts.agents import MAX_QUESTIONS, AgentOutput
from ironm.contracts.unknown import Unknown
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import make_output, no_opinion
from ironm.skills.view import SnapshotView

# (信息, 为什么需要, 没回答时怎么办)，按对安全与计划结构的影响排序
ORDER: tuple[tuple[str, str, str], ...] = (
    ("profile.conditions", "needed_for_tier", "no_plan_until_answered"),
    ("profile.birth_date", "needed_for_tier", "no_plan_until_answered"),
    ("profile.height_cm", "needed_for_tier", "no_plan_until_answered"),
    ("profile.weight_kg", "needed_for_tier", "no_plan_until_answered"),
    ("profile.sex", "needed_for_calories", "no_calorie_target_until_answered"),
    ("profile.activity", "needed_for_calories", "assume_sedentary"),
    ("profile.waist_cm", "needed_for_tier", "assume_a_prime"),
)


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    missing = [
        {"item": item, "reason_code": reason, "fallback": fallback}
        for item, reason, fallback in ORDER
        if isinstance(view.get(item), Unknown)
    ]
    if not missing:
        return no_opinion(context, "info_sufficient")
    return make_output(context, "questions", items=missing[:MAX_QUESTIONS])
