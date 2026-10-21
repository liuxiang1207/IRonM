"""分层限制（spec §29）：按分层给训练约束，参数取自 R-SF-TIER-LIMITS。"""

from typing import Any

from ironm.contracts.agents import AgentOutput
from ironm.contracts.screening import Tier
from ironm.safety.decision import plan_allowed
from ironm.safety.tiers import TIER_LIMITS_RULE, TierLimitsParams
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import make_output, no_opinion
from ironm.skills.view import SnapshotView

CONSERVATIVE = frozenset({Tier.B, Tier.COLLABORATIVE})


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    tier = view.get("tier")
    if not isinstance(tier, Tier) or not plan_allowed(tier):
        return no_opinion(context, "no_plan_tier")
    p = context.rulebook.get(TIER_LIMITS_RULE).params_as(TierLimitsParams)
    conservative = tier in CONSERVATIVE

    def item(field: str, op: str, value: Any, reason: str) -> dict[str, Any]:
        return {
            "field": field,
            "op": op,
            "value": value,
            "rule_id": TIER_LIMITS_RULE,
            "reason_code": reason,
        }

    max_rpe = p.b_max_rpe if conservative else p.a_max_rpe
    step = p.b_cardio_step_min if conservative else p.a_cardio_step_min
    items = [
        item("intensity_target", "max", {"rpe": max_rpe}, f"tier_{tier}_max_rpe"),
        item(
            "progression_rule", "max", {"cardio_minutes_per_week": step}, f"tier_{tier}_cardio_step"
        ),
    ]
    if conservative:
        cap = p.b_cardio_cap_min
        items.append(item("weekly_cardio_minutes", "max", cap, f"tier_{tier}_cardio_cap"))
    return make_output(context, "constraint", items=items)
