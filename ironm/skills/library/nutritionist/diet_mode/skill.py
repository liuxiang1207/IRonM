"""饮食模式（spec §30）：用户选过的优先，否则按饮食精度（R-NU-DIET-MODE）。"""

from collections.abc import Mapping
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ironm.contracts.agents import AgentOutput
from ironm.contracts.common import Domain
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import Change, proposals
from ironm.skills.view import SnapshotView

RULE = "R-NU-DIET-MODE"
type Mode = Literal["precise", "simple", "habit"]
MODES: tuple[Mode, ...] = ("precise", "simple", "habit")


class ModeParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    by_precision_p3: Mode
    by_precision_p2: Mode
    by_precision_p1: Mode
    by_precision_p0: Mode


RULE_PARAMS = {RULE: ModeParams}


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    p = context.rulebook.get(RULE).params_as(ModeParams)
    preference = view.get("profile.diet_mode_preference")
    precision = view.get("precision")
    level = str(precision.get(Domain.DIET, "P0")) if isinstance(precision, Mapping) else "P0"
    if preference in MODES:
        mode, fact = str(preference), f"用户选择：{preference}"
    else:
        mode = getattr(p, f"by_precision_{level.lower()}")
        fact = f"饮食精度 {level}"
    review = (view.as_of + timedelta(days=14)).date()
    change = Change(
        field="diet_mode",
        to_value=mode,
        reason_facts=(fact,),
        rule_id=RULE,
        expected_effect="记录负担与用户能坚持的程度相符",
        rollback_condition="两周内饮食记录明显减少或用户觉得太麻烦时，换成更轻的模式",
    )
    return proposals(context, view, [change], review)
