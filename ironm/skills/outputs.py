"""Skill 产出 Agent 输出的公共写法：自动填上所属 Agent 与 Skill 版本，按 JSON 严格校验。"""

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

from ironm.contracts.agents import AGENT_OUTPUT_ADAPTER, AgentOutput
from ironm.contracts.common import deep_thaw, json_equal
from ironm.contracts.plan import PlanVersion
from ironm.skills.loader import SkillContext
from ironm.skills.view import SnapshotView

__all__ = ["Change", "current_value", "make_output", "no_opinion", "proposals"]


def make_output(context: SkillContext, kind: str, **fields: Any) -> AgentOutput:
    m = context.skill.manifest
    data = {"agent": m.owner_agent.value, "skill": m.versioned, "kind": kind, **fields}
    return AGENT_OUTPUT_ADAPTER.validate_json(json.dumps(data, ensure_ascii=False))


def no_opinion(context: SkillContext, reason_code: str) -> AgentOutput:
    return make_output(context, "no_opinion", reason_code=reason_code)


def current_value(view: SnapshotView, field: str) -> Any:
    """当前计划中某字段的值（普通的列表与字典）；没有计划或没有该字段时为 None（首次设定）。"""
    plan = view.get("current_plan")
    if not isinstance(plan, PlanVersion):
        return None
    for key, value in plan.fields.items():
        if key == field:
            return deep_thaw(value)
    return None


@dataclass(frozen=True)
class Change:
    """一项拟议的修改与它的调整三件套。"""

    field: str
    to_value: Any
    reason_facts: tuple[str, ...]
    rule_id: str
    expected_effect: str
    rollback_condition: str


def proposals(
    context: SkillContext, view: SnapshotView, changes: list[Change], review_date: date
) -> AgentOutput:
    """把与当前计划不同的修改组成提案；全部相同时“本次无意见”（没有要改的）。"""
    items = []
    for c in changes:
        current = current_value(view, c.field)
        if current is not None and json_equal(current, c.to_value):
            continue
        items.append(
            {
                "proposal_id": f"{context.skill.manifest.id}:{c.field}",
                "field": c.field,
                "from_value": current,
                "to_value": c.to_value,
                "reason_facts": list(c.reason_facts),
                "rule_id": c.rule_id,
                "expected_effect": c.expected_effect,
                "rollback_condition": c.rollback_condition,
                "review_date": review_date.isoformat(),
                "evidence_ids": [],
            }
        )
    if not items:
        return no_opinion(context, "no_change")
    return make_output(context, "proposal", items=items)
