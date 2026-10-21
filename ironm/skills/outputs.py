"""Skill 产出 Agent 输出的公共写法：自动填上所属 Agent 与 Skill 版本，按 JSON 严格校验。"""

import json
from typing import Any

from ironm.contracts.agents import AGENT_OUTPUT_ADAPTER, AgentOutput
from ironm.skills.loader import SkillContext

__all__ = ["make_output", "no_opinion"]


def make_output(context: SkillContext, kind: str, **fields: Any) -> AgentOutput:
    m = context.skill.manifest
    data = {"agent": m.owner_agent.value, "skill": m.versioned, "kind": kind, **fields}
    return AGENT_OUTPUT_ADAPTER.validate_json(json.dumps(data, ensure_ascii=False))


def no_opinion(context: SkillContext, reason_code: str) -> AgentOutput:
    return make_output(context, "no_opinion", reason_code=reason_code)
