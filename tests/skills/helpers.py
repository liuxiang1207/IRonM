"""Skill 测试的公共部分：按画像与分层构造快照，用内置 Skill 跑一个 Agent。"""

import json
from typing import Any

from ironm.agents.base import Agent
from ironm.contracts.agents import AgentOutput
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Tier, TierResult
from ironm.contracts.skills import ScenePack
from ironm.contracts.snapshot import Snapshot
from ironm.contracts.unknown import Unknown, UnknownReason
from ironm.rules.loader import RULES_DIR, load_rules
from ironm.skills.loader import load_skills, select

BOOK = load_rules(RULES_DIR)
SKILLS = load_skills()

FULL_PROFILE: dict[str, Any] = {
    "conditions": [],
    "birth_date": "1990-06-15",
    "height_cm": 165.0,
    "weight_kg": 72.0,
    "sex": "female",
    "activity": "sedentary",
    "waist_cm": 86.0,
}


def snapshot(tier: Tier | None, **profile: Any) -> Snapshot:
    data: dict[str, Any] = {
        "snapshot_id": "s1",
        "user_id": "u1",
        "as_of": "2026-10-21T00:00:00+00:00",
        "tier": tier.value if tier else {"unknown": "insufficient_data"},
        "precision_by_domain": {"training": "P1", "body": "P1", "diet": "P1", "recovery": "P1"},
        "profile": profile,
        "metrics": [],
        "current_plan": None,
        "constraints": [],
        "prior_verdicts": [],
    }
    return Snapshot.model_validate_json(json.dumps(data))


def tier_result(tier: Tier | None) -> TierResult:
    value: Tier | Unknown = tier if tier else Unknown(unknown=UnknownReason.INSUFFICIENT_DATA)
    return TierResult(
        tier=value,
        status="ok" if tier else "insufficient",
        matched=(tier,) if tier else (),
        reasons=(),
        missing=(),
        bmi=25.0,
        rule_id="R-SF-TIER",
    )


def run_agent(
    agent: AgentId, intent: str, tier: Tier | None, **profile: Any
) -> dict[str, AgentOutput]:
    """返回 {Skill 编号: 输出}；没有加载任何 Skill 时键为 no_skill。"""
    sel = select(
        SKILLS, agent=agent, intent=intent, scene=ScenePack.FAT_LOSS, tier=tier_result(tier)
    )
    outputs = Agent(agent).execute(snapshot(tier, **profile), sel, BOOK)
    return {o.skill.split("@")[0]: o for o in outputs}
