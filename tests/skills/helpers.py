"""Skill 测试的公共部分：按画像与分层构造快照，用内置 Skill 跑一个 Agent。"""

import json
from typing import Any

from ironm.agents.base import Agent
from ironm.contracts.agents import AgentOutput
from ironm.contracts.common import deep_thaw
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


def snapshot(
    tier: Tier | None,
    *,
    matched: tuple[Tier, ...] = (),
    plan_fields: dict[str, Any] | None = None,
    constraints: tuple[dict[str, Any], ...] = (),
    diet_precision: str = "P1",
    review: dict[str, Any] | None = None,
    **profile: Any,
) -> Snapshot:
    plan: dict[str, Any] | None = None
    if plan_fields is not None:
        plan = {
            "version": 1,
            "parent": None,
            "fields": plan_fields,
            "diff": [],
            "rule_set_versions": {},
            "skill_versions": {},
            "created_at": "2026-10-20T00:00:00+00:00",
        }
    data: dict[str, Any] = {
        "snapshot_id": "s1",
        "user_id": "u1",
        "as_of": "2026-10-21T00:00:00+00:00",
        "timezone": "Asia/Shanghai",
        "tier": tier.value if tier else {"unknown": "insufficient_data"},
        "tier_matched": [t.value for t in matched],
        "precision_by_domain": {
            "training": "P1",
            "body": "P1",
            "diet": diet_precision,
            "recovery": "P1",
        },
        "profile": profile,
        "metrics": [],
        "current_plan": plan,
        "constraints": list(constraints),
        "prior_verdicts": [],
        "review": review or {},
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
    agent: AgentId,
    intent: str,
    tier: Tier | None,
    *,
    text: str = "",
    context_item: str | None = None,
    scene: ScenePack = ScenePack.FAT_LOSS,
    profile: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, AgentOutput]:
    """返回 {Skill 编号: 输出}；没有加载任何 Skill 时键为 no_skill。

    profile 单独传入（其余 kwargs 同 snapshot），避免与切换链路用的关键字参数冲突。
    """
    sel = select(
        SKILLS,
        agent=agent,
        intent=intent,
        scene=scene,
        tier=tier_result(tier),
    )
    outputs = Agent(agent).execute(
        snapshot(tier, **(profile or {}), **kwargs),
        sel,
        BOOK,
        text=text,
        context_item=context_item,
    )
    return {o.skill.split("@")[0]: o for o in outputs}


def proposed(output: AgentOutput) -> dict[str, Any]:
    """提案输出 → {字段: 新值}（只读映射转回普通的字典与列表，便于比较）。"""
    items = getattr(output, "items", ())
    return {p.field.value: deep_thaw(p.to_value) for p in items if hasattr(p, "to_value")}
