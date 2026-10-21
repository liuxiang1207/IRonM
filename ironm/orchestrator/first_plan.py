"""第一版计划（spec §31）：分层 → 追问 → 安全官约束 → 提案 → 复核 → 选动作 → 复核 → 计划。

最后为每条通过的提案生成依据卡，并给出基线测试方案。演示链路：路由按固定顺序，不含意图识别、盲审与持久化。
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ironm.agents.base import Agent
from ironm.contracts.agents import (
    AgentOutput,
    Constraint,
    ConstraintOutput,
    Proposal,
    ProposalOutput,
    Question,
    QuestionsOutput,
    Verdict,
)
from ironm.contracts.common import Domain, Precision, deep_thaw
from ironm.contracts.evidence import EvidenceCard
from ironm.contracts.plan import PlanVersion
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Screening, TierResult
from ironm.contracts.skills import ScenePack
from ironm.contracts.snapshot import Snapshot
from ironm.knowledge.baseline import Protocol, default_baseline
from ironm.orchestrator.evidence import build_cards
from ironm.orchestrator.review import review_proposal
from ironm.rules.loader import RuleBook
from ironm.safety.decision import decide
from ironm.safety.tiers import classify
from ironm.skills.loader import LoadedSkill, select

__all__ = ["FirstPlan", "first_plan"]

SCREENING_FIELDS = ("birth_date", "sex", "height_cm", "weight_kg", "waist_cm", "conditions")
BLOCKING = "no_plan_until_answered"  # 没回答就不能给计划的追问


@dataclass(frozen=True)
class FirstPlan:
    tier: TierResult
    notices: tuple[str, ...]  # 分层提示（三档输出）
    questions: tuple[Question, ...]
    constraints: tuple[Constraint, ...]
    verdicts: tuple[Verdict, ...]
    plan: PlanVersion | None  # 不可给计划、或停在追问时为空
    cards: tuple[EvidenceCard, ...]
    baseline: Protocol
    outputs: tuple[AgentOutput, ...]  # 全部 Agent 输出，供追溯


def _screening(profile: Mapping[str, Any]) -> Screening:
    data = {k: profile[k] for k in SCREENING_FIELDS if k in profile}
    return Screening.model_validate_json(json.dumps(data))


def _snapshot(
    profile: Mapping[str, Any],
    tier: TierResult,
    as_of: datetime,
    constraints: Iterable[Constraint] = (),
    plan: PlanVersion | None = None,
) -> Snapshot:
    data = {
        "snapshot_id": f"first-plan-{as_of.isoformat()}",
        "user_id": str(profile.get("user_id", "demo")),
        "as_of": as_of.isoformat(),
        "tier": tier.model_dump(mode="json")["tier"],
        "tier_matched": [t.value for t in tier.matched],
        "precision_by_domain": {d.value: Precision.P0.value for d in Domain},  # 新用户没有记录
        "profile": dict(profile),
        "metrics": [],
        "current_plan": None if plan is None else plan.model_dump(mode="json"),
        "constraints": [c.model_dump(mode="json") for c in constraints],
        "prior_verdicts": [],
    }
    return Snapshot.model_validate_json(json.dumps(data, ensure_ascii=False))


def _plan(
    changes: Iterable[Proposal], as_of: datetime, rulebook: RuleBook, skills: dict[str, str]
) -> PlanVersion:
    items = list(changes)
    data = {
        "version": 1,
        "parent": None,
        "fields": {p.field.value: deep_thaw(p.to_value) for p in items},
        "diff": [
            {"field": p.field.value, "from_value": None, "to_value": deep_thaw(p.to_value)}
            for p in items
        ],
        "rule_set_versions": rulebook.versions(),
        "skill_versions": skills,
        "created_at": as_of.isoformat(),
    }
    return PlanVersion.model_validate_json(json.dumps(data, ensure_ascii=False))


def first_plan(
    profile: Mapping[str, Any],
    *,
    as_of: datetime,
    tz: ZoneInfo,
    rulebook: RuleBook,
    skills: tuple[LoadedSkill, ...],
    unwell_today: bool = False,
    recent_injury: bool = False,
) -> FirstPlan:
    tier = classify(_screening(profile), as_of=as_of, tz=tz, rulebook=rulebook)
    notices = decide("", tier).notices
    baseline = default_baseline().protocol(
        tier.tier, unwell_today=unwell_today, recent_injury=recent_injury
    )
    outputs: list[AgentOutput] = []

    def run(agent: AgentId, snap: Snapshot) -> tuple[AgentOutput, ...]:
        sel = select(skills, agent=agent, intent="goal.create", scene=ScenePack.FAT_LOSS, tier=tier)
        result = Agent(agent).execute(snap, sel, rulebook)
        outputs.extend(result)
        return result

    base = _snapshot(profile, tier, as_of)
    questions = tuple(
        q for o in run(AgentId.ASSESSOR, base) if isinstance(o, QuestionsOutput) for q in o.items
    )
    constraints = tuple(
        c
        for o in run(AgentId.SAFETY_OFFICER, base)
        if isinstance(o, ConstraintOutput)
        for c in o.items
    )
    empty = FirstPlan(tier, notices, questions, constraints, (), None, (), baseline, ())
    if any(q.fallback == BLOCKING for q in questions):
        return FirstPlan(**{**empty.__dict__, "outputs": tuple(outputs)})

    verdicts: list[Verdict] = []
    accepted: list[Proposal] = []

    def review(results: Iterable[AgentOutput]) -> None:
        for o in results:
            if not isinstance(o, ProposalOutput):
                continue
            for p in o.items:
                verdict = review_proposal(p, constraints)
                verdicts.append(verdict)
                if verdict.result == "pass":
                    accepted.append(p)

    with_rules = _snapshot(profile, tier, as_of, constraints)
    review(run(AgentId.TRAINING_PLANNER, with_rules))
    review(run(AgentId.NUTRITIONIST, with_rules))
    used = {
        o.skill.split("@")[0]: o.skill.split("@")[1]
        for o in outputs
        if isinstance(o, ProposalOutput)
    }
    if accepted:  # 动作教练把草稿中的槽位落成具体动作
        draft = _plan(accepted, as_of, rulebook, used)
        review(run(AgentId.EXERCISE_COACH, _snapshot(profile, tier, as_of, constraints, draft)))
    used = {
        o.skill.split("@")[0]: o.skill.split("@")[1]
        for o in outputs
        if isinstance(o, ProposalOutput)
    }
    plan = _plan(accepted, as_of, rulebook, used) if accepted else None
    cards = build_cards(outputs, skills, rulebook, only=(p.proposal_id for p in accepted))
    return FirstPlan(
        tier=tier,
        notices=notices,
        questions=questions,
        constraints=constraints,
        verdicts=tuple(verdicts),
        plan=plan,
        cards=cards,
        baseline=baseline,
        outputs=tuple(outputs),
    )
