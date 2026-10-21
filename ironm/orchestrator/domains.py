"""查知识的领域（spec [2026-10-21] §12）：按领域打分决定查哪些领域的问答卡。

- 每个意图给各领域一个固有分，乘以意图的置信度；分数最高者为主领域。
- 相关领域：过门槛、且不低于主领域一定比例的，最多 2 个。拿不准时分数一起降低，不容易带上相关领域。
- 安全类的卡不参与打分，每次都查（安全不靠分数竞争）。
领域用负责该领域内容的专业角色表示（与问答卡的 agent 字段一致）。
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.common import ContractModel
from ironm.contracts.roles import AgentId
from ironm.orchestrator.intent import Intent, IntentResult
from ironm.rules.loader import RuleBook

__all__ = [
    "DOMAIN_RULE",
    "DOMAIN_WEIGHTS",
    "MAX_RELATED",
    "DomainParams",
    "Domains",
    "select_domains",
]

_A = AgentId
DOMAIN_RULE = "R-QA-DOMAINS"
MAX_RELATED = 2

# 意图 → 各领域的固有分。安全类不在此表：安全类的卡每次都查
DOMAIN_WEIGHTS: dict[Intent, dict[AgentId, float]] = {
    Intent.EXERCISE_TECHNIQUE: {_A.EXERCISE_COACH: 0.9, _A.TRAINING_PLANNER: 0.4},
    Intent.TRAINING_KNOWLEDGE: {_A.TRAINING_PLANNER: 0.9, _A.EXERCISE_COACH: 0.5},
    Intent.PLAN: {_A.TRAINING_PLANNER: 0.9, _A.NUTRITIONIST: 0.4, _A.RECOVERY: 0.3},
    Intent.NUTRITION: {_A.NUTRITIONIST: 0.9, _A.TRAINING_PLANNER: 0.3},
    Intent.SUPPLEMENTS_DRUGS: {_A.NUTRITIONIST: 0.85},
    Intent.PERSONAL_DATA: {_A.DATA_ANALYST: 0.9},
    Intent.LOGGING: {_A.DATA_ANALYST: 0.8, _A.BEHAVIOR_COACH: 0.4},
    Intent.RECOVERY_SLEEP: {_A.RECOVERY: 0.9, _A.BEHAVIOR_COACH: 0.4},
    Intent.HABIT_MOTIVATION: {_A.BEHAVIOR_COACH: 0.9, _A.RECOVERY: 0.3},
    Intent.WELLNESS: {_A.NUTRITIONIST: 0.7, _A.RECOVERY: 0.55},
}


class DomainParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    related_min_score: Annotated[float, Field(ge=0, le=1)]
    related_min_ratio: Annotated[float, Field(ge=0, le=1)]


DEFAULT_PARAMS = DomainParams(related_min_score=0.45, related_min_ratio=0.55)


class Domains(ContractModel):
    """主领域与相关领域，以及各领域得分（写进运行记录）。"""

    primary: AgentId | None  # 意图归不进任何领域时为空，只查安全类的卡
    related: tuple[AgentId, ...]
    scores: tuple[tuple[AgentId, float], ...]


def _scores(result: IntentResult) -> dict[AgentId, float]:
    scores: dict[AgentId, float] = {}
    for intent in {*result.domains, result.intent}:
        for role, value in DOMAIN_WEIGHTS.get(intent, {}).items():
            scores[role] = max(scores.get(role, 0.0), value)
    # 置信度整体缩放：拿不准时分数一起降低，主领域不变但不容易带上相关领域
    return {r: round(v * result.confidence, 4) for r, v in scores.items()}


def select_domains(result: IntentResult, rulebook: RuleBook | None = None) -> Domains:
    p = DEFAULT_PARAMS if rulebook is None else rulebook.get(DOMAIN_RULE).params_as(DomainParams)
    scores = _scores(result)
    if not scores:
        return Domains(primary=None, related=(), scores=())
    own = DOMAIN_WEIGHTS.get(result.intent, {})
    preferred = max(own, key=lambda r: own[r]) if own else None
    # 同分时先取本次意图本身的领域，再按名称，保证结果稳定
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0] is not preferred, kv[0].value))
    primary, top = ordered[0]
    floor = max(p.related_min_score, top * p.related_min_ratio)
    related = tuple(r for r, v in ordered[1:] if v >= floor)[:MAX_RELATED]
    return Domains(primary=primary, related=related, scores=tuple(ordered))
