"""首期 10 个 Agent 及各自允许产出的输出类型。"""

from enum import StrEnum


class AgentId(StrEnum):
    ASSESSOR = "assessor"  # 评估师
    SAFETY_OFFICER = "safety_officer"  # 安全官
    DATA_ANALYST = "data_analyst"  # 数据分析师
    RECOVERY = "recovery"  # 恢复师
    TRAINING_PLANNER = "training_planner"  # 训练规划师
    EXERCISE_COACH = "exercise_coach"  # 动作教练
    NUTRITIONIST = "nutritionist"  # 营养师
    BEHAVIOR_COACH = "behavior_coach"  # 行为教练
    EVIDENCE_REVIEWER = "evidence_reviewer"  # 证据审查官
    QUALITY_REVIEWER = "quality_reviewer"  # 质量评审官


class OutputKind(StrEnum):
    CONSTRAINT = "constraint"
    PROPOSAL = "proposal"
    VERDICT = "verdict"
    NO_OPINION = "no_opinion"


_C, _P, _V, _N = OutputKind

AGENT_OUTPUT_KINDS: dict[AgentId, frozenset[OutputKind]] = {
    AgentId.ASSESSOR: frozenset({_C, _N}),
    AgentId.SAFETY_OFFICER: frozenset({_C, _V, _N}),
    AgentId.DATA_ANALYST: frozenset({_C, _N}),
    AgentId.RECOVERY: frozenset({_C, _N}),
    AgentId.TRAINING_PLANNER: frozenset({_P, _N}),
    AgentId.EXERCISE_COACH: frozenset({_P, _N}),
    AgentId.NUTRITIONIST: frozenset({_P, _N}),
    AgentId.BEHAVIOR_COACH: frozenset({_P, _N}),
    AgentId.EVIDENCE_REVIEWER: frozenset({_V, _N}),
    AgentId.QUALITY_REVIEWER: frozenset({_V, _N}),
}
