"""调度（spec [2026-10-21] §5）：决定本次调用哪些专业 Agent、按什么阶段执行。

来源有四种（写进运行记录）：
- fast_lane：只涉及单一领域、置信度达标，按规则路由表直接调用，不经管家；
- butler：管家（模型）给出主责、辅助与理由代码，并通过规则校验；
- rule_fallback：管家的调度没通过校验，丢弃后改按规则路由表，记录原因；
- rule_table：模型不可用（或没有配置模型）时一律按规则路由表。

规则校验：只能从白名单选；主责 1 个、辅助最多 2 个；安全官必须参与；不能调用审查类 Agent；
管家只能“组织回答 / 标为需进入计划调整 / 生成画像候选”，输出其他动作（如写入计划）即不通过。
执行顺序：同时有恢复师与训练类 Agent 时，恢复师单独作为第一阶段，先于训练执行。
"""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.common import ContractModel
from ironm.contracts.model_call import ModelStage
from ironm.contracts.roles import AgentId
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.orchestrator.intent import Intent, IntentResult
from ironm.rules.loader import RuleBook

__all__ = [
    "BUTLER_AGENTS",
    "DISPATCH_RULE",
    "DOMAIN_WEIGHTS",
    "MAX_ASSISTANTS",
    "ROUTES",
    "DispatchParams",
    "DispatchPlan",
    "check_butler",
    "domain_scores",
    "page_plan",
    "plan_dispatch",
    "rank",
    "route",
    "stages_for",
]

_A = AgentId
DISPATCH_RULE = "R-QA-DISPATCH"
MAX_ASSISTANTS = 2
PROMPT_PATH = Path(__file__).parent / "prompts" / "dispatch.md"
PROMPT_TEMPLATE = "dispatch@1.0.0"

# 规则路由表：意图 → (主责 Agent, 辅助 Agent)。安全官另行加入，不在此表
ROUTES: dict[Intent, tuple[AgentId, tuple[AgentId, ...]]] = {
    Intent.EXERCISE_TECHNIQUE: (_A.EXERCISE_COACH, ()),
    Intent.TRAINING_KNOWLEDGE: (_A.TRAINING_PLANNER, ()),
    Intent.PLAN: (_A.TRAINING_PLANNER, ()),
    Intent.NUTRITION: (_A.NUTRITIONIST, ()),
    Intent.SUPPLEMENTS_DRUGS: (_A.NUTRITIONIST, ()),
    Intent.PERSONAL_DATA: (_A.DATA_ANALYST, ()),
    Intent.LOGGING: (_A.DATA_ANALYST, ()),
    Intent.RECOVERY_SLEEP: (_A.RECOVERY, ()),
    Intent.HABIT_MOTIVATION: (_A.BEHAVIOR_COACH, ()),
    Intent.WELLNESS: (_A.NUTRITIONIST, (_A.RECOVERY,)),
}

# 管家可以选的专业 Agent（白名单）；审查类 Agent 只在计划链路中使用
BUTLER_AGENTS = frozenset(
    {
        _A.EXERCISE_COACH,
        _A.TRAINING_PLANNER,
        _A.NUTRITIONIST,
        _A.RECOVERY,
        _A.BEHAVIOR_COACH,
        _A.DATA_ANALYST,
        _A.SAFETY_OFFICER,
        _A.ASSESSOR,
    }
)
_REVIEWERS = frozenset({_A.EVIDENCE_REVIEWER, _A.QUALITY_REVIEWER})
_TRAINING = frozenset({_A.TRAINING_PLANNER, _A.EXERCISE_COACH})
# 管家可以输出的动作；其余（写入计划、记录、画像）一律不通过
BUTLER_ACTIONS = frozenset({"answer", "plan_change_note", "profile_candidate"})

type Source = Literal["fast_lane", "butler", "rule_fallback", "rule_table", "page"]


class DispatchParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fast_lane_min_confidence: Annotated[float, Field(gt=0, le=1)]
    # 领域打分（spec [2026-10-21] §5）：辅助 Agent 的加入门槛与相对主责的最低比例
    assistant_min_score: Annotated[float, Field(ge=0, le=1)] = 0.45
    assistant_min_ratio: Annotated[float, Field(ge=0, le=1)] = 0.55


# 意图 → 各领域的固有分（spec §5 领域打分）。安全相关的关注点由 _concerns_score 追加
DOMAIN_WEIGHTS: dict[Intent, dict[AgentId, float]] = {
    Intent.EXERCISE_TECHNIQUE: {_A.EXERCISE_COACH: 0.9, _A.TRAINING_PLANNER: 0.4},
    Intent.TRAINING_KNOWLEDGE: {_A.TRAINING_PLANNER: 0.9, _A.EXERCISE_COACH: 0.5},
    Intent.PLAN: {_A.TRAINING_PLANNER: 0.9, _A.NUTRITIONIST: 0.4, _A.RECOVERY: 0.3},
    Intent.NUTRITION: {_A.NUTRITIONIST: 0.9, _A.TRAINING_PLANNER: 0.3},
    Intent.SUPPLEMENTS_DRUGS: {_A.NUTRITIONIST: 0.85, _A.SAFETY_OFFICER: 0.5},
    Intent.PERSONAL_DATA: {_A.DATA_ANALYST: 0.9, _A.ASSESSOR: 0.3},
    Intent.LOGGING: {_A.DATA_ANALYST: 0.8, _A.BEHAVIOR_COACH: 0.4},
    Intent.RECOVERY_SLEEP: {_A.RECOVERY: 0.9, _A.BEHAVIOR_COACH: 0.4},
    Intent.HABIT_MOTIVATION: {_A.BEHAVIOR_COACH: 0.9, _A.RECOVERY: 0.3},
    Intent.WELLNESS: {_A.NUTRITIONIST: 0.7, _A.RECOVERY: 0.55},
    Intent.CHITCHAT: {},
    Intent.OUT_OF_SCOPE: {},
    Intent.UNKNOWN: {},
}


class DispatchPlan(ContractModel):
    source: Source
    primary: AgentId
    assistants: tuple[AgentId, ...]
    stages: tuple[tuple[AgentId, ...], ...]  # 按阶段执行；同一阶段并行
    reason_code: str | None = None  # 管家给出的理由代码，或校验失败的原因


class _Butler(BaseModel):
    """管家的输出格式（只校验形状；内容由 check_butler 按规则校验）。"""

    model_config = ConfigDict(extra="forbid")

    primary: str
    assistants: list[str] = []
    reason_code: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")]
    actions: list[str] = ["answer"]


def domain_scores(result: IntentResult) -> dict[AgentId, float]:
    """意图与领域 → 各专业 Agent 的分数（spec §5 领域打分）。

    关注点不参与打分：安全官、评估师的参与由 route 强制加入，见 _with_safety。
    安全不能靠分数竞争（spec §5“安全只升不降”）。
    """
    scores: dict[AgentId, float] = {}

    def add(agent: AgentId, value: float) -> None:
        scores[agent] = max(scores.get(agent, 0.0), value)

    for intent in set(result.domains) | {result.intent}:
        for agent, value in DOMAIN_WEIGHTS.get(intent, {}).items():
            add(agent, value)
    # 置信度整体缩放：拿不准时分数一起降低，主责不变但不容易带辅助
    return {a: round(v * result.confidence, 4) for a, v in scores.items()}


def rank(
    scores: dict[AgentId, float],
    params: DispatchParams,
    preferred: AgentId | None = None,
) -> tuple[AgentId, tuple[AgentId, ...]]:
    """分数最高者为主责；辅助取过门槛且不低于主责一定比例者，最多 2 个。

    同分时先取本次意图本身的领域（preferred），再按名称排序，保证结果稳定。
    """
    if not scores:
        return _A.SAFETY_OFFICER, ()
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0] is not preferred, kv[0].value))
    primary, top = ordered[0]
    floor = max(params.assistant_min_score, top * params.assistant_min_ratio)
    picked = [a for a, v in ordered[1:] if v >= floor][:MAX_ASSISTANTS]
    return primary, tuple(picked)


_MANDATORY = frozenset({_A.SAFETY_OFFICER, _A.ASSESSOR})


def _fit(primary: AgentId, assistants: list[AgentId]) -> tuple[AgentId, ...]:
    """必要时加入的 Agent 排在最前；超出上限时先去掉非必要的辅助。"""
    unique = [a for a in dict.fromkeys(assistants) if a != primary]
    while len(unique) > MAX_ASSISTANTS:
        drop = next(a for a in reversed(unique) if a not in _MANDATORY)
        unique.remove(drop)
    return tuple(unique)


def route(
    result: IntentResult, params: DispatchParams | None = None
) -> tuple[AgentId, tuple[AgentId, ...]]:
    """领域打分 → 主责 + 辅助（安全官必在，特殊人群加评估师，辅助最多 2 个）。"""
    p = params or DispatchParams(fast_lane_min_confidence=0.7)
    own = DOMAIN_WEIGHTS.get(result.intent, {})
    preferred = max(own, key=lambda a: own[a]) if own else None
    primary, scored = rank(domain_scores(result), p, preferred)
    # 安全官每条路由都参与；特殊人群再加评估师。两者不参与打分，也不会被挤掉
    mandatory: list[AgentId] = []
    if "special_population" in result.concerns:
        mandatory.append(_A.ASSESSOR)
    if primary != _A.SAFETY_OFFICER:
        mandatory.append(_A.SAFETY_OFFICER)
    return primary, _fit(primary, [*mandatory, *scored])


def stages_for(
    primary: AgentId, assistants: tuple[AgentId, ...]
) -> tuple[tuple[AgentId, ...], ...]:
    agents = (primary, *assistants)
    if _A.RECOVERY in agents and _TRAINING & set(agents):
        rest = tuple(a for a in agents if a != _A.RECOVERY)
        return ((_A.RECOVERY,), rest)
    return (agents,)


def check_butler(out: _Butler) -> tuple[AgentId, tuple[AgentId, ...]] | str:
    """按规则校验管家的调度；通过时返回 (主责, 辅助)，否则返回原因代码。"""
    if set(out.actions) - BUTLER_ACTIONS:
        return "forbidden_action"
    names = [out.primary, *out.assistants]
    try:
        agents = [AgentId(n) for n in names]
    except ValueError:
        return "unknown_agent"
    if set(agents) & _REVIEWERS:
        return "reviewer_not_allowed"
    if set(agents) - BUTLER_AGENTS:
        return "not_in_whitelist"
    if len(set(agents)) != len(agents):
        return "duplicate_agent"
    if len(out.assistants) > MAX_ASSISTANTS:
        return "too_many_assistants"
    if _A.SAFETY_OFFICER not in agents:
        return "missing_safety_officer"
    return agents[0], tuple(agents[1:])


def page_plan(primary: AgentId) -> DispatchPlan:
    """功能页提问：只调用本功能的主责 Agent 与安全官，不经管家调度。"""
    assistants = () if primary == _A.SAFETY_OFFICER else (_A.SAFETY_OFFICER,)
    return _plan("page", primary, assistants, None)


def _plan(
    source: Source, primary: AgentId, assistants: tuple[AgentId, ...], reason: str | None
) -> DispatchPlan:
    return DispatchPlan(
        source=source,
        primary=primary,
        assistants=assistants,
        stages=stages_for(primary, assistants),
        reason_code=reason,
    )


def plan_dispatch(
    text: str,
    intent: IntentResult,
    *,
    rulebook: RuleBook | None,
    gateway: Gateway | None,
    request_id: str,
    context_item: str | None = None,
) -> DispatchPlan:
    if gateway is None or rulebook is None:
        primary, assistants = route(intent)
        return _plan("rule_table", primary, assistants, None)
    p = rulebook.get(DISPATCH_RULE).params_as(DispatchParams)
    primary, assistants = route(intent, p)
    if len(intent.domains) <= 1 and intent.confidence >= p.fast_lane_min_confidence:
        return _plan("fast_lane", primary, assistants, None)
    payload = {
        "message": text,
        "intent": intent.intent.value,
        "domains": [d.value for d in intent.domains],
        "concerns": list(intent.concerns),
        "context_item": context_item,
        "agents": sorted(a.value for a in BUTLER_AGENTS),
    }
    result = gateway.complete(
        ModelStage.DISPATCH,
        request_id=request_id,
        template=PROMPT_TEMPLATE,
        messages=[
            Message("system", PROMPT_PATH.read_text(encoding="utf-8")),
            Message("user", json.dumps(payload, ensure_ascii=False)),
        ],
        validate=json_model(_Butler),
    )
    if result.value is None:
        return _plan("rule_table", primary, assistants, None)
    checked = check_butler(result.value)
    if isinstance(checked, str):
        return _plan("rule_fallback", primary, assistants, f"dispatch_invalid:{checked}")
    return _plan("butler", checked[0], checked[1], result.value.reason_code)
