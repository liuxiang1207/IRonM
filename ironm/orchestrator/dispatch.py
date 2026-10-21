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
    "MAX_ASSISTANTS",
    "ROUTES",
    "DispatchParams",
    "DispatchPlan",
    "check_butler",
    "page_plan",
    "plan_dispatch",
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


def route(result: IntentResult) -> tuple[AgentId, tuple[AgentId, ...]]:
    """规则路由表：主责 + 辅助（安全官必在，特殊人群加评估师，辅助最多 2 个）。"""
    primary, extra = ROUTES[result.intent]
    assistants = [*extra, _A.SAFETY_OFFICER]
    if "special_population" in result.concerns:
        assistants.append(_A.ASSESSOR)
    unique = [a for a in dict.fromkeys(assistants) if a != primary]
    # 安全官必须保留；超过上限时先去掉排在前面的普通辅助 Agent
    while len(unique) > MAX_ASSISTANTS:
        unique.remove(next(a for a in unique if a not in (_A.SAFETY_OFFICER, _A.ASSESSOR)))
    return primary, tuple(unique)


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
    primary, assistants = route(intent)
    if gateway is None or rulebook is None:
        return _plan("rule_table", primary, assistants, None)
    p = rulebook.get(DISPATCH_RULE).params_as(DispatchParams)
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
