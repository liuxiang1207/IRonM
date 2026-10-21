"""问答编排（spec [2026-10-21] §5–§7）：路由 → 收集（并行、互不可见）→ 组织回答（L0 / L1 / 一般）。

- 路由：按意图选主 Agent；安全官每条路由都参与；特殊人群再加评估师；辅助 Agent 最多 2 个。
- 收集：各 Agent 并行运行，只拿到问题、意图与关注点，看不到彼此的输出；只能用白名单里的工具；
  单个 Agent 出错或越权时为“本次无意见”，其余照常。
- 组织回答：
  - L0：命中问答卡、又没有要加的安全提示时，直接用最具体的卡（主 Agent 的优先；不调用模型）；
  - L1：需要把安全提示融进回答时，由模型按统一结构表达，数字必须有登记来源，
    不合规时重写 1 次，仍不合规则退到 L0；
  - 一般：没有审定知识时，不给具体建议。降级都告诉用户，且不跳过安全检查。
- 计划相关的问题只作说明、标为“需进入计划调整”，不改计划；
  但命中安全官的卡时（如“帮我安排 7 天断食”），先按卡回答。
运行记录（RunRecord）不含用户原话，可按请求编号回放。
"""

import json
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.common import ContractModel
from ironm.contracts.model_call import ModelStage
from ironm.contracts.roles import AgentId
from ironm.core.body import bmi
from ironm.core.log import get_logger
from ironm.knowledge.qa_cards import CardBook, PartKind, QACard
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.orchestrator.intent import Concern, Intent, IntentResult
from ironm.orchestrator.numbers import numbers_in, unregistered
from ironm.safety.matching import clauses

__all__ = [
    "MAX_ASSISTANTS",
    "ROUTES",
    "TOOLS",
    "AnswerPart",
    "Fact",
    "QAResult",
    "RunRecord",
    "Step",
    "answer_question",
    "route",
]

_A = AgentId
MAX_ASSISTANTS = 2
ANSWER_TEMPLATE = "answer_l1@1.0.0"
PROMPT_PATH = Path(__file__).parent / "prompts" / "answer_l1.md"

# 意图 → (主 Agent, 辅助 Agent)。安全官另行加入，不在此表
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

# 每个 Agent 可以调用的工具（白名单）
TOOLS: dict[AgentId, frozenset[str]] = {
    _A.EXERCISE_COACH: frozenset({"search_cards"}),
    _A.TRAINING_PLANNER: frozenset({"search_cards"}),
    _A.NUTRITIONIST: frozenset({"search_cards"}),
    _A.RECOVERY: frozenset({"search_cards"}),
    _A.BEHAVIOR_COACH: frozenset({"search_cards"}),
    _A.DATA_ANALYST: frozenset({"search_cards"}),
    _A.SAFETY_OFFICER: frozenset({"search_cards"}),
    _A.ASSESSOR: frozenset(),
}

_log = get_logger("ironm.qa")


class ToolDenied(Exception):
    pass


class Fact(ContractModel):
    """Agent 拿出的一条依据：问答卡、安全提示，或用户自己说的数字算出的结果（personal）。

    values 只在本次组织回答时使用，不写入运行记录（可能是健康数据）。
    """

    kind: Literal["card", "caution", "personal"]
    source_id: Annotated[str, Field(min_length=1)]
    values: tuple[tuple[str, str], ...] = ()


class Step(ContractModel):
    """运行记录中的一步。不含用户原话与回答原文。"""

    agent: AgentId
    stage: Literal["gather"]
    status: Literal["ok", "no_opinion"]
    reason_code: str | None = None
    tools: tuple[str, ...] = ()
    facts: tuple[Fact, ...] = ()
    duration_ms: float


class RunRecord(ContractModel):
    request_id: Annotated[str, Field(min_length=1)]
    intent: Intent
    primary: AgentId
    assistants: tuple[AgentId, ...]
    steps: tuple[Step, ...]
    layer: Literal["l0", "l1", "general", "plan_note"]
    degraded: bool
    card_ids: tuple[str, ...]  # 回答用到的问答卡
    number_rewrites: int  # 数字校验后重写的次数


class AnswerPart(ContractModel):
    kind: Literal[
        "conclusion", "for_you", "why", "how", "caution", "uncertain", "clarify", "notice", "answer"
    ]
    text: Annotated[str, Field(min_length=1)]


class QAResult(ContractModel):
    parts: tuple[AnswerPart, ...]
    record: RunRecord


class _Texts(BaseModel):
    """回答用到的固定说法（与 answer_texts.yaml 对应的子集）。"""

    model_config = ConfigDict(frozen=True, extra="ignore")

    general: str
    degraded: str
    plan_change: str
    caution_discomfort: str
    caution_special: str
    bmi_measured: str
    bmi_stated: str


@dataclass(frozen=True)
class _Ask:
    """交给每个 Agent 的输入：只有问题本身，不含其他 Agent 的输出。"""

    text: str
    intent: Intent
    concerns: tuple[Concern, ...]


def route(result: IntentResult) -> tuple[AgentId, tuple[AgentId, ...]]:
    primary, extra = ROUTES[result.intent]
    assistants = [*extra, _A.SAFETY_OFFICER]
    if "special_population" in result.concerns:
        assistants.append(_A.ASSESSOR)
    unique = [a for a in dict.fromkeys(assistants) if a != primary]
    # 安全官必须保留；超过上限时先去掉排在前面的普通辅助 Agent
    while len(unique) > MAX_ASSISTANTS:
        unique.remove(next(a for a in unique if a not in (_A.SAFETY_OFFICER, _A.ASSESSOR)))
    return primary, tuple(unique)


class _Toolbox:
    def __init__(self, agent: AgentId, cards: CardBook) -> None:
        self._agent = agent
        self._cards = cards
        self.used: list[str] = []

    def call(self, tool: str, ask: _Ask) -> list[tuple[QACard, int]]:
        if tool not in TOOLS.get(self._agent, frozenset()):
            raise ToolDenied(tool)
        self.used.append(tool)
        if tool == "search_cards":
            return [(c, s) for c, s in self._cards.matches(ask.text) if c.agent == self._agent]
        raise ToolDenied(tool)  # pragma: no cover - 白名单里只有已实现的工具


type AgentFn = Callable[[AgentId, _Ask, _Toolbox], tuple[Fact, ...]]


def _domain_agent(agent: AgentId, ask: _Ask, tools: _Toolbox) -> tuple[Fact, ...]:
    return tuple(Fact(kind="card", source_id=c.id) for c, _ in tools.call("search_cards", ask)[:2])


_HEIGHT_RE = re.compile(r"身高(1\d{2}(?:\.\d)?)|(1\d{2}(?:\.\d)?)(?:cm|厘米|公分)")
_WEIGHT_RE = re.compile(r"(\d{2,3}(?:\.\d)?)(公斤|kg|千克|斤)")
_BMI_RE = re.compile(r"bmi(?:是|为|只有)?(\d{1,2}(?:\.\d)?)")


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _bmi_fact(text: str) -> Fact | None:
    """用户在这句话里说的身高体重（算出 BMI）或 BMI。"""
    joined = "".join(clauses(text))
    height, weight = _HEIGHT_RE.search(joined), _WEIGHT_RE.search(joined)
    if height and weight:
        cm = float(height.group(1) or height.group(2))
        kg = float(weight.group(1)) / (2 if weight.group(2) == "斤" else 1)
        if 120 <= cm <= 230 and 25 <= kg <= 300:
            values = (("height", _fmt(cm)), ("weight", _fmt(kg)), ("bmi", _fmt(bmi(cm, kg))))
            return Fact(kind="personal", source_id="bmi_measured", values=values)
    stated = _BMI_RE.search(joined)
    if stated and 10 <= float(stated.group(1)) <= 60:
        return Fact(kind="personal", source_id="bmi_stated", values=(("bmi", stated.group(1)),))
    return None


def _safety_officer(agent: AgentId, ask: _Ask, tools: _Toolbox) -> tuple[Fact, ...]:
    out = [Fact(kind="card", source_id=c.id) for c, _ in tools.call("search_cards", ask)[:2]]
    personal = _bmi_fact(ask.text)
    if personal is not None:
        out.append(personal)
    if "discomfort" in ask.concerns:
        out.append(Fact(kind="caution", source_id="caution_discomfort"))
    if "special_population" in ask.concerns:
        out.append(Fact(kind="caution", source_id="caution_special"))
    return tuple(out)


def _assessor(agent: AgentId, ask: _Ask, tools: _Toolbox) -> tuple[Fact, ...]:
    return ()  # 建档追问在计划链路中；问答里只占位，保证特殊人群时有评估师参与的记录


AGENTS: dict[AgentId, AgentFn] = {
    **{a: _domain_agent for a, _ in ROUTES.values()},
    _A.RECOVERY: _domain_agent,
    _A.SAFETY_OFFICER: _safety_officer,
    _A.ASSESSOR: _assessor,
}


def _run_agent(
    agent: AgentId, ask: _Ask, cards: CardBook, agents: Mapping[AgentId, AgentFn]
) -> tuple[Step, tuple[Fact, ...]]:
    """(运行记录中的一步, 本次组织回答用的依据)。"""
    start = time.perf_counter()
    tools = _Toolbox(agent, cards)
    try:
        facts = agents[agent](agent, ask, tools)
        status, reason = "ok", None
    except ToolDenied:
        facts, status, reason = (), "no_opinion", "tool_denied"
    except Exception as e:  # 单个 Agent 出错不影响其他
        facts, status, reason = (), "no_opinion", f"agent_error:{type(e).__name__}"
    step = Step(
        agent=agent,
        stage="gather",
        status=status,  # type: ignore[arg-type]
        reason_code=reason,
        tools=tuple(tools.used),
        facts=tuple(Fact(kind=f.kind, source_id=f.source_id) for f in facts),
        duration_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return step, facts


class _L1Answer(BaseModel):
    """L1 模型输出：统一结构的各部分，没有的给 null。"""

    model_config = ConfigDict(extra="forbid")

    conclusion: Annotated[str, Field(min_length=1)]
    why: str | None = None
    how: str | None = None
    caution: str | None = None
    uncertain: str | None = None


def _card_parts(card: QACard) -> list[AnswerPart]:
    return [AnswerPart(kind=k, text=v) for k, v in card.parts()]


def _l1(
    ask: _Ask,
    cards: list[QACard],
    cautions: list[str],
    gateway: Gateway,
    request_id: str,
    extra_numbers: set[str],
) -> tuple[list[AnswerPart] | None, int]:
    """用模型表达；返回 (各部分, 数字重写次数)。数字两次都不合规时返回 None。"""
    knowledge = [
        {"id": c.id, **{k: v for k, v in c.parts()}, "sources": list(c.sources)} for c in cards
    ]
    registered = set().union(*(numbers_in(" ".join(v for _, v in c.parts())) for c in cards))
    registered |= extra_numbers
    payload = {"question": ask.text, "knowledge": knowledge, "cautions": cautions}
    system = PROMPT_PATH.read_text(encoding="utf-8")
    rewrites = 0
    for attempt in range(2):
        extra = (
            ""
            if attempt == 0
            else f"\n上一次的回答用了没有来源的数字。只能使用这些数字：{sorted(registered)}。"
        )
        result = gateway.complete(
            ModelStage.ANSWER,
            request_id=request_id,
            template=ANSWER_TEMPLATE,
            messages=[
                Message("system", system + extra),
                Message("user", json.dumps(payload, ensure_ascii=False)),
            ],
            validate=json_model(_L1Answer),
        )
        if result.value is None:
            return None, rewrites
        answer = result.value
        values: list[tuple[PartKind, str | None]] = [
            ("conclusion", answer.conclusion),
            ("why", answer.why),
            ("how", answer.how),
            ("caution", answer.caution),
            ("uncertain", answer.uncertain),
        ]
        text = " ".join(v for _, v in values if v)
        if not unregistered(text, registered):
            return [AnswerPart(kind=k, text=v) for k, v in values if v], rewrites
        rewrites += 1
    return None, rewrites


def answer_question(
    text: str,
    intent: IntentResult,
    *,
    cards: CardBook,
    texts: BaseModel,
    gateway: Gateway | None,
    request_id: str,
    agents: Mapping[AgentId, AgentFn] = AGENTS,
) -> QAResult:
    fixed = _Texts.model_validate(texts.model_dump())
    if intent.intent not in ROUTES:
        raise ValueError(f"意图 {intent.intent} 不经过问答编排（由对话入口处理）")
    primary, assistants = route(intent)
    ask = _Ask(text=text, intent=intent.intent, concerns=intent.concerns)
    order = (primary, *assistants)
    # 同一阶段并行；每个 Agent 只拿到 ask，彼此看不到输出，全部结束后再汇总
    with ThreadPoolExecutor(max_workers=len(order)) as pool:
        results = list(pool.map(lambda a: _run_agent(a, ask, cards, agents), order))
    steps = [step for step, _ in results]
    facts = [f for _, fs in results for f in fs]
    card_ids = list(dict.fromkeys(f.source_id for f in facts if f.kind == "card"))
    cautions = [getattr(fixed, f.source_id) for f in facts if f.kind == "caution"]
    caution_parts = [AnswerPart(kind="caution", text=c) for c in cautions]
    # 针对你：用户自己说的数字算出的结果（数字视为登记过）
    for_you = [
        AnswerPart(kind="for_you", text=getattr(fixed, f.source_id).format(**dict(f.values)))
        for f in facts
        if f.kind == "personal"
    ]
    # 选卡：安全官的卡优先（安全类问题），其次主 Agent 的卡
    owner = {cid: cards.get(cid).agent for cid in card_ids}
    ranked = sorted(card_ids, key=lambda c: (owner[c] != _A.SAFETY_OFFICER, owner[c] != primary))

    parts: list[AnswerPart]
    degraded, rewrites, used = False, 0, []
    safety_first = bool(ranked) and owner[ranked[0]] == _A.SAFETY_OFFICER
    if intent.intent is Intent.PLAN and not safety_first:
        layer = "plan_note"
        parts = [AnswerPart(kind="answer", text=fixed.plan_change), *caution_parts]
    elif not card_ids:
        layer, degraded = "general", True
        parts = [AnswerPart(kind="answer", text=fixed.general), *caution_parts]
    else:
        best = cards.get(ranked[0])
        # 安全官的卡就是为这类情况审定的：直接用，安全提示附在后面
        direct = not cautions or safety_first
        l1_parts: list[AnswerPart] | None = None
        if not direct and gateway is not None:
            chosen = [cards.get(c) for c in ranked]
            extra = {n for p in for_you for n in numbers_in(p.text)}
            l1_parts, rewrites = _l1(ask, chosen, cautions, gateway, request_id, extra)
        if l1_parts is not None:
            layer, parts, used = "l1", [*l1_parts[:1], *for_you, *l1_parts[1:]], ranked
        else:
            # L0：直接用最具体的卡；本应走 L1 却失败时要告诉用户
            degraded = not direct
            layer, used = "l0", [best.id]
            notice = [AnswerPart(kind="notice", text=fixed.degraded)] if degraded else []
            card = _card_parts(best)
            parts = [*notice, *card[:1], *for_you, *card[1:], *caution_parts]

    record = RunRecord(
        request_id=request_id,
        intent=intent.intent,
        primary=primary,
        assistants=assistants,
        steps=tuple(steps),
        layer=layer,  # type: ignore[arg-type]
        degraded=degraded,
        card_ids=tuple(used),
        number_rewrites=rewrites,
    )
    _log.info("qa.answered", request_id=request_id, status=layer, count=len(parts))
    return QAResult(parts=tuple(parts), record=record)
