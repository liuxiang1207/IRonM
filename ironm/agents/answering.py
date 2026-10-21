"""专业 Agent 的问答（spec [2026-10-21] §12）：检索本领域知识 → 调用本领域工具 → 生成带依据的意见。

- 每个 Agent 一份配置：职责、能读的画像字段、能用的工具。问答不加载 Skill。
- 生成：模型只能基于检索到的段落与工具结果作答，并标出用到了哪几条；数字必须出现在依据里
  （由意见契约校验）。没有任何依据时不作答。
- 兜底（模型不可用、输出不合规、意见作废）：有工具给出的确定性内容时按它作答，否则用本领域
  最相关的问答卡；都没有时为缺失。
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ironm.agents.knowledge import card_opinion
from ironm.agents.tools import ToolInput, ToolResult, run_tools
from ironm.contracts.agents import Citation, Opinion
from ironm.contracts.model_call import ModelStage
from ironm.contracts.roles import AgentId
from ironm.knowledge.kb import Retrieved, Retriever
from ironm.knowledge.qa_cards import CardBook
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.rules.loader import RuleBook

__all__ = ["SPECS", "AgentAnswer", "AgentSpec", "answer_as"]

_A = AgentId
PROMPT_PATH = Path(__file__).parent / "prompts" / "agent_answer.md"
PROMPT_TEMPLATE = "agent_answer@1.0.0"


@dataclass(frozen=True)
class AgentSpec:
    agent: AgentId
    role: str  # 职责：负责回答哪类问题
    data: tuple[str, ...]  # 能读的画像字段
    tools: tuple[str, ...]  # 能用的工具


SPECS: dict[AgentId, AgentSpec] = {
    _A.EXERCISE_COACH: AgentSpec(
        _A.EXERCISE_COACH,
        "动作教练：动作技术、姿势、常见错误、替代动作与降阶进阶",
        ("restrictions", "environment", "training_level"),
        ("exercise_lookup",),
    ),
    _A.TRAINING_PLANNER: AgentSpec(
        _A.TRAINING_PLANNER,
        "训练规划师：训练频率、组数次数、强度、有氧与力量的安排",
        ("training_level", "available_days", "environment", "current_cardio_minutes", "goal"),
        (),
    ),
    _A.NUTRITIONIST: AgentSpec(
        _A.NUTRITIONIST,
        "营养师：饮食、热量、蛋白质、外卖与食堂、补剂",
        (
            "sex",
            "birth_date",
            "height_cm",
            "weight_kg",
            "waist_cm",
            "activity",
            "high_protein",
            "conditions",
            "goal",
        ),
        ("bmi", "calorie_target"),
    ),
    _A.RECOVERY: AgentSpec(_A.RECOVERY, "恢复师：睡眠、酸痛、疲劳与休息安排", (), ()),
    _A.DATA_ANALYST: AgentSpec(
        _A.DATA_ANALYST,
        "数据分析师：体重、体脂、趋势与进展的解读",
        ("height_cm", "weight_kg", "goal"),
        ("bmi",),
    ),
    _A.BEHAVIOR_COACH: AgentSpec(
        _A.BEHAVIOR_COACH, "行为教练：坚持、动力、时间安排与习惯", ("available_days",), ()
    ),
}

type Status = Literal["model", "tool", "card", "missing"]


@dataclass(frozen=True)
class AgentAnswer:
    """一个 Agent 这次的结果。依据编号写进运行记录，不含原话。"""

    agent: AgentId
    status: Status  # model 模型生成；tool / card 兜底；missing 没有依据
    opinion: Opinion | None
    passages: tuple[str, ...]  # 检索到的段落编号
    tools: tuple[str, ...]  # 有结果的工具
    reason: str | None = None  # 缺失或兜底的原因


class _Generated(BaseModel):
    """模型输出：意见各部分与用到的依据编号。"""

    model_config = ConfigDict(extra="forbid")

    conclusion: Annotated[str, Field(min_length=1)]
    reasoning: Annotated[str, Field(min_length=1)]
    how: str | None = None
    caution: str | None = None
    uncertain: str | None = None
    cites: list[str] = Field(min_length=1)


def _citations(
    cites: list[str], passages: list[Retrieved], tools: list[ToolResult]
) -> tuple[Citation, ...]:
    """把模型标出的编号换成依据；编号不在给它的依据里即无效。"""
    by_passage = {r.passage.passage_id: r.passage for r in passages}
    by_tool = {c.ref: c for t in tools for c in t.citations}
    out: list[Citation] = []
    for ref in dict.fromkeys(cites):
        if ref in by_passage:
            p = by_passage[ref]
            out.append(
                Citation(kind="knowledge", ref=ref, label=f"{p.title}（{p.source}）", detail=p.text)
            )
        elif ref in by_tool:
            out.append(by_tool[ref])
        else:
            raise ValueError(f"引用了不存在的依据：{ref}")
    return tuple(out)


def _generate(
    spec: AgentSpec,
    text: str,
    profile: Mapping[str, Any],
    passages: list[Retrieved],
    tools: list[ToolResult],
    gateway: Gateway,
    request_id: str,
) -> Opinion | None:
    payload = {
        "role": spec.role,
        "question": text,
        "profile": dict(profile),
        "passages": [
            {"id": r.passage.passage_id, "title": r.passage.title, "text": r.passage.text,
             "source": r.passage.source, "level": r.passage.level}
            for r in passages
        ],
        "facts": [
            {"id": c.ref, "label": c.label, "detail": c.detail} for t in tools for c in t.citations
        ],
    }  # fmt: skip
    result = gateway.complete(
        ModelStage.ANSWER,
        request_id=request_id,
        template=PROMPT_TEMPLATE,
        messages=[
            Message("system", PROMPT_PATH.read_text(encoding="utf-8")),
            Message("user", json.dumps(payload, ensure_ascii=False)),
        ],
        validate=json_model(_Generated),
    )
    g = result.value
    if g is None:
        return None
    try:
        return Opinion(
            conclusion=g.conclusion,
            citations=_citations(g.cites, passages, tools),
            reasoning=g.reasoning,
            how=g.how,
            caution=g.caution,
            uncertain=g.uncertain,
        )
    except (ValidationError, ValueError):
        return None  # 数字没有依据、引用了不存在的依据：这条意见作废


def _from_tool(tools: list[ToolResult]) -> Opinion | None:
    """工具给出了确定性内容（如动作库查询）时，直接按它作答。"""
    for t in tools:
        if t.conclusion:
            return Opinion(
                conclusion=t.conclusion,
                citations=t.citations,
                reasoning=f"按{t.citations[0].label}给出要点与常见错误。",
                how="；".join(t.how) or None,
                caution="；".join(t.caution) or None,
            )
    return None


def answer_as(
    agent: AgentId,
    *,
    text: str,
    profile: Mapping[str, Any],
    context_item: str | None,
    retriever: Retriever,
    cards: CardBook,
    rulebook: RuleBook,
    gateway: Gateway | None,
    request_id: str,
    as_of: datetime,
    tz: ZoneInfo,
) -> AgentAnswer:
    spec = SPECS[agent]
    visible = {k: profile[k] for k in spec.data if k in profile}  # 只给声明能读的字段
    # 问答卡的触发词命中时，卡作为人工精选的条目优先召回
    pinned = [f"card:{c.id}" for c, _ in cards.matches(text) if c.agent == agent]
    passages = retriever.search(text, agent, pinned=pinned)
    tools = run_tools(
        spec.tools,
        ToolInput(
            text=text,
            profile=visible,
            context_item=context_item,
            rulebook=rulebook,
            as_of=as_of,
            tz=tz,
        ),
    )
    ids = tuple(r.passage.passage_id for r in passages)
    names = tuple(t.name for t in tools)
    if not passages and not tools:
        return AgentAnswer(agent, "missing", None, ids, names, "no_basis")
    reason = "no_model"
    if gateway is not None:
        opinion = _generate(spec, text, visible, passages, tools, gateway, request_id)
        if opinion is not None:
            return AgentAnswer(agent, "model", opinion, ids, names)
        reason = "model_failed_or_invalid"
    tool_opinion = _from_tool(tools)
    if tool_opinion is not None:
        return AgentAnswer(agent, "tool", tool_opinion, ids, names, reason)
    card_ids = [pid.removeprefix("card:") for pid in ids if pid.startswith("card:")]
    card = card_opinion(agent, [cards.get(card_ids[0])]) if card_ids else None
    if card is not None:
        return AgentAnswer(agent, "card", card, ids, names, reason)
    return AgentAnswer(agent, "missing", None, ids, names, reason)
