"""对话入口（spec [2026-10-21] §3、§4）：每条消息先过急症检查与拒答检查，再识别意图。

- 急症或红旗：直接返回审定的四段式回答，不调用模型。说的是本人（或拿不准）时记一条安全标记；
  说的是旁人时照常给急救指引，但不给本人记标记、不暂停本人的计划。
- 命中拒答：先给拒答模板，再接问答链路允许的部分。
- 意图识别：归不进任何领域时追问 1 句；闲聊与超范围用固定说法回答。
- 其余意图交给问答编排（qa）：L0 问答卡、L1 模型表达、一般说明，或计划调整说明。
回答中不含用户原话。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.common import ContractModel
from ironm.contracts.memory import Candidate
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import SafetyFlag
from ironm.core.clock import Clock
from ironm.core.ids import IdGenerator
from ironm.core.log import get_logger
from ironm.core.yamlfile import load_yaml
from ironm.knowledge.qa_cards import CardBook
from ironm.llm.gateway import Gateway
from ironm.memory.conversation import compress, context
from ironm.memory.profile import ProfileService
from ironm.memory.repo import MemoryRepo
from ironm.orchestrator.dispatch import page_plan
from ironm.orchestrator.intent import Concern, Intent, IntentBook, recognize
from ironm.orchestrator.pages import PAGE_AGENTS, PAGE_DEFAULT, Page, PageContext
from ironm.orchestrator.qa import AnswerPart, RunRecord, answer_question
from ironm.rules.loader import RuleBook
from ironm.safety.decision import SafetyKit
from ironm.safety.emergency import escalate_during_training
from ironm.safety.refusals import RefusalKind
from ironm.safety.replies import ReplySection, four_section, refusal_sections
from ironm.safety.subject import Subject, subject_of

__all__ = [
    "ANSWER_TEXTS_PATH",
    "MAX_TEXT_CHARS",
    "AnswerPart",
    "AnswerTexts",
    "CandidateView",
    "ChatDeps",
    "ChatReply",
    "after_reply",
    "load_answer_texts",
    "respond",
]

MAX_TEXT_CHARS = 2000
HISTORY_TURNS = 4  # 交给意图识别的最近几句用户消息
ANSWER_TEXTS_PATH = Path(__file__).with_name("answer_texts.yaml")

_log = get_logger("ironm.chat")

type Layer = Literal["safety", "clarify", "template", "l0", "l1", "general", "plan_note", "handoff"]


class AnswerTexts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    review: Literal["pending_review", "approved"]
    clarify: Annotated[str, Field(min_length=1)]
    chitchat: Annotated[str, Field(min_length=1)]
    out_of_scope: Annotated[str, Field(min_length=1)]
    general: Annotated[str, Field(min_length=1)]
    degraded: Annotated[str, Field(min_length=1)]
    plan_change: Annotated[str, Field(min_length=1)]
    caution_discomfort: Annotated[str, Field(min_length=1)]
    caution_special: Annotated[str, Field(min_length=1)]
    profile_immediate: Annotated[str, Field(min_length=1)]
    profile_pending: Annotated[str, Field(min_length=1)]
    bmi_measured: Annotated[str, Field(min_length=1)]
    bmi_stated: Annotated[str, Field(min_length=1)]
    page_handoff: Annotated[str, Field(min_length=1)]


def load_answer_texts(path: Path = ANSWER_TEXTS_PATH) -> AnswerTexts:
    return AnswerTexts.model_validate(load_yaml(path))


class CandidateView(ContractModel):
    """回答里列出的画像候选：用户可以据此确认或否认。"""

    candidate_id: Annotated[str, Field(min_length=1)]
    field: str
    value: str | int
    immediate: bool  # 已经生效（健康状况），等待确认


class ChatReply(ContractModel):
    request_id: Annotated[str, Field(min_length=1)]
    layer: Layer
    level: Literal["emergency", "red_flag", "none"]
    subject: Subject
    flagged: bool  # 是否给本人记了安全标记
    refusals: tuple[RefusalKind, ...]
    intent: Intent | None  # 急症或红旗时不识别意图
    concerns: tuple[Concern, ...]
    sections: tuple[ReplySection, ...]  # 安全与拒答模板
    answer: tuple[AnswerPart, ...]
    degraded: bool = False  # 是否降级（回答里已写明）
    candidates: tuple[CandidateView, ...] = ()
    page: Page | None = None  # 功能页提问时的页面
    handoff: PageContext | None = None  # 转入管家对话时带上的上下文


@dataclass(frozen=True)
class ChatDeps:
    kit: SafetyKit
    clock: Clock
    ids: IdGenerator
    record_flag: Callable[[str, str, SafetyFlag], None]  # (用户, 请求编号, 标记)
    record_run: Callable[[str, RunRecord], None]  # (用户, 运行记录)
    intents: IntentBook
    cards: CardBook
    rulebook: RuleBook
    texts: AnswerTexts
    memory: MemoryRepo
    profiles: ProfileService
    gateway: Gateway | None = None  # 没有时只用词表识别意图


def respond(
    user_id: str,
    text: str,
    deps: ChatDeps,
    *,
    page: Page | None = None,
    page_context: PageContext | None = None,
) -> ChatReply:
    """page：在功能页上就本功能提问；page_context：带入的页面与条目标识（管家对话也可带）。"""
    if page is not None and page_context is not None and page_context.page != page:
        raise ValueError("上下文所在页面与提问的页面不一致")
    start = time.perf_counter()
    request_id = deps.ids.new_id()
    assessment = deps.kit.terms.assess(text)
    if (page or (page_context.page if page_context else None)) == "training":
        assessment = escalate_during_training(assessment)
    refusals = deps.kit.refusals.detect(text)
    subject = subject_of(text)
    refusal_part = refusal_sections(deps.kit.replies, refusals.kinds)
    # 画像：先读（判断特殊人群），再记下这句话里的变化（健康状况立即生效）
    profile: dict[str, Any] = {}
    noted: list[Candidate] = []
    try:
        profile = deps.profiles.profile(user_id)
        noted = deps.profiles.note(user_id, request_id, text)
    except Exception as e:  # 画像出错不影响回答（spec [2026-10-21] §8）
        _log.exception("chat.profile_failed", e, request_id=request_id)
    flagged = False
    intent: Intent | None = None
    concerns: tuple[Concern, ...] = ()
    answer: tuple[AnswerPart, ...] = ()
    degraded = False
    handoff: PageContext | None = None
    layer: Layer
    if assessment.level != "none":
        if subject == "self":
            flag = SafetyFlag(
                at=deps.clock.now(),
                level=assessment.level,
                rule_ids=assessment.rule_ids,
                categories=assessment.categories,
            )
            deps.record_flag(user_id, request_id, flag)
            flagged = True
        layer = "safety"
        sections = four_section(deps.kit.replies, assessment) + refusal_part
    else:
        sections = refusal_part
        ctx = context(deps.memory, user_id)
        history = tuple(m.text for m in ctx.recent if m.role == "user")[-HISTORY_TURNS:]
        result = recognize(
            text,
            book=deps.intents,
            rulebook=deps.rulebook,
            gateway=deps.gateway,
            request_id=request_id,
            history=history,
        )
        if (profile.get("conditions") or noted_conditions(noted)) and (
            "special_population" not in result.concerns
        ):
            # 画像里有健康状况：按特殊人群处理（只升不降）
            result = result.model_copy(
                update={"concerns": (*result.concerns, "special_population")}
            )
        if result.intent is Intent.UNKNOWN and (hits := deps.cards.matches(text)):
            # 意图识别不出来，但审定的问答卡命中：按卡的意图回答，不追问
            card_intent = Intent(hits[0][0].intents[0])
            result = result.model_copy(update={"intent": card_intent, "clarify": False})
        seen_page = page or (page_context.page if page_context else None)
        if seen_page is not None and (result.clarify or result.intent is Intent.UNKNOWN):
            # 带着页面上下文的含糊问题（“这个动作我做得怎么样”）：按所在页面理解，不追问
            result = result.model_copy(update={"intent": PAGE_DEFAULT[seen_page], "clarify": False})
        intent, concerns = result.intent, result.concerns
        templates = (Intent.CHITCHAT, Intent.OUT_OF_SCOPE)
        # 安全类问题（命中安全官的卡）在任何页面都就地回答：安全官本来就参与每个功能页
        safety_card = any(c.agent == AgentId.SAFETY_OFFICER for c, _ in deps.cards.matches(text))
        page_agent = (
            None
            if page is None
            else PAGE_AGENTS[page].get(intent, AgentId.SAFETY_OFFICER if safety_card else None)
        )
        if page is not None and page_agent is None and intent not in templates:
            # 本功能以外的问题：转入管家对话并带上上下文，不调用其他功能的 Agent
            handoff = page_context or PageContext(page=page)
            layer, answer = "handoff", (AnswerPart(kind="notice", text=deps.texts.page_handoff),)
        elif result.clarify or intent is Intent.UNKNOWN:
            layer, answer = "clarify", (AnswerPart(kind="clarify", text=deps.texts.clarify),)
        elif intent is Intent.CHITCHAT:
            layer, answer = "template", (AnswerPart(kind="answer", text=deps.texts.chitchat),)
        elif intent is Intent.OUT_OF_SCOPE:
            layer, answer = "template", (AnswerPart(kind="answer", text=deps.texts.out_of_scope),)
        else:
            qa = answer_question(
                text,
                result,
                cards=deps.cards,
                texts=deps.texts,
                gateway=deps.gateway,
                request_id=request_id,
                rulebook=deps.rulebook,
                forced=None if page_agent is None else page_plan(page_agent),
                page=page,
                context_item=page_context.item if page_context else None,
            )
            deps.record_run(user_id, qa.record)
            layer, answer, degraded = qa.record.layer, qa.parts, qa.record.degraded
    if noted:
        immediate = any(c.immediate for c in noted)
        note = deps.texts.profile_immediate if immediate else deps.texts.profile_pending
        answer = (*answer, AnswerPart(kind="notice", text=note))
    _log.info(
        "chat.replied",
        request_id=request_id,
        status=layer,
        count=len(sections) + len(answer),
        duration_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return ChatReply(
        request_id=request_id,
        layer=layer,
        level=assessment.level,
        subject=subject,
        flagged=flagged,
        refusals=refusals.kinds,
        intent=intent,
        concerns=concerns,
        sections=sections,
        answer=answer,
        degraded=degraded,
        page=page,
        handoff=handoff,
        candidates=tuple(
            CandidateView(
                candidate_id=c.candidate_id, field=c.field, value=c.value, immediate=c.immediate
            )
            for c in noted
        ),
    )


def noted_conditions(noted: list[Candidate]) -> bool:
    return any(c.field == "conditions" for c in noted)


def after_reply(
    user_id: str, text: str, reply: ChatReply, deps: ChatDeps, *, now: Callable[[], datetime]
) -> None:
    """回答之后：保存这一轮对话，需要时压缩。出错只记日志，不影响已经返回的回答。"""
    try:
        at = now()
        intent = None if reply.intent is None else reply.intent.value
        deps.memory.add_message(user_id, reply.request_id, "user", text, intent, at)
        said = [s.text for s in reply.sections] + [p.text for p in reply.answer]
        deps.memory.add_message(
            user_id, reply.request_id, "assistant", "\n".join(said) or "（无）", None, at
        )
        compress(
            deps.memory,
            user_id,
            rulebook=deps.rulebook,
            gateway=deps.gateway,
            request_id=reply.request_id,
            now=now,
        )
    except Exception as e:
        _log.exception("chat.after_reply_failed", e, request_id=reply.request_id)
