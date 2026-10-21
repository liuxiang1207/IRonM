"""对话入口（spec [2026-10-21] §3、§4）：每条消息先过急症检查与拒答检查，再识别意图。

- 急症或红旗：直接返回审定的四段式回答，不调用模型。说的是本人（或拿不准）时记一条安全标记；
  说的是旁人时照常给急救指引，但不给本人记标记、不暂停本人的计划。
- 命中拒答：先给拒答模板，再接问答链路允许的部分。
- 意图识别：归不进任何领域时追问 1 句；闲聊与超范围用固定说法回答。
- 编排与回答生成（§5–§6）完成前，其余意图的回答层标为 pending。
回答中不含用户原话。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.common import ContractModel
from ironm.contracts.screening import SafetyFlag
from ironm.core.clock import Clock
from ironm.core.ids import IdGenerator
from ironm.core.log import get_logger
from ironm.core.yamlfile import load_yaml
from ironm.llm.gateway import Gateway
from ironm.orchestrator.intent import Concern, Intent, IntentBook, recognize
from ironm.rules.loader import RuleBook
from ironm.safety.decision import SafetyKit
from ironm.safety.refusals import RefusalKind
from ironm.safety.replies import ReplySection, four_section, refusal_sections
from ironm.safety.subject import Subject, subject_of

__all__ = [
    "ANSWER_TEXTS_PATH",
    "MAX_TEXT_CHARS",
    "AnswerPart",
    "AnswerTexts",
    "ChatDeps",
    "ChatReply",
    "load_answer_texts",
    "respond",
]

MAX_TEXT_CHARS = 2000
ANSWER_TEXTS_PATH = Path(__file__).with_name("answer_texts.yaml")

_log = get_logger("ironm.chat")

type Layer = Literal["safety", "clarify", "template", "pending"]


class AnswerTexts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    review: Literal["pending_review", "approved"]
    clarify: Annotated[str, Field(min_length=1)]
    chitchat: Annotated[str, Field(min_length=1)]
    out_of_scope: Annotated[str, Field(min_length=1)]


def load_answer_texts(path: Path = ANSWER_TEXTS_PATH) -> AnswerTexts:
    return AnswerTexts.model_validate(load_yaml(path))


class AnswerPart(ContractModel):
    """问答回答的一部分（§6 的统一结构在回答生成完成后补齐）。"""

    kind: Literal["clarify", "answer"]
    text: Annotated[str, Field(min_length=1)]


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


@dataclass(frozen=True)
class ChatDeps:
    kit: SafetyKit
    clock: Clock
    ids: IdGenerator
    record_flag: Callable[[str, str, SafetyFlag], None]  # (用户, 请求编号, 标记)
    intents: IntentBook
    rulebook: RuleBook
    texts: AnswerTexts
    gateway: Gateway | None = None  # 没有时只用词表识别意图


def respond(user_id: str, text: str, deps: ChatDeps) -> ChatReply:
    start = time.perf_counter()
    request_id = deps.ids.new_id()
    assessment = deps.kit.terms.assess(text)
    refusals = deps.kit.refusals.detect(text)
    subject = subject_of(text)
    refusal_part = refusal_sections(deps.kit.replies, refusals.kinds)
    flagged = False
    intent: Intent | None = None
    concerns: tuple[Concern, ...] = ()
    answer: tuple[AnswerPart, ...] = ()
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
        result = recognize(
            text,
            book=deps.intents,
            rulebook=deps.rulebook,
            gateway=deps.gateway,
            request_id=request_id,
        )
        intent, concerns = result.intent, result.concerns
        if result.clarify:
            layer, answer = "clarify", (AnswerPart(kind="clarify", text=deps.texts.clarify),)
        elif intent is Intent.CHITCHAT:
            layer, answer = "template", (AnswerPart(kind="answer", text=deps.texts.chitchat),)
        elif intent is Intent.OUT_OF_SCOPE:
            layer, answer = "template", (AnswerPart(kind="answer", text=deps.texts.out_of_scope),)
        else:
            layer = "pending"
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
    )
