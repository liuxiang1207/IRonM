"""对话入口（spec [2026-10-21] §3）：每条消息先过急症检查与拒答检查，再交给问答链路。

- 急症或红旗：直接返回审定的四段式回答，不调用模型。说的是本人（或拿不准）时记一条安全标记；
  说的是旁人时照常给急救指引，但不给本人记标记、不暂停本人的计划。
- 命中拒答：先给拒答模板，再接问答链路允许的部分。
- 意图识别与回答生成（§4–§6）完成前，非安全类消息只返回拒答模板（如有），回答层标为 pending。
回答中不含用户原话。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from ironm.contracts.common import ContractModel
from ironm.contracts.screening import SafetyFlag
from ironm.core.clock import Clock
from ironm.core.ids import IdGenerator
from ironm.core.log import get_logger
from ironm.safety.decision import SafetyKit
from ironm.safety.refusals import RefusalKind
from ironm.safety.replies import ReplySection, four_section, refusal_sections
from ironm.safety.subject import Subject, subject_of

__all__ = ["MAX_TEXT_CHARS", "ChatDeps", "ChatReply", "respond"]

MAX_TEXT_CHARS = 2000

_log = get_logger("ironm.chat")


class ChatReply(ContractModel):
    request_id: Annotated[str, Field(min_length=1)]
    layer: Literal["safety", "pending"]  # 问答层；后续加入 L0、L1、追问、兜底
    level: Literal["emergency", "red_flag", "none"]
    subject: Subject
    flagged: bool  # 是否给本人记了安全标记
    refusals: tuple[RefusalKind, ...]
    sections: tuple[ReplySection, ...]


@dataclass(frozen=True)
class ChatDeps:
    kit: SafetyKit
    clock: Clock
    ids: IdGenerator
    record_flag: Callable[[str, str, SafetyFlag], None]  # (用户, 请求编号, 标记)


def respond(user_id: str, text: str, deps: ChatDeps) -> ChatReply:
    start = time.perf_counter()
    request_id = deps.ids.new_id()
    assessment = deps.kit.terms.assess(text)
    refusals = deps.kit.refusals.detect(text)
    subject = subject_of(text)
    refusal_part = refusal_sections(deps.kit.replies, refusals.kinds)
    flagged = False
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
        layer: Literal["safety", "pending"] = "safety"
        sections = four_section(deps.kit.replies, assessment) + refusal_part
    else:
        layer, sections = "pending", refusal_part
    _log.info(
        "chat.replied",
        request_id=request_id,
        status=layer,
        count=len(sections),
        duration_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return ChatReply(
        request_id=request_id,
        layer=layer,
        level=assessment.level,
        subject=subject,
        flagged=flagged,
        refusals=refusals.kinds,
        sections=sections,
    )
