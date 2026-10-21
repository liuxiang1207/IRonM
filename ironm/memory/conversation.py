"""对话上下文与压缩（spec [2026-10-21] §8）。

- 上下文：较早对话的摘要 + 还没压缩的消息（原文）。
- 压缩：还没压缩的消息达到 compress_at 条时，把除最近 keep_recent 条以外的部分并入摘要。
  摘要由模型生成（与上一份摘要合并）；模型不可用时按话题汇总，不含原话。
摘要不展示给用户，只作为后续回答的上下文；压缩在回答之后进行，不拖慢回答。
"""

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.memory import StoredMessage
from ironm.contracts.model_call import ModelStage
from ironm.core.log import get_logger
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.memory.repo import MemoryRepo
from ironm.rules.loader import RuleBook

__all__ = ["MEMORY_RULE", "ChatContext", "MemoryParams", "compress", "context"]

MEMORY_RULE = "R-QA-MEMORY"
PROMPT_PATH = Path(__file__).parent / "prompts" / "summary.md"
PROMPT_TEMPLATE = "summary@1.0.0"

# 意图的中文说法（按话题汇总的兜底摘要用）
_TOPICS = {
    "exercise_technique": "动作技术",
    "training_knowledge": "训练知识",
    "plan": "训练计划",
    "nutrition": "饮食",
    "supplements_drugs": "补剂与药物",
    "personal_data": "个人数据",
    "logging": "记录",
    "recovery_sleep": "恢复与睡眠",
    "habit_motivation": "习惯与动力",
    "wellness": "养生保健",
}

_log = get_logger("ironm.memory")


class MemoryParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    compress_at: Annotated[int, Field(ge=2)]
    keep_recent: Annotated[int, Field(ge=1)]

    def model_post_init(self, _: object) -> None:
        if self.keep_recent >= self.compress_at:
            raise ValueError("保留原文的条数必须少于触发压缩的条数")


@dataclass(frozen=True)
class ChatContext:
    summary: str | None
    recent: tuple[StoredMessage, ...]  # 还没压缩的消息，按时间先后


def context(store: MemoryRepo, user_id: str) -> ChatContext:
    summary = store.latest_summary(user_id)
    after = summary.upto_seq if summary else 0
    return ChatContext(
        summary=summary.text if summary else None,
        recent=tuple(store.messages_after(user_id, after)),
    )


class _Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: Annotated[str, Field(min_length=1, max_length=500)]


def _rule_summary(previous: str | None, older: list[StoredMessage]) -> str:
    topics = Counter(_TOPICS[m.intent] for m in older if m.intent in _TOPICS)
    listed = "、".join(t for t, _ in topics.most_common()) or "闲聊"
    line = f"此前的 {len(older)} 条消息聊到：{listed}。"
    return f"{previous}\n{line}" if previous else line


def _model_summary(
    previous: str | None, older: list[StoredMessage], gateway: Gateway, request_id: str
) -> str | None:
    payload = {
        "previous_summary": previous,
        "messages": [{"role": m.role, "text": m.text} for m in older],
    }
    result = gateway.complete(
        ModelStage.REFLECT,
        request_id=request_id,
        template=PROMPT_TEMPLATE,
        messages=[
            Message("system", PROMPT_PATH.read_text(encoding="utf-8")),
            Message("user", json.dumps(payload, ensure_ascii=False)),
        ],
        validate=json_model(_Summary),
    )
    return None if result.value is None else result.value.summary


def compress(
    store: MemoryRepo,
    user_id: str,
    *,
    rulebook: RuleBook,
    gateway: Gateway | None,
    request_id: str,
    now: Callable[[], datetime],
) -> bool:
    """需要时压缩；返回是否压缩了。"""
    p = rulebook.get(MEMORY_RULE).params_as(MemoryParams)
    ctx = context(store, user_id)
    if len(ctx.recent) < p.compress_at:
        return False
    older = list(ctx.recent[: -p.keep_recent])
    text = _model_summary(ctx.summary, older, gateway, request_id) if gateway is not None else None
    source: Literal["model", "rule"] = "model" if text is not None else "rule"
    store.add_summary(
        user_id, older[-1].seq, text or _rule_summary(ctx.summary, older), source, now()
    )
    _log.info("memory.compressed", request_id=request_id, status=source, count=len(older))
    return True
