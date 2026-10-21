"""检索前的多查询改写（spec [2026-10-21] §13）。

把用户这句话（结合最近几轮与页面条目）改写为一条主查询和几条不同说法；一句里问了几件事时拆开。
- 改写只能补全、拆分、换说法：查询里的数字必须是用户说过的；用户提到的身体不适、特殊人群
  必须留在主查询里，也不能凭空加上；用户这句话命中的安全类问答卡，主查询也必须命中，且改写不能
  命中用户没说到的安全类卡。不符合时改写作废，只用原话。
- 原话总在查询里（放在最后），保证关键词一路至少能按原话召回。
- 模型不可用时只用原话。急症判断不经过这里，只看原话。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.model_call import ModelStage
from ironm.contracts.roles import AgentId
from ironm.core.numbers import numbers_in
from ironm.knowledge.qa_cards import CardBook
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.orchestrator.intent import IntentBook, concerns_in

__all__ = ["Rewritten", "rewrite_queries"]

PROMPT_PATH = Path(__file__).parent / "prompts" / "rewrite.md"
PROMPT_TEMPLATE = "rewrite@1.0.0"

type Status = Literal["model", "original", "rejected"]

_SAFETY = AgentId.SAFETY_OFFICER


@dataclass(frozen=True)
class Rewritten:
    queries: tuple[str, ...]  # 第一条为主查询；原话在最后
    status: Status  # model 改写成功；original 没有调用模型；rejected 改写不合规、只用原话
    reason: str | None = None  # 作废的原因代码：numbers、concerns、safety_cards、model_failed


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main: Annotated[str, Field(min_length=1, max_length=120)]
    variants: list[Annotated[str, Field(min_length=1, max_length=120)]] = []


def _safety_cards(text: str, cards: CardBook) -> set[str]:
    return {c.id for c, _ in cards.matches(text) if c.agent is _SAFETY}


def _changed_meaning(
    text: str, history: tuple[str, ...], out: "_Output", book: IntentBook, cards: CardBook
) -> str | None:
    """改写是否改变了这句话的数字、关注点或安全含义；返回原因代码。"""
    said = " ".join([text, *history])
    written = [out.main, *out.variants]
    if numbers_in(" ".join(written)) - numbers_in(said):
        return "numbers"  # 加了用户没说过的数字
    if set(concerns_in(text, book)) - set(concerns_in(out.main, book)):
        return "concerns"  # 丢了身体不适 / 特殊人群
    if {c for q in written for c in concerns_in(q, book)} - set(concerns_in(said, book)):
        return "concerns"  # 凭空加上了身体状况
    if _safety_cards(text, cards) - _safety_cards(out.main, cards):
        return "safety_cards"  # 丢了安全类问题的说法（如“快速掉秤”）
    if set().union(*(_safety_cards(q, cards) for q in written)) - _safety_cards(said, cards):
        return "safety_cards"  # 改写里冒出用户没说到的安全类问题
    return None


def rewrite_queries(
    text: str,
    *,
    history: tuple[str, ...],
    context_item: str | None,
    book: IntentBook,
    cards: CardBook,
    gateway: Gateway | None,
    request_id: str,
    max_queries: int,
) -> Rewritten:
    if gateway is None or max_queries < 2:
        return Rewritten((text,), "original")
    system = PROMPT_PATH.read_text(encoding="utf-8").replace(
        "{max_variants}", str(max(max_queries - 2, 1))
    )
    payload = {"message": text, "recent": list(history), "page_item": context_item}
    result = gateway.complete(
        ModelStage.QUERY_REWRITE,
        request_id=request_id,
        template=PROMPT_TEMPLATE,
        messages=[
            Message("system", system),
            Message("user", json.dumps(payload, ensure_ascii=False)),
        ],
        validate=json_model(_Output),
    )
    out = result.value
    if out is None:
        return Rewritten((text,), "rejected", "model_failed")
    reason = _changed_meaning(text, history, out, book, cards)
    if reason is not None:
        return Rewritten((text,), "rejected", reason)
    variants = [v for v in out.variants if v.strip()][: max_queries - 2]
    queries = tuple(dict.fromkeys(q.strip() for q in (out.main, *variants, text)))
    return Rewritten(queries[:max_queries], "model")
