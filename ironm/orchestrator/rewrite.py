"""检索前的多查询改写（spec [2026-10-21] §13）。

把用户这句话（结合最近几轮与页面条目）改写为一条主查询和几条不同说法；一句里问了几件事时拆开。
- 改写只能补全、拆分、换说法：查询里的数字必须是用户说过的；用户提到的身体不适、特殊人群
  必须留在主查询里。不符合时改写作废，只用原话。
- 原话总在查询里（放在最后），保证关键词一路至少能按原话召回。
- 模型不可用时只用原话。急症判断不经过这里，只看原话。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.model_call import ModelStage
from ironm.core.numbers import numbers_in
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.orchestrator.intent import IntentBook, concerns_in

__all__ = ["Rewritten", "rewrite_queries"]

PROMPT_PATH = Path(__file__).parent / "prompts" / "rewrite.md"
PROMPT_TEMPLATE = "rewrite@1.0.0"

type Status = Literal["model", "original", "rejected"]


@dataclass(frozen=True)
class Rewritten:
    queries: tuple[str, ...]  # 第一条为主查询；原话在最后
    status: Status  # model 改写成功；original 没有调用模型；rejected 改写不合规、只用原话
    reason: str | None = None  # 作废的原因代码：numbers、concerns、model_failed


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main: Annotated[str, Field(min_length=1, max_length=120)]
    variants: list[Annotated[str, Field(min_length=1, max_length=120)]] = []


def rewrite_queries(
    text: str,
    *,
    history: tuple[str, ...],
    context_item: str | None,
    book: IntentBook,
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
    said = numbers_in(" ".join([text, *history]))
    if numbers_in(" ".join([out.main, *out.variants])) - said:
        return Rewritten((text,), "rejected", "numbers")  # 加了用户没说过的数字
    if set(concerns_in(text, book)) - set(concerns_in(out.main, book)):
        return Rewritten((text,), "rejected", "concerns")  # 丢了身体不适 / 特殊人群
    variants = [v for v in out.variants if v.strip()][: max_queries - 2]
    queries = tuple(dict.fromkeys(q.strip() for q in (out.main, *variants, text)))
    return Rewritten(queries[:max_queries], "model")
