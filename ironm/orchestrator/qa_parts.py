"""回答的组成部分（统一结构：结论 → 针对你 → 为什么 → 怎么做 → 注意 → 不确定）。"""

from typing import Annotated, Literal

from pydantic import Field

from ironm.contracts.common import ContractModel

__all__ = ["AnswerPart"]


class AnswerPart(ContractModel):
    kind: Literal[
        "conclusion",
        "for_you",
        "why",
        "how",
        "caution",
        "uncertain",
        "clarify",
        "notice",
        "answer",
    ]
    text: Annotated[str, Field(min_length=1)]
