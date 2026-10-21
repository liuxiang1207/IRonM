"""通用 Agent（spec [2026-10-21] §12）：基本通用问题直接回答，不检索。

适用：打招呼、闲聊、App 怎么用、鼓励与陪伴。回答里不能出现具体数字（训练量、热量、剂量等）：
出现数字说明这个问题本该走专业流程，这次改用固定说法。模型不可用时也用固定说法。
"""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.model_call import ModelStage
from ironm.core.numbers import numbers_in
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model

__all__ = ["general_answer"]

PROMPT_PATH = Path(__file__).parent / "prompts" / "general.md"
PROMPT_TEMPLATE = "general@1.0.0"


class _General(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Annotated[str, Field(min_length=1, max_length=300)]


def general_answer(
    text: str,
    kind: Literal["chitchat", "app_help"],
    *,
    gateway: Gateway | None,
    request_id: str,
    fallback: str,
) -> tuple[str, bool]:
    """(回答, 是否由模型生成)。"""
    if gateway is None:
        return fallback, False
    result = gateway.complete(
        ModelStage.ANSWER,
        request_id=request_id,
        template=PROMPT_TEMPLATE,
        messages=[
            Message("system", PROMPT_PATH.read_text(encoding="utf-8")),
            Message("user", json.dumps({"kind": kind, "message": text}, ensure_ascii=False)),
        ],
        validate=json_model(_General),
    )
    if result.value is None or numbers_in(result.value.text):
        return fallback, False
    return result.value.text, True
