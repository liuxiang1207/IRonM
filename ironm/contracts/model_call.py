"""大模型调用记录。只记元数据，不存提示与回答原文。"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ironm.contracts.common import ContractModel, UtcDatetime, VersionedName
from ironm.contracts.unknown import NOT_RECORDED, Unknown

type Text = Annotated[str, Field(min_length=1)]
type Tokens = Annotated[int, Field(ge=0)] | Unknown


class ModelStage(StrEnum):
    INTENT = "intent_l3"
    CAPTURE = "capture"
    ANSWER = "answer"
    REVIEW_SEMANTIC = "review_semantic"
    REFLECT = "reflect"


class ModelCall(ContractModel):
    call_id: Text
    request_id: Text  # 贯穿整条链路的请求编号
    stage: ModelStage
    provider: Text
    model: Text
    prompt_template: VersionedName
    started_at: UtcDatetime
    duration_ms: int = Field(ge=0)
    input_tokens: Tokens = NOT_RECORDED
    output_tokens: Tokens = NOT_RECORDED
    status: Literal["ok", "timeout", "error", "rejected"]  # rejected：输出未通过校验
    fallback_used: bool  # 是否改用模板兜底
    error_type: Text | None = None

    @model_validator(mode="after")
    def _check_error(self) -> Self:
        if self.status == "error" and self.error_type is None:
            raise ValueError("出错的调用必须记录错误类型")
        if self.status == "ok" and self.error_type is not None:
            raise ValueError("成功的调用不应带错误类型")
        return self
