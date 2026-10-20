"""契约公共部分：基类、时间类型、精度等级与领域。"""

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    WrapSerializer,
)

from ironm.core.clock import ensure_utc


class ContractModel(BaseModel):
    """契约基类：不可变、拒绝未定义字段、不做类型宽松转换、拒绝 NaN 与无穷。"""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


# 带时区的时间，统一换算为 UTC；不带时区直接拒绝
type UtcDatetime = Annotated[datetime, AfterValidator(ensure_utc)]


def _freeze[K, V](mapping: Mapping[K, V]) -> Mapping[K, V]:
    return MappingProxyType(dict(mapping))


# 只读映射：模型本身 frozen 只防重新赋值，映射内容也要防改
def _thaw(mapping: Mapping[object, object], handler: SerializerFunctionWrapHandler) -> object:
    return handler(dict(mapping))


type FrozenMap[K, V] = Annotated[
    Mapping[K, V], AfterValidator(_freeze), WrapSerializer(_thaw, when_used="always")
]

# 带版本的名称，如 diet_review@1.0.0
type VersionedName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*@\d+\.\d+\.\d+$")]


class Precision(StrEnum):
    """数据精度等级：P3 精确、P2 结构化、P1 粗略、P0 感受。"""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class Domain(StrEnum):
    TRAINING = "training"
    BODY = "body"
    DIET = "diet"
    RECOVERY = "recovery"
