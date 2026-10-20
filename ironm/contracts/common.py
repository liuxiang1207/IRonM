"""契约公共部分：基类、时间类型、精度等级与领域。"""

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    WrapSerializer,
)

from ironm.core.clock import ensure_utc


class ContractModel(BaseModel):
    """契约基类：不可变、拒绝未定义字段、不做类型宽松转换、拒绝 NaN 与无穷。"""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)


# 带时区的时间，统一换算为 UTC；不带时区直接拒绝
type UtcDatetime = Annotated[datetime, AfterValidator(ensure_utc)]


def deep_freeze(value: object) -> object:
    """递归转为只读：映射 → 只读映射，列表 → 元组。"""
    if isinstance(value, Mapping):
        return MappingProxyType({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(deep_freeze(v) for v in value)
    return value


def deep_thaw(value: object) -> object:
    """deep_freeze 的逆过程，用于序列化。"""
    if isinstance(value, Mapping):
        return {k: deep_thaw(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [deep_thaw(v) for v in value]
    return value


def json_equal(a: object, b: object) -> bool:
    """JSON 值比较；与 Python 的 == 不同，数值与布尔不视为相等（1 != True）。"""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _freeze_map[K, V](mapping: Mapping[K, V]) -> Mapping[K, V]:
    return cast(Mapping[K, V], deep_freeze(mapping))


def _thaw(value: object, handler: SerializerFunctionWrapHandler) -> object:
    return handler(deep_thaw(value))


# 只读映射：模型本身 frozen 只防重新赋值，映射及其嵌套内容也要防改
type FrozenMap[K, V] = Annotated[
    Mapping[K, V], AfterValidator(_freeze_map), WrapSerializer(_thaw, when_used="always")
]

# 只读的任意 JSON 值（嵌套的列表与对象同样不可改）
type FrozenJson = Annotated[
    JsonValue, AfterValidator(deep_freeze), WrapSerializer(_thaw, when_used="always")
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
