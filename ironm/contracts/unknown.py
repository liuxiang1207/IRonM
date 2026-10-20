"""显式的缺失值。缺失必须写成 Unknown 并注明原因，禁止用 0 或空值代替。"""

from enum import StrEnum
from typing import Any, NoReturn, TypeGuard

from pydantic import Field

from ironm.contracts.common import ContractModel


class UnknownReason(StrEnum):
    NOT_RECORDED = "not_recorded"
    DECLINED = "declined"
    UNPARSEABLE = "unparseable"
    INSUFFICIENT_DATA = "insufficient_data"


class Unknown(ContractModel):
    """JSON 形如 {"unknown": "not_recorded"}。不能参与运算，也不能当真假值判断。"""

    unknown: UnknownReason

    def __bool__(self) -> NoReturn:
        raise TypeError("“未知”不能当真假值判断，请用 is_known() 显式检查")


def not_recorded() -> Unknown:
    return Unknown(unknown=UnknownReason.NOT_RECORDED)


# 可选测量项省略时视为“没记录”，仍是显式的 Unknown 而不是 0 或空值
NOT_RECORDED: Any = Field(default_factory=not_recorded)


def is_known[T](value: T | Unknown) -> TypeGuard[T]:
    return not isinstance(value, Unknown)
