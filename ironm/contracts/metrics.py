"""指标结果契约：数值总是和精度、可信范围、状态、“还差什么”一起出现。"""

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from ironm.contracts.common import ContractModel, Domain, Precision, UtcDatetime
from ironm.contracts.unknown import Unknown


class MetricStatus(StrEnum):
    OK = "ok"
    INSUFFICIENT = "insufficient"  # 数据不足，不下结论
    STALE = "stale"  # 已过期
    PENDING_CONFIRM = "pending_confirm"  # 有待用户确认的异常值


class ConfidenceInterval(ContractModel):
    low: float
    high: float
    level: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.low > self.high:
            raise ValueError("可信范围下限大于上限")
        return self


class Need(ContractModel):
    """还差什么：如再称重 2 次。item 为稳定编码，由界面翻译成文字。"""

    item: str = Field(min_length=1)
    count: int = Field(ge=1)


class MetricResult(ContractModel):
    name: str = Field(min_length=1)
    domain: Domain
    value: float | Unknown
    unit: str = Field(min_length=1)
    ci: ConfidenceInterval | None = None
    precision: Precision
    n_records: int = Field(ge=0)
    window_days: int = Field(ge=0)
    status: MetricStatus
    needed: tuple[Need, ...] = ()
    last_record_at: UtcDatetime | None
    computed_at: UtcDatetime
    formula_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        unknown = isinstance(self.value, Unknown)
        if self.status is MetricStatus.OK and unknown:
            raise ValueError("状态为 ok 时必须有数值")
        if self.ci is not None:
            if unknown:
                raise ValueError("数值未知时不得有可信范围")
            if not self.ci.low <= self.value <= self.ci.high:  # type: ignore[operator]
                raise ValueError("可信范围必须包含数值")
        if self.status is MetricStatus.INSUFFICIENT and not self.needed:
            raise ValueError("数据不足时必须说明还差什么")
        return self
