"""规则表格式：每个 YAML 文件是一个规则集，每条规则带编号、审定状态、来源与参数。"""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from ironm.contracts.common import ContractModel, FrozenMap

type Text = Annotated[str, Field(min_length=1)]
type RuleId = Annotated[str, Field(pattern=r"^R(-[A-Z0-9]+)+$")]
type Scalar = bool | int | float | str
type ParamValue = Scalar | tuple[Scalar, ...]


class SourceKind(StrEnum):
    GUIDELINE = "guideline"  # 【指南】
    DESIGN = "design"  # 【设计】工程决定，内测或模拟校准
    BORROWED = "borrowed"  # 【借鉴】竞品或开源项目
    ESTIMATE = "estimate"  # 【估】


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"  # 【待审定】
    APPROVED = "approved"


class RuleParamError(ValueError):
    pass


class RuleSource(ContractModel):
    kind: SourceKind
    ref: Text | None = None  # 出处；“设计”可不写

    @model_validator(mode="after")
    def _check_ref(self) -> Self:
        if self.kind is not SourceKind.DESIGN and self.ref is None:
            raise ValueError(f"来源为 {self.kind} 时必须写明出处 ref")
        return self


class Rule(ContractModel):
    id: RuleId
    title: Text
    status: ReviewStatus
    source: RuleSource
    params: FrozenMap[str, ParamValue]
    note: Text | None = None

    def params_as[M: BaseModel](self, model: type[M]) -> M:
        """按使用方声明的类型读取参数，不符即报错。"""
        try:
            return model.model_validate(dict(self.params))
        except ValidationError as e:
            raise RuleParamError(f"规则 {self.id} 的参数不符合 {model.__name__}：{e}") from e


class RuleSet(ContractModel):
    rule_set: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    version: Annotated[str, Field(pattern=r"^\d{4}\.\d{1,2}\.\d+$")]
    rules: tuple[Rule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        ids = [r.id for r in self.rules]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"规则编号重复：{', '.join(dup)}")
        return self
