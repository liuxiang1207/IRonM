"""Agent 输出：约束、提案、判定、追问、无意见。越权与输出类型在契约层校验。"""

from collections.abc import Mapping
from datetime import date
from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from ironm.contracts.common import ContractModel, FrozenJson, VersionedName, json_equal
from ironm.contracts.plan import FIELD_OWNERS, PlanField
from ironm.contracts.roles import AGENT_OUTPUT_KINDS, AgentId, OutputKind

type Text = Annotated[str, Field(min_length=1)]


class Constraint(ContractModel):
    """边界，如“热量不得低于 1200 千卡”“禁止跳跃类动作”。"""

    field: PlanField
    op: Literal["max", "min", "exclude"]
    value: FrozenJson
    rule_id: Text
    reason_code: Text

    @model_validator(mode="after")
    def _check_value(self) -> Self:
        if self.value is None:
            raise ValueError("约束必须给出取值")
        if self.op in ("max", "min") and not _is_limit(self.value):
            raise ValueError("上限与下限的取值只能是数值，或“键 → 数值”")
        return self


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_limit(value: object) -> bool:
    """数值，或非空的“键 → 数值”。"""
    if _is_number(value):
        return True
    if isinstance(value, Mapping):
        return bool(value) and all(_is_number(v) for v in value.values())
    return False


class Proposal(ContractModel):
    proposal_id: Text
    field: PlanField
    from_value: FrozenJson  # null 表示首次设定
    to_value: FrozenJson
    reason_facts: tuple[Text, ...] = Field(min_length=1)
    rule_id: Text
    # 调整三件套：缺一项即拒绝
    expected_effect: Text
    rollback_condition: Text
    review_date: date
    evidence_ids: tuple[Text, ...]

    @model_validator(mode="after")
    def _check_change(self) -> Self:
        if self.to_value is None:
            raise ValueError("提案必须给出新值")
        if json_equal(self.to_value, self.from_value):
            raise ValueError("新值与原值相同，不是提案")
        return self


class Verdict(ContractModel):
    proposal_id: Text
    result: Literal["pass", "reject", "warn"]
    reason_code: Text | None = None

    @model_validator(mode="after")
    def _check_reason(self) -> Self:
        if self.result != "pass" and self.reason_code is None:
            raise ValueError("拒绝或警告必须带理由代码")
        return self


class _OutputBase(ContractModel):
    agent: AgentId
    skill: VersionedName
    kind: OutputKind

    @model_validator(mode="after")
    def _check_kind_allowed(self) -> Self:
        if self.kind not in AGENT_OUTPUT_KINDS[self.agent]:
            raise ValueError(f"{self.agent} 不能产出 {self.kind} 类输出")
        return self


class ConstraintOutput(_OutputBase):
    kind: Literal[OutputKind.CONSTRAINT]
    items: tuple[Constraint, ...] = Field(min_length=1)


class ProposalOutput(_OutputBase):
    kind: Literal[OutputKind.PROPOSAL]
    items: tuple[Proposal, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_ownership(self) -> Self:
        seen: set[PlanField] = set()
        for p in self.items:
            owner = FIELD_OWNERS[p.field]
            if owner != self.agent:
                raise ValueError(f"越权：字段 {p.field} 归 {owner} 负责，{self.agent} 无权修改")
            if p.field in seen:
                raise ValueError(f"同一输出中字段 {p.field} 被提出多次")
            seen.add(p.field)
        return self


class VerdictOutput(_OutputBase):
    kind: Literal[OutputKind.VERDICT]
    items: tuple[Verdict, ...] = Field(min_length=1)


MAX_QUESTIONS = 3  # 每次最多追问几项（spec §29）


class Question(ContractModel):
    """要用户补充的一项信息。"""

    item: Annotated[str, Field(pattern=r"^profile\.[a-z][a-z0-9_]*$")]  # 如 profile.birth_date
    reason_code: Text  # 为什么需要，如 needed_for_tier
    fallback: Text  # 没回答时怎么办，如 no_plan_until_answered、assume_a_prime


class QuestionsOutput(_OutputBase):
    kind: Literal[OutputKind.QUESTIONS]
    items: tuple[Question, ...] = Field(min_length=1, max_length=MAX_QUESTIONS)

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        names = [q.item for q in self.items]
        if len(names) != len(set(names)):
            raise ValueError("同一项信息被追问多次")
        return self


class NoOpinionOutput(_OutputBase):
    kind: Literal[OutputKind.NO_OPINION]
    reason_code: Text | None = None


type AgentOutput = Annotated[
    ConstraintOutput | ProposalOutput | VerdictOutput | QuestionsOutput | NoOpinionOutput,
    Field(discriminator="kind"),
]

AGENT_OUTPUT_ADAPTER: TypeAdapter[AgentOutput] = TypeAdapter(AgentOutput)
