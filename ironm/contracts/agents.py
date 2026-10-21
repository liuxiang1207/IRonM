"""Agent 输出：约束、提案、判定、追问、意见、无意见。越权与输出类型在契约层校验。

意见（opinion）是问答链路的输出，与提案共用一个原则：结论必须能追溯到依据。
- 每个数字必须出现在 calculation 或 facts 里；
- 每条结论必须有规则或知识来源；
- reasoning 只描述“依据怎么得到结论”，不引入新的断言。
"""

import json
from collections.abc import Mapping
from datetime import date
from typing import Annotated, Any, Literal, Self

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


class Citation(ContractModel):
    """一条可追溯的依据。用户在回答上点“为什么”时逐条展开。"""

    kind: Literal["rule", "knowledge", "record", "calculation"]
    ref: Text  # 规则编号、知识条目编号、记录编号，或计算结果的标识
    label: Text  # 给用户看的一句话，如“体重 70 公斤（3 月 12 日记录）”
    detail: Text | None = None  # 更深一层：规则摘要、记录时间与来源、公式与代入值


class Opinion(ContractModel):
    """一个 Agent 对本次提问的专业意见（问答链路）。"""

    conclusion: Text
    citations: tuple[Citation, ...] = Field(min_length=1)  # 结论的依据，至少一条
    reasoning: Text  # 依据怎么得到结论（模板生成，不自由发挥）
    how: Text | None = None
    caution: Text | None = None
    uncertain: Text | None = None

    @model_validator(mode="after")
    def _check_numbers(self) -> Self:
        # 意见每个部分里的数字都必须在依据里出现，否则这条意见作废（数字校验前置到意见层）
        # 只有依据算来源：规则/知识条目/计算结果的原文本身就可能含数字（如“45 秒”）；
        # 推理是意见自己写的，不能给自己登记数字（spec [2026-10-21] §12 独立审查 C1）
        registered: set[str] = set()
        for c in self.citations:
            registered |= _numbers(" ".join(v for v in (c.label, c.detail) if v))
        parts = (self.conclusion, self.reasoning, self.how, self.caution, self.uncertain)
        used = _numbers(" ".join(v for v in parts if v))
        missing = used - registered
        if missing:
            raise ValueError(f"意见里出现没有依据的数字：{sorted(missing)}")
        return self


def make_opinion(context: Any, opinion: Opinion) -> "AgentOutput":
    """Skill 产出意见的公共写法：自动填上所属 Agent 与 Skill 版本。"""
    m = context.skill.manifest
    data = {
        "agent": m.owner_agent.value,
        "skill": m.versioned,
        "kind": "opinion",
        "items": [opinion.model_dump()],
    }
    return AGENT_OUTPUT_ADAPTER.validate_json(json.dumps(data, ensure_ascii=False, default=str))


def _numbers(text: str) -> set[str]:
    """意见层的数字提取：与回答校验同一份实现，避免两处规则漂移。"""
    from ironm.core.numbers import numbers_in

    return numbers_in(text)


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


class OpinionOutput(_OutputBase):
    kind: Literal[OutputKind.OPINION]
    items: tuple[Opinion, ...] = Field(min_length=1)


class NoOpinionOutput(_OutputBase):
    kind: Literal[OutputKind.NO_OPINION]
    reason_code: Text | None = None


type AgentOutput = Annotated[
    ConstraintOutput
    | ProposalOutput
    | VerdictOutput
    | QuestionsOutput
    | OpinionOutput
    | NoOpinionOutput,
    Field(discriminator="kind"),
]

AGENT_OUTPUT_ADAPTER: TypeAdapter[AgentOutput] = TypeAdapter(AgentOutput)
