"""依据卡（spec §31）：每条建议用了哪些数据、依据什么、怎么算的、何时复核。"""

from datetime import date
from typing import Annotated

from pydantic import Field

from ironm.contracts.common import ContractModel, FrozenJson, VersionedName
from ironm.contracts.plan import PlanField
from ironm.contracts.roles import AgentId

type Text = Annotated[str, Field(min_length=1)]


class EvidenceSource(ContractModel):
    """依据：规则表中的规则，或知识库（如动作库）。"""

    ref: Text  # 规则编号或知识库编号
    title: Text
    kind: Text  # guideline / design / borrowed / estimate / knowledge
    citation: Text | None  # 出处；“设计”可为空
    review: Text  # 审定状态


class EvidenceCard(ContractModel):
    card_id: Text  # 与提案编号相同
    field: PlanField
    from_value: FrozenJson  # null 表示首次设定
    to_value: FrozenJson
    agent: AgentId
    skill: VersionedName
    data_used: tuple[Text, ...]  # Skill 声明的数据
    facts: tuple[Text, ...] = Field(min_length=1)  # 计算过程与事实
    source: EvidenceSource
    expected_effect: Text
    rollback_condition: Text
    review_date: date
