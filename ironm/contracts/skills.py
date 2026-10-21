"""Skill 清单（SKILL.md 头信息，spec §28）。Skill 是某个 Agent 的能力包。"""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ironm.contracts.common import ContractModel
from ironm.contracts.plan import FIELD_OWNERS, PlanField
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Tier

type Text = Annotated[str, Field(min_length=1)]
type SkillId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
# 可声明的数据：快照中的几大块，或具体的档案字段与指标
type DataKey = Annotated[
    str,
    Field(
        pattern=r"^(tier|tier_matched|precision|current_plan|constraints|prior_verdicts"
        r"|profile\.[a-z][a-z0-9_]*|metric\.[a-z][a-z0-9_]*)$"
    ),
]


class ScenePack(StrEnum):
    FAT_LOSS = "fat_loss"  # 减脂（第一期）
    STRENGTH = "strength"  # 增肌 / 力量
    FITNESS = "fitness"  # 体能 / 心肺
    HABITS = "habits"  # 健康习惯
    MAINTENANCE = "maintenance"  # 维持期


class Triggers(ContractModel):
    intents: tuple[Text, ...] = Field(min_length=1)


class AppliesTo(ContractModel):
    scene_packs: tuple[ScenePack, ...] = Field(min_length=1)
    population_tiers: tuple[Tier, ...] = Field(min_length=1)
    unknown_tier: bool = False  # 分层未知时也加载（如建档追问）；处方类不能为真


class SkillManifest(ContractModel):
    id: SkillId
    owner_agent: AgentId  # 只能被这个 Agent 加载
    version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]
    description: Text
    triggers: Triggers
    applies_to: AppliesTo
    requires_data: tuple[DataKey, ...] = ()  # 平台按此裁剪快照
    produces: tuple[PlanField, ...] = ()  # 产出的计划字段；须归所属 Agent 负责
    priority: Annotated[int, Field(ge=0)] = 100  # 同一 Agent 内数字小者优先
    prescriptive: bool  # 处方类：会给出训练或饮食的具体安排
    review: Literal["pending_review", "approved"]
    reviewed_by: Text
    tests: Annotated[str, Field(pattern=r"^tests/[A-Za-z0-9_./-]+\.py$")]  # 该 Skill 自带的测试文件

    @property
    def versioned(self) -> str:
        return f"{self.id}@{self.version}"

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.prescriptive and self.applies_to.unknown_tier:
            raise ValueError(f"{self.id}：处方类 Skill 不能在分层未知时加载")
        if self.produces and not self.prescriptive:
            raise ValueError(f"{self.id}：产出计划字段的 Skill 必须是处方类")
        stolen = [f for f in self.produces if FIELD_OWNERS[f] != self.owner_agent]
        if stolen:
            raise ValueError(f"越权：{self.id} 产出的字段 {stolen} 不归 {self.owner_agent} 负责")
        for name, items in [
            ("requires_data", self.requires_data),
            ("produces", self.produces),
            ("population_tiers", self.applies_to.population_tiers),
        ]:
            if len(set(items)) != len(items):
                raise ValueError(f"{self.id}：{name} 有重复")
        return self
