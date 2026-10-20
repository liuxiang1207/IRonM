"""计划字段、字段唯一归属与计划版本。

每个计划字段只归一个提案型 Agent，两个 Agent 不可能改同一字段，冲突从结构上消失。
字段取值的具体结构随计划生成定义，这里先用 JSON 值。
"""

from enum import StrEnum
from typing import Self

from pydantic import Field, JsonValue, model_validator

from ironm.contracts.common import ContractModel, FrozenMap, UtcDatetime
from ironm.contracts.roles import AgentId


class PlanField(StrEnum):
    TRAINING_DAYS_PER_WEEK = "training_days_per_week"
    TRAINING_SCHEDULE = "training_schedule"  # 每周哪几天练
    SESSION_SLOTS = "session_slots"  # 训练槽位，如“下肢推 3 组 8–12 次”
    INTENSITY_TARGET = "intensity_target"  # RPE / RIR 范围
    PROGRESSION_RULE = "progression_rule"  # 双进阶与每次增量上限
    DELOAD_SCHEDULE = "deload_schedule"  # 减量周安排
    WEEKLY_CARDIO_MINUTES = "weekly_cardio_minutes"
    DAILY_STEPS_TARGET = "daily_steps_target"
    EXERCISE_SELECTION = "exercise_selection"  # 槽位落成的具体动作、要点与替代
    MINIMUM_TASK = "minimum_task"
    ALTERNATIVE_VERSION = "alternative_version"  # 居家 / 酒店替代版本
    DIET_MODE = "diet_mode"  # 精确 / 简化 / 习惯
    DAILY_KCAL_TARGET = "daily_kcal_target"
    DAILY_PROTEIN_G_TARGET = "daily_protein_g_target"
    PORTION_GUIDE = "portion_guide"  # 简化模式的餐盘与手掌份量
    EATING_OUT_STRATEGY = "eating_out_strategy"  # 外卖、食堂、聚餐
    WEEKLY_HABIT_STEP = "weekly_habit_step"
    REMINDER_SCHEDULE = "reminder_schedule"
    WEIGH_IN_DAYS = "weigh_in_days"  # 固定称重日
    CHECKIN_DAY = "checkin_day"  # 每周签到日


_F = PlanField
FIELD_OWNERS: dict[PlanField, AgentId] = {
    _F.TRAINING_DAYS_PER_WEEK: AgentId.TRAINING_PLANNER,
    _F.TRAINING_SCHEDULE: AgentId.TRAINING_PLANNER,
    _F.SESSION_SLOTS: AgentId.TRAINING_PLANNER,
    _F.INTENSITY_TARGET: AgentId.TRAINING_PLANNER,
    _F.PROGRESSION_RULE: AgentId.TRAINING_PLANNER,
    _F.DELOAD_SCHEDULE: AgentId.TRAINING_PLANNER,
    _F.WEEKLY_CARDIO_MINUTES: AgentId.TRAINING_PLANNER,
    _F.DAILY_STEPS_TARGET: AgentId.TRAINING_PLANNER,
    _F.EXERCISE_SELECTION: AgentId.EXERCISE_COACH,
    _F.MINIMUM_TASK: AgentId.EXERCISE_COACH,
    _F.ALTERNATIVE_VERSION: AgentId.EXERCISE_COACH,
    _F.DIET_MODE: AgentId.NUTRITIONIST,
    _F.DAILY_KCAL_TARGET: AgentId.NUTRITIONIST,
    _F.DAILY_PROTEIN_G_TARGET: AgentId.NUTRITIONIST,
    _F.PORTION_GUIDE: AgentId.NUTRITIONIST,
    _F.EATING_OUT_STRATEGY: AgentId.NUTRITIONIST,
    _F.WEEKLY_HABIT_STEP: AgentId.BEHAVIOR_COACH,
    _F.REMINDER_SCHEDULE: AgentId.BEHAVIOR_COACH,
    _F.WEIGH_IN_DAYS: AgentId.BEHAVIOR_COACH,
    _F.CHECKIN_DAY: AgentId.BEHAVIOR_COACH,
}


class FieldChange(ContractModel):
    field: PlanField
    from_value: JsonValue  # null 表示首次设定
    to_value: JsonValue


class PlanVersion(ContractModel):
    """计划版本只增不改；回退也是生成一个新版本。"""

    version: int = Field(ge=1)
    parent: int | None
    fields: FrozenMap[PlanField, JsonValue]
    diff: tuple[FieldChange, ...]
    rule_set_versions: FrozenMap[str, str]
    skill_versions: FrozenMap[str, str]
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _check_lineage(self) -> Self:
        if self.version == 1:
            if self.parent is not None:
                raise ValueError("第 1 版没有上一版本")
        elif self.parent is None or not 1 <= self.parent < self.version:
            raise ValueError("上一版本号必须小于当前版本号")
        changed = [c.field for c in self.diff]
        if len(changed) != len(set(changed)):
            raise ValueError("差异中同一字段出现多次")
        for change in self.diff:
            if change.field not in self.fields or self.fields[change.field] != change.to_value:
                raise ValueError(f"差异中 {change.field} 的新值与本版字段取值不一致")
        return self
