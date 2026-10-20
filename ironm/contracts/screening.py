"""安全分层的输入与结果（spec §23）：筛查信息、健康状况、红旗记录、分层结果。"""

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ironm.contracts.common import ContractModel, UtcDatetime
from ironm.contracts.unknown import NOT_RECORDED, Unknown

type Text = Annotated[str, Field(min_length=1)]


class Tier(StrEnum):
    """安全分层，按从严到宽排列；第一期只有减脂场景。"""

    PAUSED = "paused"  # 红旗暂停
    D = "d"  # 暂不开放（未成年）
    C = "c"  # 记录与协作
    NOT_APPLICABLE = "not_applicable"  # 不适用（减脂）：BMI 过低
    COLLABORATIVE = "collaborative"  # 医生指导下用药协作
    B = "b"  # 保守
    A_PRIME = "a_prime"  # 体态减脂
    A = "a"  # 完整


# 从严到宽的次序：分层变化时据此判断“更严 / 更松”
TIER_ORDER: tuple[Tier, ...] = tuple(Tier)


class Condition(StrEnum):
    """筛查问卷中可申报的健康状况。各自归入哪一层写在规则表里。"""

    HYPERTENSION = "hypertension"  # 高血压
    RECENT_INJURY = "recent_injury"  # 近期受伤
    WEIGHT_LOSS_DRUG_SUPERVISED = "weight_loss_drug_supervised"  # 医生指导下使用减重药物
    WEIGHT_LOSS_DRUG_UNSUPERVISED = "weight_loss_drug_unsupervised"  # 未在医生指导下使用减重药物
    PREGNANT = "pregnant"  # 孕期
    BREASTFEEDING = "breastfeeding"  # 哺乳期
    POSTPARTUM_WITHIN_YEAR = "postpartum_within_year"  # 产后 1 年内
    HEART_DISEASE = "heart_disease"  # 心脏病
    KIDNEY_DISEASE = "kidney_disease"  # 肾脏病
    DIABETES_ON_MEDICATION = "diabetes_on_medication"  # 正在用药的糖尿病
    INJURY_REHAB = "injury_rehab"  # 伤后康复中
    EATING_DISORDER_HISTORY = "eating_disorder_history"  # 进食障碍史
    EATING_DISORDER_SIGNAL = "eating_disorder_signal"  # 进食障碍信号


class Sex(StrEnum):
    MALE = "male"
    FEMALE = "female"


class SafetyFlag(ContractModel):
    """一次红旗或急症判定的记录；不含用户原话。"""

    at: UtcDatetime
    level: Literal["emergency", "red_flag"]
    rule_ids: tuple[Text, ...] = Field(min_length=1)
    categories: tuple[Text, ...] = ()


class Screening(ContractModel):
    """分层所需的筛查信息；每项都可以是“未知”。"""

    birth_date: date | Unknown = NOT_RECORDED
    sex: Sex | Unknown = NOT_RECORDED
    height_cm: Annotated[float, Field(ge=100, le=250)] | Unknown = NOT_RECORDED
    weight_kg: Annotated[float, Field(ge=20, le=400)] | Unknown = NOT_RECORDED
    waist_cm: Annotated[float, Field(ge=40, le=250)] | Unknown = NOT_RECORDED
    # 问卷没答为“未知”；答了“以上都没有”为空
    conditions: tuple[Condition, ...] | Unknown = NOT_RECORDED
    flags: tuple[SafetyFlag, ...] = ()
    clearances: tuple[UtcDatetime, ...] = ()  # “已就医、可以运动”的确认时间

    @model_validator(mode="after")
    def _check(self) -> Self:
        if isinstance(self.conditions, tuple) and len(set(self.conditions)) != len(self.conditions):
            raise ValueError("健康状况重复")
        return self


class TierResult(ContractModel):
    """分层结果。tier 为主层；matched 为满足的所有层（从严到宽）；missing 为还差的信息。"""

    tier: Tier | Unknown
    status: Literal["ok", "assumed", "insufficient"]  # assumed：按保守假设给出
    matched: tuple[Tier, ...]
    reasons: tuple[Text, ...]
    missing: tuple[Text, ...]
    bmi: float | Unknown
    rule_id: Text
