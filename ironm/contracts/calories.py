"""热量目标（spec §25）：静息能量消耗、每日总消耗、缺口、底线、营养素与三餐。全部为估算。"""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from ironm.contracts.common import ContractModel
from ironm.contracts.unknown import Unknown

type Kcal = Annotated[int, Field(ge=0, le=20000)]
type Text = Annotated[str, Field(min_length=1)]


class Activity(StrEnum):
    """用户选择的身体活动水平（FAO/WHO/UNU 三档）。"""

    SEDENTARY = "sedentary"  # 久坐或轻度
    MODERATE = "moderate"  # 中度
    VIGOROUS = "vigorous"  # 重度


class Macros(ContractModel):
    carb_g: Annotated[int, Field(ge=0)]
    fat_g: Annotated[int, Field(ge=0)]
    protein_g: Annotated[int, Field(ge=0)]
    high_protein: bool


class CalorieTarget(ContractModel):
    """status：ok 已算出；no_plan 当前分层不给热量目标；insufficient 信息不全。"""

    status: Literal["ok", "no_plan", "insufficient"]
    estimated: Literal[True] = True  # 界面标“估算”
    ree_kcal: Kcal | Unknown
    pal: float | Unknown
    tdee_kcal: Kcal | Unknown
    deficit_kcal: Kcal | Unknown  # 实际缺口（被底线抬高后可能小于规则缺口，最小为 0）
    target_kcal: Kcal | Unknown
    floor_kcal: Kcal | Unknown
    floor_applied: bool  # 目标被底线抬高
    macros: Macros | Unknown
    meals_kcal: tuple[Kcal, ...] | Unknown  # 早、中、晚，之和等于目标
    reasons: tuple[Text, ...]  # 如 no_deficit_possible、tier_no_plan:c
    missing: tuple[Text, ...]
    rule_ids: tuple[Text, ...]
