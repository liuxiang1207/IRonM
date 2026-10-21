"""动作库契约（spec §27）：动作模式、器材、身体限制与单个动作。"""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import Field, model_validator

from ironm.contracts.common import ContractModel

type Text = Annotated[str, Field(min_length=1)]
type ExerciseId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]


class Pattern(StrEnum):
    """动作模式：训练规划师按模式定槽位，动作教练在模式内选具体动作。"""

    SQUAT = "squat"  # 下蹲
    HINGE = "hinge"  # 髋铰链
    LUNGE = "lunge"  # 单腿
    PUSH = "push"  # 推
    PULL = "pull"  # 拉
    CORE = "core"  # 核心
    CARDIO = "cardio"  # 有氧


class Equipment(StrEnum):
    CHAIR = "chair"  # 椅子、床沿
    STEP = "step"  # 台阶、楼梯
    DUMBBELL = "dumbbell"
    KETTLEBELL = "kettlebell"
    BARBELL = "barbell"
    RACK = "rack"  # 深蹲架、史密斯架
    BENCH = "bench"  # 训练凳、箱子
    BAND = "band"  # 弹力带
    CABLE = "cable"  # 龙门架、绳索
    MACHINE = "machine"  # 固定器械
    PULL_UP_BAR = "pull_up_bar"
    CARDIO_MACHINE = "cardio_machine"  # 固定单车、椭圆机、跑步机
    JUMP_ROPE = "jump_rope"


class Restriction(StrEnum):
    """用户报告的身体限制（无红旗时；有红旗走安全通道）。"""

    KNEE_PAIN = "knee_pain"
    LOW_BACK_PAIN = "low_back_pain"
    SHOULDER_PAIN = "shoulder_pain"
    WRIST_PAIN = "wrist_pain"
    ANKLE_PAIN = "ankle_pain"


class Exercise(ContractModel):
    id: ExerciseId
    name: Text
    pattern: Pattern
    equipment: tuple[Equipment, ...] = ()  # 全部需要；徒手为空
    difficulty: Annotated[int, Field(ge=1, le=3)]
    unilateral: bool = False
    default: bool = False  # 该模式的默认安全动作
    avoid_if: tuple[Restriction, ...] = ()  # 有这些限制时排除
    caution_if: tuple[Restriction, ...] = ()  # 有这些限制时仍可选，附提示
    key_points: tuple[Text, ...] = Field(min_length=2, max_length=4)
    common_errors: tuple[Text, ...] = Field(min_length=1, max_length=3)
    self_check: tuple[Text, ...] = Field(min_length=1, max_length=2)
    easier: ExerciseId | None = None
    harder: ExerciseId | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if set(self.avoid_if) & set(self.caution_if):
            raise ValueError(f"{self.id}：同一限制不能既“避免”又“谨慎”")
        for name, items in [("equipment", self.equipment), ("avoid_if", self.avoid_if)]:
            if len(set(items)) != len(items):
                raise ValueError(f"{self.id}：{name} 有重复")
        if self.id in (self.easier, self.harder):
            raise ValueError(f"{self.id}：降阶或进阶不能是自己")
        return self
