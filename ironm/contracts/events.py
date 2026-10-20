"""记录（事件）契约。记录只追加不修改；改错是追加一条同类型的更正记录。

规整后的内容按类型固定字段、统一单位（kg、kcal、g）；原文保存在 raw。
数值上下限只拦物理上不可能的输入【设计】；可疑但可能的值由异常值规则标“待确认”。
"""

from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, TypeAdapter, model_validator

from ironm.contracts.common import ContractModel, Precision, UtcDatetime
from ironm.contracts.unknown import NOT_RECORDED, Unknown

type Score = Annotated[int, Field(ge=1, le=5)]


class Source(StrEnum):
    USER = "user"
    DEVICE = "device"
    ESTIMATE = "estimate"


# ---- 各类型的规整内容 ----


class WorkoutCheckPayload(ContractModel):
    status: Literal["done", "half", "skipped"]
    effort: Score | Unknown = NOT_RECORDED
    plan_session_id: str | None = None


class SetLogPayload(ContractModel):
    exercise_id: str = Field(min_length=1)
    set_index: int = Field(ge=1)
    reps: int = Field(ge=0, le=1000)
    load_kg: float = Field(ge=0, le=1000)  # 自重动作为 0：真实的“无外加负重”，不是缺失
    rir: Annotated[int, Field(ge=0, le=10)] | Unknown = NOT_RECORDED
    rpe: Annotated[float, Field(ge=1, le=10)] | Unknown = NOT_RECORDED
    workout_id: str | None = None


class WeightPayload(ContractModel):
    weight_kg: float = Field(ge=20, le=400)


class CheckinPayload(ContractModel):
    """每周签到。"""

    fatigue: Score | Unknown = NOT_RECORDED
    sleep_quality: Score | Unknown = NOT_RECORDED
    energy: Score | Unknown = NOT_RECORDED
    hunger: Score | Unknown = NOT_RECORDED
    clothes_fit: Literal["looser", "same", "tighter"] | Unknown = NOT_RECORDED
    diet_adherence_days: Annotated[int, Field(ge=0, le=7)] | Unknown = NOT_RECORDED
    record_completeness: (
        Literal["full", "partial", "followed_plan_unrecorded", "elsewhere"] | Unknown
    ) = NOT_RECORDED
    menstruation: bool | Unknown = NOT_RECORDED


class MealPayload(ContractModel):
    slot: Literal["breakfast", "lunch", "dinner", "snack"]
    kcal: Annotated[float, Field(ge=0, le=10000)] | Unknown = NOT_RECORDED
    protein_g: Annotated[float, Field(ge=0, le=1000)] | Unknown = NOT_RECORDED


class SymptomPayload(ContractModel):
    text: str = Field(min_length=1)
    severity: Score | Unknown = NOT_RECORDED


class BaselineTestPayload(ContractModel):
    test_id: str = Field(min_length=1)
    value: float = Field(ge=0, le=100000)
    unit: Literal["reps", "s", "kg", "cm", "m"]


class CircumferencePayload(ContractModel):
    """围度，单位厘米。腰围必填，其他部位可选。"""

    waist_cm: float = Field(ge=40, le=250)
    hip_cm: Annotated[float, Field(ge=50, le=250)] | Unknown = NOT_RECORDED
    chest_cm: Annotated[float, Field(ge=50, le=250)] | Unknown = NOT_RECORDED
    upper_arm_cm: Annotated[float, Field(ge=10, le=100)] | Unknown = NOT_RECORDED
    thigh_cm: Annotated[float, Field(ge=20, le=150)] | Unknown = NOT_RECORDED


class EnergyCheckPayload(ContractModel):
    """每日精力：1 很差、2 疲惫、3 一般、4 良好、5 充沛。"""

    level: Score


# ---- 记录 ----


class _EventBase(ContractModel):
    event_id: UUID  # 客户端生成，去重键
    user_id: str = Field(min_length=1)
    occurred_at: UtcDatetime
    recorded_at: UtcDatetime
    source: Source
    confirmed: bool
    raw: str | None = None  # 用户原话或设备原始字段
    stream_key: str | None = Field(default=None, min_length=1)  # 数据流，不同设备不混算
    corrects: UUID | None = None  # 更正记录指向被更正的记录
    precision_hint: Precision | None = None

    @model_validator(mode="after")
    def _check_envelope(self) -> Self:
        if self.occurred_at > self.recorded_at:
            raise ValueError("occurred_at 不得晚于 recorded_at")
        if self.source is Source.DEVICE and self.stream_key is None:
            raise ValueError("设备来源必须提供 stream_key")
        if self.corrects == self.event_id:
            raise ValueError("更正记录不能指向自己")
        return self


class WorkoutCheckEvent(_EventBase):
    type: Literal["workout_check"]
    payload: WorkoutCheckPayload


class SetLogEvent(_EventBase):
    type: Literal["set_log"]
    payload: SetLogPayload


class WeightEvent(_EventBase):
    type: Literal["weight"]
    payload: WeightPayload


class CheckinEvent(_EventBase):
    type: Literal["checkin"]
    payload: CheckinPayload


class MealEvent(_EventBase):
    type: Literal["meal"]
    payload: MealPayload


class SymptomEvent(_EventBase):
    type: Literal["symptom"]
    payload: SymptomPayload


class BaselineTestEvent(_EventBase):
    type: Literal["baseline_test"]
    payload: BaselineTestPayload


class CircumferenceEvent(_EventBase):
    type: Literal["circumference"]
    payload: CircumferencePayload


class EnergyCheckEvent(_EventBase):
    type: Literal["energy_check"]
    payload: EnergyCheckPayload


type Event = Annotated[
    WorkoutCheckEvent
    | SetLogEvent
    | WeightEvent
    | CheckinEvent
    | MealEvent
    | SymptomEvent
    | BaselineTestEvent
    | CircumferenceEvent
    | EnergyCheckEvent,
    Field(discriminator="type"),
]

EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)
