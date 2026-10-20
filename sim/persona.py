"""模拟用户定义：可见档案、记录习惯、隐藏真值、陷阱。sim/personas/*.yaml 每个文件一个用户。"""

from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ironm.rules.loader import load_yaml

PERSONAS_DIR = Path(__file__).parent / "personas"

type Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
type Score = Annotated[int, Field(ge=1, le=5)]
type Day = Annotated[int, Field(ge=1)]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Profile(_Model):
    sex: Literal["male", "female"]
    age: int
    height_cm: float
    start_weight_kg: float


class Habits(_Model):
    weigh_days: tuple[Weekday, ...]
    training_days: tuple[Weekday, ...]
    meal_logging: Literal["kcal", "slot_only", "none"]  # 记热量 / 只记吃了 / 不记
    checkin_day: Weekday
    scale_stream: str


class Phase(_Model):
    """分阶段的隐藏变量：从第 from_week 周起生效，直到下一阶段。"""

    from_week: Annotated[int, Field(ge=1)]


class WeightPhase(Phase):
    kg_per_week: float


class AdherencePhase(Phase):
    p_done: Annotated[float, Field(ge=0, le=1)]  # 训练日完成整次训练的概率


class FatiguePhase(Phase):
    level: Score


class Hidden(_Model):
    """隐藏真值：只用于生成数据和事后判分，不出现在任何记录里。"""

    weight_phases: tuple[WeightPhase, ...] = Field(min_length=1)
    weight_noise_sd_kg: Annotated[float, Field(ge=0)]
    adherence_phases: tuple[AdherencePhase, ...] = Field(min_length=1)
    fatigue_phases: tuple[FatiguePhase, ...] = Field(min_length=1)
    daily_kcal: Annotated[float, Field(gt=0)]


class Trap(_Model):
    kind: Literal["spike", "unit_error", "device_switch", "stop_logging", "gap"]
    day: Day | None = None  # spike / unit_error / device_switch
    delta_kg: float | None = None  # spike
    stream: str | None = None  # device_switch
    offset_kg: float | None = None  # device_switch
    from_day: Day | None = None  # stop_logging / gap
    to_day: Day | None = None  # gap

    @model_validator(mode="after")
    def _check_fields(self) -> "Trap":
        required = {
            "spike": ("day", "delta_kg"),
            "unit_error": ("day",),
            "device_switch": ("day", "stream", "offset_kg"),
            "stop_logging": ("from_day",),
            "gap": ("from_day", "to_day"),
        }[self.kind]
        missing = [f for f in required if getattr(self, f) is None]
        if missing:
            raise ValueError(f"陷阱 {self.kind} 缺少 {', '.join(missing)}")
        return self


class Persona(_Model):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    title: str
    user_id: str
    seed: int
    start_date: date
    weeks: Annotated[int, Field(ge=1)]
    tz_offset_hours: int
    profile: Profile
    habits: Habits
    hidden: Hidden
    traps: tuple[Trap, ...] = ()

    def tz(self) -> timezone:
        return timezone(timedelta(hours=self.tz_offset_hours))

    def start_at(self) -> datetime:
        """第 1 天当地 0 点，换算为 UTC。"""
        local = datetime.combine(self.start_date, datetime.min.time(), self.tz())
        return local.astimezone(UTC)


def load_personas(directory: Path) -> dict[str, Persona]:
    personas: dict[str, Persona] = {}
    for path in sorted(directory.glob("*.yaml")):
        p = Persona.model_validate(load_yaml(path))
        if p.id != path.stem:
            raise ValueError(f"{path}：id {p.id} 与文件名不一致")
        personas[p.id] = p
    if not personas:
        raise ValueError(f"没有任何模拟用户：{directory}")
    return personas
