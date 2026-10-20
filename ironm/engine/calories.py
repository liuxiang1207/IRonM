"""热量计算（spec §25）：Mifflin 静息能量消耗 × 活动水平 − 按分层的缺口，不低于底线；营养素与三餐。

全部按十进制计算，最后四舍五入到整千卡（避免二进制小数在 .5 处出错）。结果标为估算。
"""

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Self
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ironm.contracts.calories import Activity, CalorieTarget, Macros
from ironm.contracts.screening import Screening, Sex, Tier, TierResult
from ironm.contracts.unknown import Unknown, UnknownReason, is_known
from ironm.core.clock import age_on
from ironm.rules.loader import RuleBook

REE = "R-CAL-REE"
PAL = "R-CAL-PAL"
DEFICIT = "R-CAL-DEFICIT"
FLOOR = "R-CAL-FLOOR"
MACRO = "R-CAL-MACRO"
RULE_IDS = (REE, PAL, DEFICIT, FLOOR, MACRO)

# 可给热量目标的分层 → 规则表中缺口参数的名称（须与三档输出中“可给计划”的分层一致）
PLAN_TIERS: dict[Tier, str] = {
    Tier.A: "a",
    Tier.A_PRIME: "a_prime",
    Tier.B: "b",
    Tier.COLLABORATIVE: "collaborative",
}

type Pct = Annotated[int, Field(ge=0, le=100)]


class _Params(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReeParams(_Params):
    weight_coef: float
    height_coef: float
    age_coef: float
    male_const: float
    female_const: float


class PalParams(_Params):
    sedentary: Annotated[float, Field(ge=1)]
    moderate: Annotated[float, Field(ge=1)]
    vigorous: Annotated[float, Field(ge=1)]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.sedentary < self.moderate < self.vigorous:
            raise ValueError("活动水平应依次增大")
        return self


class DeficitParams(_Params):
    a: Annotated[int, Field(ge=0)]
    a_prime: Annotated[int, Field(ge=0)]
    b: Annotated[int, Field(ge=0)]
    collaborative: Annotated[int, Field(ge=0)]


class FloorParams(_Params):
    female: Annotated[int, Field(gt=0)]
    male: Annotated[int, Field(gt=0)]
    absolute: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if min(self.female, self.male) < self.absolute:
            raise ValueError("性别底线不能低于绝对底线")
        return self


class MacroParams(_Params):
    carb: Pct
    fat: Pct
    protein: Pct
    high_protein_carb: Pct
    high_protein_fat: Pct
    high_protein_protein: Pct
    kcal_per_g_carb: Annotated[int, Field(gt=0)]
    kcal_per_g_fat: Annotated[int, Field(gt=0)]
    kcal_per_g_protein: Annotated[int, Field(gt=0)]
    meals: tuple[Pct, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.carb + self.fat + self.protein != 100:
            raise ValueError("营养素占比之和应为 100")
        hp = self.high_protein_carb + self.high_protein_fat + self.high_protein_protein
        if hp != 100:
            raise ValueError("高蛋白选项的营养素占比之和应为 100")
        if sum(self.meals) != 100:
            raise ValueError("三餐占比之和应为 100")
        return self


def _round(value: Decimal) -> int:
    """四舍五入到整数（0.5 进位）。"""
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _dec(value: float) -> Decimal:
    return Decimal(str(value))


def _unknown() -> Unknown:
    return Unknown(unknown=UnknownReason.INSUFFICIENT_DATA)


def _empty(
    status: str, reasons: tuple[str, ...] = (), missing: tuple[str, ...] = ()
) -> CalorieTarget:
    return CalorieTarget.model_validate(
        {
            "status": status,
            "ree_kcal": _unknown(),
            "pal": _unknown(),
            "tdee_kcal": _unknown(),
            "deficit_kcal": _unknown(),
            "target_kcal": _unknown(),
            "floor_kcal": _unknown(),
            "floor_applied": False,
            "macros": _unknown(),
            "meals_kcal": _unknown(),
            "reasons": reasons,
            "missing": missing,
            "rule_ids": RULE_IDS,
        }
    )


def _macros(target: int, p: MacroParams, high_protein: bool) -> Macros:
    if high_protein:
        carb, fat, protein = p.high_protein_carb, p.high_protein_fat, p.high_protein_protein
    else:
        carb, fat, protein = p.carb, p.fat, p.protein
    kcal = Decimal(target)

    def grams(pct: int, per_g: int) -> int:
        return _round(kcal * pct / 100 / per_g)

    return Macros(
        carb_g=grams(carb, p.kcal_per_g_carb),
        fat_g=grams(fat, p.kcal_per_g_fat),
        protein_g=grams(protein, p.kcal_per_g_protein),
        high_protein=high_protein,
    )


def _meals(target: int, shares: tuple[int, ...]) -> tuple[int, ...]:
    """按占比分配，最后一餐取差额，保证之和等于目标。"""
    first = [_round(Decimal(target) * pct / 100) for pct in shares[:-1]]
    return (*first, target - sum(first))


def calorie_target(
    screening: Screening,
    activity: Activity | Unknown,
    tier: TierResult,
    *,
    as_of: datetime,
    tz: ZoneInfo,
    rulebook: RuleBook,
    high_protein: bool = False,
) -> CalorieTarget:
    """每日热量目标（估算）。分层不给计划或信息不全时给不出，并写明原因或还差什么。"""
    if not is_known(tier.tier):
        return _empty("no_plan", ("tier_unknown",))
    if tier.tier not in PLAN_TIERS:
        return _empty("no_plan", (f"tier_no_plan:{tier.tier}",))

    s = screening
    age = age_on(s.birth_date, as_of.astimezone(tz).date()) if is_known(s.birth_date) else None
    missing = [
        name
        for name, absent in [
            ("sex", not is_known(s.sex)),
            ("birth_date", age is None),
            ("height_cm", not is_known(s.height_cm)),
            ("weight_kg", not is_known(s.weight_kg)),
            ("activity", not is_known(activity)),
        ]
        if absent
    ]
    if (
        missing
        or age is None
        or not is_known(s.sex)
        or not is_known(s.height_cm)
        or not is_known(s.weight_kg)
        or not is_known(activity)
    ):
        return _empty("insufficient", missing=tuple(missing))
    return _compute(
        s.sex, age, s.height_cm, s.weight_kg, activity, tier, rulebook, high_protein=high_protein
    )


def _compute(
    sex: Sex,
    age: int,
    height_cm: float,
    weight_kg: float,
    activity: Activity,
    tier: TierResult,
    rulebook: RuleBook,
    *,
    high_protein: bool,
) -> CalorieTarget:
    ree_p = rulebook.get(REE).params_as(ReeParams)
    pal_p = rulebook.get(PAL).params_as(PalParams)
    deficits = rulebook.get(DEFICIT).params_as(DeficitParams)
    floor_p = rulebook.get(FLOOR).params_as(FloorParams)
    macro_p = rulebook.get(MACRO).params_as(MacroParams)

    male = sex is Sex.MALE
    ree = (
        _dec(ree_p.weight_coef) * _dec(weight_kg)
        + _dec(ree_p.height_coef) * _dec(height_cm)
        - _dec(ree_p.age_coef) * age
        + _dec(ree_p.male_const if male else ree_p.female_const)
    )
    pal = _dec(getattr(pal_p, activity.value))
    tdee = ree * pal
    # 同时满足多层时取最小的缺口；按保守假设给出的主层也计入
    tiers = {*tier.matched, *([tier.tier] if is_known(tier.tier) else [])}
    deficit = min(getattr(deficits, PLAN_TIERS[t]) for t in tiers if t in PLAN_TIERS)
    floor = max(floor_p.male if male else floor_p.female, floor_p.absolute)

    raw = tdee - deficit
    floor_applied = raw < floor
    target = _round(max(raw, Decimal(floor)))
    tdee_kcal = _round(tdee)
    reasons: list[str] = []
    if floor_applied:
        reasons.append("floor_applied")
    if floor >= tdee:
        reasons.append("no_deficit_possible")  # 只能靠增加活动

    return CalorieTarget(
        status="ok",
        ree_kcal=_round(ree),
        pal=float(pal),
        tdee_kcal=tdee_kcal,
        deficit_kcal=max(0, tdee_kcal - target),
        target_kcal=target,
        floor_kcal=floor,
        floor_applied=floor_applied,
        macros=_macros(target, macro_p, high_protein),
        meals_kcal=_meals(target, macro_p.meals),
        reasons=tuple(reasons),
        missing=(),
        rule_ids=RULE_IDS,
    )
