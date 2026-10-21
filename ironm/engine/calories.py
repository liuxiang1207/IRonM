"""热量计算（spec §25、§26）：Mifflin 静息能量消耗 × 活动水平 − 按分层的缺口，不低于底线。

营养素与三餐按占比分配；蛋白质另有按参考体重的最低量与上限，多出的热量先从脂肪、再从碳水中扣。

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
from ironm.core.body import bmi
from ironm.core.clock import age_on
from ironm.rules.loader import RuleBook

REE = "R-CAL-REE"
PAL = "R-CAL-PAL"
DEFICIT = "R-CAL-DEFICIT"
FLOOR = "R-CAL-FLOOR"
MACRO = "R-CAL-MACRO"
PROTEIN = "R-CAL-PROTEIN"
RULE_IDS = (REE, PAL, DEFICIT, FLOOR, MACRO, PROTEIN)

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
    senior_age: Annotated[int, Field(gt=0)]
    senior_maintenance_bmi_below: Annotated[float, Field(gt=0)]


class FloorParams(_Params):
    female: Annotated[int, Field(gt=0)]
    male: Annotated[int, Field(gt=0)]
    absolute: Annotated[int, Field(gt=0)]
    not_below_ree: bool

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


class ProteinParams(_Params):
    ideal_bmi: Annotated[float, Field(gt=0)]
    adjust_from_bmi: Annotated[float, Field(gt=0)]
    adjust_factor: Annotated[float, Field(ge=0, le=1)]
    g_per_kg: Annotated[float, Field(gt=0)]
    g_per_kg_high_protein: Annotated[float, Field(gt=0)]
    senior_age: Annotated[int, Field(gt=0)]
    senior_male_g: Annotated[int, Field(ge=0)]
    senior_female_g: Annotated[int, Field(ge=0)]
    max_g_per_kg: Annotated[float, Field(gt=0)]
    max_pct: Pct
    fat_min_pct: Pct
    carb_min_pct: Pct

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.max_pct + self.fat_min_pct + self.carb_min_pct > 100:
            raise ValueError("蛋白质上限与脂肪、碳水下限之和不能超过 100")
        if max(self.g_per_kg, self.g_per_kg_high_protein) > self.max_g_per_kg:
            raise ValueError("每千克蛋白质最低量不能超过上限")
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


def _reference_weight(height_cm: float, weight_kg: float, p: ProteinParams) -> Decimal:
    """参考体重：BMI 未达界值用实际体重；达到时用校正体重（理想体重 + 系数 ×（实际 − 理想））。"""
    metres = _dec(height_cm) / 100
    weight = _dec(weight_kg)
    ideal = _dec(p.ideal_bmi) * metres * metres
    if weight / (metres * metres) < _dec(p.adjust_from_bmi):
        return weight
    return ideal + _dec(p.adjust_factor) * (weight - ideal)


def _macros(
    target: int,
    p: MacroParams,
    pp: ProteinParams,
    *,
    high_protein: bool,
    sex: Sex,
    age: int,
    height_cm: float,
    weight_kg: float,
) -> tuple[Macros, list[str]]:
    """营养素克数与原因。蛋白质取占比、每千克参考体重、老年推荐量中最大者，再受上限约束。"""
    if high_protein:
        carb, fat, protein = p.high_protein_carb, p.high_protein_fat, p.high_protein_protein
    else:
        carb, fat, protein = p.carb, p.fat, p.protein
    kcal = Decimal(target)
    reasons: list[str] = []
    by_share = kcal * protein / 100 / p.kcal_per_g_protein
    ref = _reference_weight(height_cm, weight_kg, pp)
    per_kg = _dec(pp.g_per_kg_high_protein if high_protein else pp.g_per_kg) * ref
    senior = 0
    if age >= pp.senior_age:
        senior = pp.senior_male_g if sex is Sex.MALE else pp.senior_female_g
    need = max(by_share, per_kg, Decimal(senior))
    cap = min(_dec(pp.max_g_per_kg) * ref, kcal * pp.max_pct / 100 / p.kcal_per_g_protein)
    protein_g = min(need, cap)
    if protein_g > by_share:
        reasons.append("protein_raised")
    if need > cap:
        reasons.append("protein_capped")
    # 蛋白质多出（或少出）的热量：先从脂肪扣（不低于下限），再从碳水扣
    extra = protein_g * p.kcal_per_g_protein - kcal * protein / 100
    fat_kcal = kcal * fat / 100
    fat_cut = max(Decimal(0), min(extra, fat_kcal - kcal * pp.fat_min_pct / 100))
    fat_kcal -= fat_cut
    carb_kcal = kcal * carb / 100 - (extra - fat_cut)
    return (
        Macros(
            carb_g=_round(carb_kcal / p.kcal_per_g_carb),
            fat_g=_round(fat_kcal / p.kcal_per_g_fat),
            protein_g=_round(protein_g),
            high_protein=high_protein,
        ),
        reasons,
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
    protein_p = rulebook.get(PROTEIN).params_as(ProteinParams)

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
    reasons: list[str] = []
    senior = age >= deficits.senior_age
    if senior and bmi(height_cm, weight_kg) < deficits.senior_maintenance_bmi_below:
        deficit = 0  # 老年人体重在适宜范围：只给维持热量
        reasons.append("senior_maintenance")
    if tier.tier is Tier.COLLABORATIVE:
        reasons.append("collaborative_minimum_intake")  # 用药协作层：目标是最低摄入，不是上限
    floor = max(floor_p.male if male else floor_p.female, floor_p.absolute)
    if floor_p.not_below_ree:
        floor = max(floor, _round(ree))

    raw = tdee - deficit
    floor_applied = raw < floor
    target = _round(max(raw, Decimal(floor)))
    tdee_kcal = _round(tdee)
    if floor_applied:
        reasons.append("floor_applied")
    if floor >= tdee:
        reasons.append("no_deficit_possible")  # 只能靠增加活动
    macros, macro_reasons = _macros(
        target,
        macro_p,
        protein_p,
        high_protein=high_protein,
        sex=sex,
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
    )

    return CalorieTarget(
        status="ok",
        ree_kcal=_round(ree),
        pal=float(pal),
        tdee_kcal=tdee_kcal,
        deficit_kcal=max(0, tdee_kcal - target),
        target_kcal=target,
        floor_kcal=floor,
        floor_applied=floor_applied,
        macros=macros,
        meals_kcal=_meals(target, macro_p.meals),
        reasons=(*reasons, *macro_reasons),
        missing=(),
        rule_ids=RULE_IDS,
    )
