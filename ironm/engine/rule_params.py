"""规则参数登记：每条被计算使用的规则，登记它的参数类型；服务启动时逐条校验，写错即启动失败。"""

from pydantic import BaseModel

from ironm.contracts.rules import RuleParamError
from ironm.engine.calories import (
    DEFICIT,
    FLOOR,
    MACRO,
    PAL,
    PROTEIN,
    REE,
    DeficitParams,
    FloorParams,
    MacroParams,
    PalParams,
    ProteinParams,
    ReeParams,
)
from ironm.engine.completeness import (
    TRAINING_COMPLETION,
    WEIGHT_STALE,
    CompletionParams,
    StaleParams,
)
from ironm.engine.data_quality import KCAL_RANGE, WEIGHT_JUMP, KcalRangeParams, WeightJumpParams
from ironm.engine.energy import EnergyParams, rule_id_for
from ironm.engine.metrics_view import DIET_MIN_DATA, DietMinDataParams
from ironm.engine.precision import (
    BODY,
    DIET,
    RECOVERY,
    TRAINING,
    BodyParams,
    DietParams,
    RecoveryParams,
    TrainingParams,
)
from ironm.engine.trend import WEIGHT_TREND, TrendParams
from ironm.engine.window import JUDGEMENT_WINDOW, WindowParams
from ironm.rules.loader import RuleBook
from ironm.safety.tiers import TIER_LIMITS_RULE, TIER_RULE, TierLimitsParams, TierParams

RULE_PARAMS: dict[str, type[BaseModel]] = {
    WEIGHT_JUMP: WeightJumpParams,
    KCAL_RANGE: KcalRangeParams,
    WEIGHT_STALE: StaleParams,
    TRAINING_COMPLETION: CompletionParams,
    WEIGHT_TREND: TrendParams,
    BODY: BodyParams,
    TRAINING: TrainingParams,
    DIET: DietParams,
    RECOVERY: RecoveryParams,
    JUDGEMENT_WINDOW: WindowParams,
    TIER_RULE: TierParams,
    TIER_LIMITS_RULE: TierLimitsParams,
    REE: ReeParams,
    PAL: PalParams,
    DEFICIT: DeficitParams,
    FLOOR: FloorParams,
    MACRO: MacroParams,
    PROTEIN: ProteinParams,
    DIET_MIN_DATA: DietMinDataParams,
    **{rule_id_for(level): EnergyParams for level in range(1, 6)},
}


def validate_rule_params(rulebook: RuleBook) -> None:
    """规则缺失或参数不符时直接报错（错误信息含规则编号）；再检查跨规则的一致性（spec §32）。"""
    for rule_id, model in RULE_PARAMS.items():
        rulebook.get(rule_id).params_as(model)
    _check_cross_rules(rulebook)


def _check_cross_rules(rulebook: RuleBook) -> None:
    def params[M: BaseModel](rule_id: str, model: type[M]) -> M:
        return rulebook.get(rule_id).params_as(model)

    ages = {
        TIER_RULE: params(TIER_RULE, TierParams).senior_age,
        DEFICIT: params(DEFICIT, DeficitParams).senior_age,
        PROTEIN: params(PROTEIN, ProteinParams).senior_age,
    }
    if len(set(ages.values())) != 1:
        raise RuleParamError(f"各规则中的老年年龄界值不一致：{ages}")
    window = params(JUDGEMENT_WINDOW, WindowParams).min_window_days
    span = params(WEIGHT_TREND, TrendParams).min_span_days
    if window < span:
        raise RuleParamError(
            f"{JUDGEMENT_WINDOW} 的最短窗口 {window} 天短于 {WEIGHT_TREND} 的最少覆盖 {span} 天"
        )
