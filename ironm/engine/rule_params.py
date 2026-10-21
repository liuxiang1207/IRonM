"""规则参数登记：每条被计算使用的规则，登记它的参数类型；服务启动时逐条校验，写错即启动失败。"""

from pydantic import BaseModel

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
from ironm.safety.tiers import TIER_RULE, TierParams

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
    """规则缺失或参数不符时直接报错（错误信息含规则编号）。"""
    for rule_id, model in RULE_PARAMS.items():
        rulebook.get(rule_id).params_as(model)
