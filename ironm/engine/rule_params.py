"""规则参数登记：每条被计算使用的规则，登记它的参数类型；服务启动时逐条校验，写错即启动失败。"""

from pydantic import BaseModel

from ironm.engine.completeness import (
    TRAINING_COMPLETION,
    WEIGHT_STALE,
    CompletionParams,
    StaleParams,
)
from ironm.engine.data_quality import KCAL_RANGE, WEIGHT_JUMP, KcalRangeParams, WeightJumpParams
from ironm.engine.energy import EnergyParams, rule_id_for
from ironm.engine.trend import WEIGHT_TREND, TrendParams
from ironm.rules.loader import RuleBook

RULE_PARAMS: dict[str, type[BaseModel]] = {
    WEIGHT_JUMP: WeightJumpParams,
    KCAL_RANGE: KcalRangeParams,
    WEIGHT_STALE: StaleParams,
    TRAINING_COMPLETION: CompletionParams,
    WEIGHT_TREND: TrendParams,
    **{rule_id_for(level): EnergyParams for level in range(1, 6)},
}


def validate_rule_params(rulebook: RuleBook) -> None:
    """规则缺失或参数不符时直接报错（错误信息含规则编号）。"""
    for rule_id, model in RULE_PARAMS.items():
        rulebook.get(rule_id).params_as(model)
