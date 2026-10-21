"""快照：一次运行中所有 Agent 读取的同一份只读数据。Agent 之间不通信，只读快照。"""

from typing import Annotated, Self

from pydantic import Field, JsonValue, model_validator

from ironm.contracts.agents import Constraint, Verdict
from ironm.contracts.common import ContractModel, Domain, FrozenMap, Precision, UtcDatetime
from ironm.contracts.metrics import MetricResult
from ironm.contracts.plan import PlanVersion
from ironm.contracts.screening import Tier
from ironm.contracts.unknown import Unknown


class Snapshot(ContractModel):
    snapshot_id: Annotated[str, Field(min_length=1)]
    user_id: Annotated[str, Field(min_length=1)]
    as_of: UtcDatetime
    tier: Tier | Unknown  # 还没建档时为“未知”
    tier_matched: tuple[Tier, ...] = ()  # 同时满足的所有分层（热量等计算取更保守的约束）
    precision_by_domain: FrozenMap[Domain, Precision]
    profile: FrozenMap[str, JsonValue]
    metrics: tuple[MetricResult, ...]
    current_plan: PlanVersion | None  # 首版计划之前为空
    constraints: tuple[Constraint, ...]
    prior_verdicts: tuple[Verdict, ...]  # 上一轮被拒的理由写进下一轮快照

    @model_validator(mode="after")
    def _check(self) -> Self:
        missing = set(Domain) - set(self.precision_by_domain)
        if missing:
            raise ValueError(f"精度表缺少领域：{', '.join(sorted(missing))}")
        names = [m.name for m in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("同名指标重复")
        return self
