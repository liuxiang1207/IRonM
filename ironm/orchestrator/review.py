"""复核（spec §31）：提案逐条对照安全官的约束；违反的被拒绝，并写明违反了哪条约束。

约束取值的比较方式：
- 数值上限 / 下限：提案中的所有数值（区间取两端）都不超过上限、不低于下限
- 字典形式（如 {"rpe": 7}）：只比较提案中同名键的数值
- 排除：提案新值与约束值相同即违反；热量目标排除 "deficit"（不设缺口）时，拒绝任何热量目标提案
"""

from collections.abc import Iterable, Mapping

from ironm.contracts.agents import Constraint, Proposal, Verdict
from ironm.contracts.common import deep_thaw, json_equal

__all__ = ["review_proposal"]


def _numbers(value: object) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, list | tuple):
        return [n for v in value for n in _numbers(v)]
    return []


def _pairs(limit: object, value: object) -> list[tuple[float, list[float]]]:
    """(界值, 提案中对应的数值)。字典形式只比较同名键。"""
    if isinstance(limit, Mapping):
        if not isinstance(value, Mapping):
            return []
        return [
            (float(lv), _numbers(value[k]))
            for k, lv in limit.items()
            if k in value and isinstance(lv, int | float) and not isinstance(lv, bool)
        ]
    if isinstance(limit, int | float) and not isinstance(limit, bool):
        return [(float(limit), _numbers(value))]
    return []


def _violates(c: Constraint, value: object) -> bool:
    limit = deep_thaw(c.value)
    if c.op == "exclude":
        if c.field == "daily_kcal_target" and limit == "deficit":
            return True  # 不设缺口：这时不给热量目标，按平常吃
        return json_equal(limit, value)
    for bound, numbers in _pairs(limit, value):
        if c.op == "max" and any(n > bound for n in numbers):
            return True
        if c.op == "min" and any(n < bound for n in numbers):
            return True
    return False


def review_proposal(p: Proposal, constraints: Iterable[Constraint]) -> Verdict:
    value = deep_thaw(p.to_value)
    for c in constraints:
        if c.field == p.field and _violates(c, value):
            reason = f"violates:{c.rule_id}:{c.field}:{c.op}"
            return Verdict(proposal_id=p.proposal_id, result="reject", reason_code=reason)
    return Verdict(proposal_id=p.proposal_id, result="pass")
