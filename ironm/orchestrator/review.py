"""复核（spec §31、§32）：提案逐条对照安全官的约束；违反或无法比较的都拒绝，并写明原因。

约束取值的比较方式：
- 数值上限 / 下限：提案中的所有数值（区间取两端）都不超过上限、不低于下限
- 字典形式（如 {"rpe": 7}）：比较提案中同名键的数值；提案缺少该键或该键不是数值即无法比较
- 排除：提案新值与约束值相同即违反；热量目标排除 "deficit"（不设缺口）时，拒绝任何热量目标提案
无法比较（取值形状不符、没有数值）时一律拒绝（uncomparable）：这是唯一的约束闸门，宁可错拒。
"""

from collections.abc import Iterable, Mapping

from ironm.contracts.agents import Constraint, Proposal, Verdict
from ironm.contracts.common import deep_thaw, json_equal

__all__ = ["review_proposal"]


class _Uncomparable(Exception):
    pass


def _numbers(value: object) -> list[float]:
    """提案取值中的全部数值；出现数值以外的东西（文字、真假值、字典）即无法比较。"""
    if isinstance(value, bool):
        raise _Uncomparable
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, list | tuple) and value:
        return [n for v in value for n in _numbers(v)]
    raise _Uncomparable


def _pairs(limit: object, value: object) -> list[tuple[float, list[float]]]:
    """(界值, 提案中对应的数值)。"""
    if isinstance(limit, Mapping):
        if not isinstance(value, Mapping) or not limit:
            raise _Uncomparable
        pairs = []
        for key, bound in limit.items():
            if key not in value or isinstance(bound, bool) or not isinstance(bound, int | float):
                raise _Uncomparable
            pairs.append((float(bound), _numbers(value[key])))
        return pairs
    if isinstance(limit, int | float) and not isinstance(limit, bool):
        return [(float(limit), _numbers(value))]
    raise _Uncomparable


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
        if c.field != p.field:
            continue
        try:
            violated = _violates(c, value)
            reason = f"violates:{c.rule_id}:{c.field}:{c.op}"
        except _Uncomparable:
            violated, reason = True, f"uncomparable:{c.rule_id}:{c.field}:{c.op}"
        if violated:
            return Verdict(proposal_id=p.proposal_id, result="reject", reason_code=reason)
    return Verdict(proposal_id=p.proposal_id, result="pass")
