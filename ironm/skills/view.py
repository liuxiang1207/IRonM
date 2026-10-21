"""按 Skill 声明裁剪的快照（spec §28）：Agent 只能读本次加载的 Skill 声明过的数据。"""

from collections.abc import Iterable

from pydantic import JsonValue

from ironm.contracts.snapshot import Snapshot
from ironm.contracts.unknown import Unknown, not_recorded

__all__ = ["SnapshotView", "UndeclaredDataError"]


class UndeclaredDataError(KeyError):
    pass


class SnapshotView:
    """只读视图。get(键) 读声明过的数据；声明了但快照里没有的为“未知（没记录）”；读未声明的报错。"""

    def __init__(self, snapshot: Snapshot, declared: Iterable[str]) -> None:
        self._snapshot = snapshot
        self._declared = frozenset(declared)

    @property
    def declared(self) -> frozenset[str]:
        return self._declared

    @property
    def user_id(self) -> str:
        return self._snapshot.user_id

    def get(self, key: str) -> object | Unknown:
        if key not in self._declared:
            raise UndeclaredDataError(f"本次加载的 Skill 没有声明数据 {key}，不能读取")
        s = self._snapshot
        fixed: dict[str, object] = {
            "tier": s.tier,
            "precision": s.precision_by_domain,
            "current_plan": s.current_plan,
            "constraints": s.constraints,
            "prior_verdicts": s.prior_verdicts,
        }
        if key in fixed:
            value = fixed[key]
            return not_recorded() if value is None else value
        group, _, name = key.partition(".")
        if group == "profile":
            profile: dict[str, JsonValue] = dict(s.profile)
            return profile[name] if name in profile else not_recorded()
        if group == "metric":
            return next((m for m in s.metrics if m.name == name), not_recorded())
        raise UndeclaredDataError(f"未知的数据键：{key}")
