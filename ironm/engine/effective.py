"""有效记录层（spec §16）：所有计算统一从这里取得有效记录，不各自处理更正。

- 被更正取代的记录不参与计算；更正链取最新一版；同一条记录出现两个更正（分叉）直接报错。
- 更正记录若没有给出数据流，沿用它所更正的记录的数据流（用户确认设备读数时常见）。
- 给出截止时间时，只使用发生时间与记录时间都不晚于它的记录：截止时刻还不知道的更正不生效。
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, Self
from uuid import UUID


class CorrectionForkError(ValueError):
    """同一条记录被更正了两次，无法确定哪一版有效。"""


class _Record(Protocol):
    @property
    def event_id(self) -> UUID: ...
    @property
    def corrects(self) -> UUID | None: ...
    @property
    def occurred_at(self) -> datetime: ...
    @property
    def recorded_at(self) -> datetime: ...
    @property
    def stream_key(self) -> str | None: ...
    def model_copy(self, *, update: dict[str, object] | None = None) -> Self: ...


def resolve[E: _Record](events: Iterable[E], *, as_of: datetime | None = None) -> list[E]:
    """有效记录，按发生时间、记录标识排序。"""
    items = [
        e for e in events if as_of is None or (e.occurred_at <= as_of and e.recorded_at <= as_of)
    ]
    by_id = {e.event_id: e for e in items}
    corrected: dict[UUID, UUID] = {}
    for e in items:
        if e.corrects is None:
            continue
        if e.corrects in corrected and corrected[e.corrects] != e.event_id:
            raise CorrectionForkError(f"记录 {e.corrects} 被更正了两次，无法确定有效版本")
        corrected[e.corrects] = e.event_id

    def inherited_stream(e: E) -> str | None:
        seen: set[UUID] = set()
        current: E | None = e
        while current is not None and current.event_id not in seen:
            if current.stream_key is not None:
                return current.stream_key
            seen.add(current.event_id)
            current = by_id.get(current.corrects) if current.corrects is not None else None
        return None

    effective = []
    for e in items:
        if e.event_id in corrected:
            continue
        if e.corrects is not None and e.stream_key is None:
            stream = inherited_stream(e)
            if stream is not None:
                e = e.model_copy(update={"stream_key": stream})
        effective.append(e)
    return sorted(effective, key=lambda e: (e.occurred_at, str(e.event_id)))
