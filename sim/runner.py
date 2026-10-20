"""快进运行器：逐天推进时钟，把当天的记录写入仓储，并在每天、每周结束时调用回调。"""

from collections.abc import Callable
from datetime import timedelta

from ironm.core.clock import FixedClock
from ironm.store.events import EventStore
from sim.generate import Simulation


def run(
    sim: Simulation,
    store: EventStore,
    clock: FixedClock,
    on_day_end: Callable[[int], None] | None = None,
    on_week_end: Callable[[int], None] | None = None,
) -> None:
    for day in sim.days:
        clock.set(day.closes_at)
        for event in day.events:
            store.append(event)
        if on_day_end is not None:
            on_day_end(day.index)
        if day.index % 7 == 0 and on_week_end is not None:
            on_week_end(day.index // 7)
    # 停在最后一天之后的当地 0 点，即第 57 天开始
    last = sim.days[-1]
    clock.set(last.closes_at + timedelta(hours=1))
