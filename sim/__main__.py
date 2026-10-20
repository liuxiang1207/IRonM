"""跑完全部模拟用户并输出汇总。用法：make sim（数据库写到 data/sim/sim.sqlite，每次重建）。"""

import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

from ironm.contracts.events import WeightEvent
from ironm.core.clock import FixedClock
from ironm.store.db import connect
from ironm.store.events import EventStore
from ironm.store.migrate import MIGRATIONS_DIR, migrate
from sim.generate import Simulation, generate
from sim.persona import PERSONAS_DIR, load_personas
from sim.runner import run

DEFAULT_DB = Path("data/sim/sim.sqlite")


def summary_row(sim: Simulation) -> str:
    counts = Counter(e.type for d in sim.days for e in d.events)
    # 只剔除假读数；换秤后的读数是真实测量（带偏差），照常统计
    fake_ids = {
        i for t in sim.truth.traps if t.kind in {"spike", "unit_error"} for i in t.event_ids
    }
    weights = [
        e.payload.weight_kg
        for d in sim.days
        for e in d.events
        if isinstance(e, WeightEvent) and e.event_id not in fake_ids
    ]
    truth = sim.truth.true_weight_kg
    measured = f"{weights[0]:.1f} → {weights[-1]:.1f}" if weights else "—"
    return (
        f"| {sim.persona_id} | {counts['weight']} | {counts['meal']} | {counts['workout_check']}"
        f" | {counts['checkin']} | {sum(len(t.event_ids) for t in sim.truth.traps)}"
        f" | {truth[0]:.1f} → {truth[-1]:.1f} | {measured} |"
    )


def main(db_path: Path) -> str:
    for suffix in ("", "-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    personas = load_personas(PERSONAS_DIR)
    first = next(iter(personas.values()))
    clock = FixedClock(first.start_at())
    lines = [
        "| 模拟用户 | 称重 | 饮食 | 训练 | 签到 | 陷阱记录"
        " | 真实体重（首 → 末） | 测得体重（首 → 末） |",
        "|---|---|---|---|---|---|---|---|",
    ]
    with closing(connect(db_path)) as conn:
        migrate(conn, MIGRATIONS_DIR, clock)
        store = EventStore(conn, clock)
        for p in personas.values():
            sim = generate(p)
            clock.set(p.start_at())
            run(sim, store, clock)
            lines.append(summary_row(sim))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB))
