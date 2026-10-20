"""跑完全部模拟用户并输出汇总。用法：make sim（数据库写到 data/sim/sim.sqlite，每次重建）。"""

import sys
from collections import Counter
from collections.abc import Iterable
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from ironm.contracts.common import Domain
from ironm.contracts.events import Event, WeightEvent
from ironm.contracts.metrics import MetricResult
from ironm.core.clock import FixedClock
from ironm.engine.metrics_view import MetricsReport, build_metrics, lookback_days
from ironm.rules.loader import RULES_DIR, load_rules
from ironm.store.db import connect
from ironm.store.events import EventStore
from ironm.store.migrate import MIGRATIONS_DIR, migrate
from sim.generate import Simulation, generate
from sim.persona import PERSONAS_DIR, Persona, load_personas
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


_STATUS = {
    "ok": "正常",
    "insufficient": "数据不足",
    "stale": "已过期",
    "pending_confirm": "有待确认",
}
_NEED = {
    "weight_record": "再称重 {} 次",
    "span_days": "再覆盖 {} 天",
    "days": "还差 {} 天",
    "weigh_ins_per_week": "每周至少称 {} 次",
    "diet_days": "还差 {} 天饮食记录",
}


def _end_of(sim: Simulation) -> datetime:
    """最后一天结束后的当地 0 点，即第 57 天开始。"""
    return sim.days[-1].closes_at + timedelta(hours=1)


def report_at_end(events: Iterable[Event], sim: Simulation, persona: Persona) -> MetricsReport:
    return build_metrics(
        events, as_of=_end_of(sim), tz=persona.tz(), rulebook=load_rules(RULES_DIR)
    )


def metrics_at_end(sim: Simulation, persona: Persona) -> MetricResult:
    """第 8 周末的体重趋势指标（直接用模拟生成的记录）。"""
    events = [e for d in sim.days for e in d.events]
    report = report_at_end(events, sim, persona)
    return next(m for m in report.metrics if m.name == "weight_trend")


def true_slope(sim: Simulation, window_days: int) -> float:
    """隐藏真值在最后 window_days 天的平均变化（千克 / 周）。"""
    truth = sim.truth.true_weight_kg
    span = min(window_days, len(truth)) - 1
    return (truth[-1] - truth[-1 - span]) / span * 7


def metrics_row(persona_id: str, report: MetricsReport, sim: Simulation) -> str:
    p = report.precision
    precision = "/".join(p[d] for d in (Domain.BODY, Domain.TRAINING, Domain.DIET, Domain.RECOVERY))
    m = next(x for x in report.metrics if x.name == "weight_trend")
    value = "—"
    if isinstance(m.value, float):
        value = f"{m.value:+.2f}"
        if m.ci is not None:
            value += f"（{m.ci.low:+.2f}, {m.ci.high:+.2f}）"
    needed = "；".join(_NEED[n.item].format(n.count) for n in m.needed) or "—"
    return (
        f"| {persona_id}（指标） | {precision} | {value} | {true_slope(sim, m.window_days):+.2f}"
        f" | {_STATUS[m.status]} | {needed} |"
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
    metric_lines = [
        "",
        "第 8 周末的指标（从数据库按回看范围取数，用指标组装计算）：",
        "",
        "| 模拟用户 | 精度（体重/训练/饮食/恢复） | 体重趋势 kg/周（95% 可信范围） | 真实趋势"
        " | 状态 | 还差什么 |",
        "|---|---|---|---|---|---|",
    ]
    rulebook = load_rules(RULES_DIR)
    with closing(connect(db_path)) as conn:
        migrate(conn, MIGRATIONS_DIR, clock)
        store = EventStore(conn, clock)
        for p in personas.values():
            sim = generate(p)
            clock.set(p.start_at())
            run(sim, store, clock)
            lines.append(summary_row(sim))
            since = _end_of(sim) - timedelta(days=lookback_days(rulebook))
            events = store.list_for_user(p.user_id, since=since)
            metric_lines.append(metrics_row(p.id, report_at_end(events, sim, p), sim))
    return "\n".join([*lines, *metric_lines]) + "\n"


if __name__ == "__main__":
    print(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DB))
