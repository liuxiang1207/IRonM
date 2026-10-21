"""判断窗口（spec §19）。所需天数的边界见 tests/golden/judgement_window.yaml。"""

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ironm.contracts.events import EVENT_ADAPTER
from ironm.engine.window import judgement_window, required_window_days
from ironm.rules.loader import RULES_DIR, load_rules

BOOK = load_rules(RULES_DIR)
SH = ZoneInfo("Asia/Shanghai")
AS_OF = datetime(2026, 10, 28, 15, 0, tzinfo=UTC)  # 上海时间 10 月 28 日 23:00
_n = 0


def _w(day_offset: int, kg: float) -> Any:
    """AS_OF 当天往前 day_offset 天的早上 7:30（上海）称重。"""
    global _n
    _n += 1
    local = datetime(2026, 10, 28, 7, 30, tzinfo=SH) - timedelta(days=day_offset)
    data = {
        "event_id": f"00000000-0000-4000-8000-{_n:012d}",
        "user_id": "u1",
        "type": "weight",
        "occurred_at": local.isoformat(),
        "recorded_at": local.isoformat(),
        "source": "user",
        "confirmed": True,
        "payload": {"weight_kg": kg},
    }
    return EVENT_ADAPTER.validate_json(json.dumps(data))


def test_ready_when_coverage_enough() -> None:
    # 每天称、25 天覆盖，读数 ≥ 6 → 用自己的波动；所需天数 ≤ 25 → 正常
    rng = random.Random(1)
    events = [_w(d, round(80 - 0.06 * (25 - d) + rng.gauss(0, 0.3), 1)) for d in range(26)]
    r = judgement_window(events, as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.sd_source == "estimated"
    assert r.status == "ok", r


def test_not_ready_tells_days_remaining() -> None:
    # 每天称但只覆盖 10 天；数据完全线性，估计出的波动为 0，按下限 0.3 千克计
    events = [_w(d, 80.0 - 0.05 * (10 - d)) for d in range(11)]
    r = judgement_window(events, as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.status == "insufficient"
    assert isinstance(r.required_days, int)
    [need] = r.needed
    assert need.item == "days" and need.count == r.required_days - 10


def test_default_sd_when_few_readings() -> None:
    events = [_w(d, 80.0) for d in (0, 7, 14, 21)]
    r = judgement_window(events, as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.sd_source == "default" and r.residual_sd_kg == 0.5
    assert r.readings_per_week == 1.0
    assert r.required_days == 42


def test_extra_weigh_in_hint() -> None:
    events = [_w(d, 80.0) for d in (0, 7, 14, 21)]
    r = judgement_window(events, as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.faster_by_days == 8  # 每周 1 次 → 2 次：42 天 → 34 天


def test_no_weighing() -> None:
    r = judgement_window([], as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.status == "insufficient"
    assert [(n.item, n.count) for n in r.needed] == [("weigh_ins_per_week", 1)]


def test_target_from_rule_table(tmp_path: Path) -> None:
    d = tmp_path / "rules"
    d.mkdir()
    for f in RULES_DIR.glob("*.yaml"):
        text = f.read_text(encoding="utf-8").replace(
            "target_half_width_kg_per_week: 0.25", "target_half_width_kg_per_week: 0.5"
        )
        (d / f.name).write_text(text, encoding="utf-8")
    loose = required_window_days(rulebook=load_rules(d), readings_per_week=7, residual_sd_kg=0.5)
    strict = required_window_days(rulebook=BOOK, readings_per_week=7, residual_sd_kg=0.5)
    assert isinstance(loose.window_days, int) and isinstance(strict.window_days, int)
    assert loose.window_days < strict.window_days


def test_frequency_counts_days_and_skips_pending() -> None:
    base = [_w(d, 80.0) for d in (0, 7, 14, 21)]
    same_day = [_w(d, 80.1) for d in (0, 7)]  # 同一天第二次，不多算
    spike = _w(17, 84.0)  # 与 3 天前的读数比跳了 4 千克 → 待确认，不算
    r = judgement_window([*base, *same_day, spike], as_of=AS_OF, tz=SH, rulebook=BOOK)
    assert r.readings_per_week == 1.0


def test_extra_days_for_target() -> None:
    from ironm.engine.window import extra_days_for_target

    # 32 × (0.31 / 0.25)^(2/3) = 32 × 1.1541 ≈ 36.93 → 还差 5 天
    assert extra_days_for_target(32, 0.31, 0.25) == 5
    assert extra_days_for_target(32, 0.25, 0.25) == 0
    assert extra_days_for_target(30, 0.2501, 0.25) == 1  # 稍超也至少 1 天


def test_daily_weigher_with_huge_noise_gets_consistency_hint() -> None:
    plan = required_window_days(rulebook=BOOK, readings_per_week=7, residual_sd_kg=2.5)
    assert plan.status == "insufficient"
    assert [(n.item, n.count) for n in plan.needed] == [("consistent_weighing_time", 1)]


def test_new_user_frequency_uses_actual_span() -> None:
    week = [_w(d, 80.0) for d in range(7)]  # 第 0–6 天每天称
    assert judgement_window(week, as_of=AS_OF, tz=SH, rulebook=BOOK).readings_per_week == 7.0
    once = [_w(0, 80.0)]
    assert judgement_window(once, as_of=AS_OF, tz=SH, rulebook=BOOK).readings_per_week == 1.0
    veteran = [_w(d, 80.0) for d in (0, 7, 14, 21, 35)]  # 用满 4 周以上：35 天前那次不在 4 周内
    assert judgement_window(veteran, as_of=AS_OF, tz=SH, rulebook=BOOK).readings_per_week == 1.0
