"""每日精力选择与训练量调整（spec [2026-10-20] §14）。"""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from ironm.contracts.events import EVENT_ADAPTER, EnergyCheckEvent
from ironm.contracts.unknown import Unknown
from ironm.core.clock import local_day_bounds
from ironm.engine.energy import EnergyAdjustment, adjust_item, energy_today
from ironm.rules.loader import RULES_DIR, RuleBook, load_rules

BOOK = load_rules(RULES_DIR)
SH = ZoneInfo("Asia/Shanghai")
DAY = date(2026, 10, 20)
_n = 0


def _e(at: str, level: int) -> EnergyCheckEvent:
    global _n
    _n += 1
    data: dict[str, Any] = {
        "event_id": f"00000000-0000-4000-8000-{_n:012d}",
        "user_id": "u1",
        "type": "energy_check",
        "occurred_at": at,
        "recorded_at": at,
        "source": "user",
        "confirmed": True,
        "payload": {"level": level},
    }
    e = EVENT_ADAPTER.validate_json(json.dumps(data))
    assert isinstance(e, EnergyCheckEvent)
    return e


def _adj(level: int, book: RuleBook = BOOK) -> EnergyAdjustment:
    today = energy_today([_e("2026-10-20T08:00:00+08:00", level)], day=DAY, tz=SH, rulebook=book)
    assert isinstance(today.adjustment, EnergyAdjustment)
    return today.adjustment


def test_not_selected() -> None:
    today = energy_today([], day=DAY, tz=SH, rulebook=BOOK)
    assert today.selected is False
    assert isinstance(today.level, Unknown) and isinstance(today.adjustment, Unknown)


def test_level_three() -> None:
    a = _adj(3)
    assert (a.mode, a.volume_factor, a.rpe_cap_delta, a.rule_id) == (
        "reduced",
        0.8,
        -1,
        "R-RC-ENERGY-3",
    )


def test_last_choice_of_the_day_wins_and_yesterday_ignored() -> None:
    events = [
        _e("2026-10-19T21:00:00+08:00", 5),  # 昨天
        _e("2026-10-20T07:00:00+08:00", 2),
        _e("2026-10-20T09:00:00+08:00", 4),
    ]
    today = energy_today(events, day=DAY, tz=SH, rulebook=BOOK)
    assert today.level == 4


def test_timezone_decides_the_day() -> None:
    # 上海时间 10 月 20 日 0:30 = UTC 10 月 19 日 16:30
    event = _e("2026-10-20T00:30:00+08:00", 3)
    assert energy_today([event], day=DAY, tz=SH, rulebook=BOOK).selected
    assert not energy_today([event], day=DAY, tz=ZoneInfo("UTC"), rulebook=BOOK).selected


def test_local_day_bounds() -> None:
    start, end = local_day_bounds(DAY, SH)
    assert start == datetime(2026, 10, 19, 16, 0, tzinfo=UTC)
    assert end == datetime(2026, 10, 20, 16, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("level", "sets", "rpe"),
    [(5, 4, 8), (4, 4, 8), (3, 3, 7), (2, 2, 6), (1, 0, 8)],
)
def test_adjust_item(level: int, sets: int, rpe: float) -> None:
    item = adjust_item(planned_sets=4, planned_rpe_cap=8, adjustment=_adj(level))
    assert item.sets == sets
    if sets:
        assert item.rpe_cap == rpe


@pytest.mark.parametrize("level", [3, 2])
def test_single_set_not_reduced_to_zero(level: int) -> None:
    # 1 组 × 0.5 四舍五入会得到 0；除“很差”外至少保留 1 组
    assert adjust_item(planned_sets=1, planned_rpe_cap=8, adjustment=_adj(level)).sets == 1


def test_very_low_energy_carries_safety_hint() -> None:
    hint = _adj(1).safety_hint
    assert hint is not None and "安全通道" in hint
    assert _adj(3).safety_hint is None


def _rules_with(tmp_path: Path, old: str, new: str) -> RuleBook:
    d = tmp_path / "rules"
    d.mkdir()
    for f in RULES_DIR.glob("*.yaml"):
        text = f.read_text(encoding="utf-8")
        if f.name == "recovery.yaml":
            assert old in text
            text = text.replace(old, new, 1)
        (d / f.name).write_text(text, encoding="utf-8")
    return load_rules(d)


def test_factor_comes_from_rule_table(tmp_path: Path) -> None:
    book = _rules_with(tmp_path, "volume_factor: 0.8", "volume_factor: 0.6")
    assert adjust_item(planned_sets=5, planned_rpe_cap=8, adjustment=_adj(3, book)).sets == 3


def test_never_above_plan(tmp_path: Path) -> None:
    book = _rules_with(tmp_path, "volume_factor: 0.8", "volume_factor: 1.5")
    item = adjust_item(planned_sets=4, planned_rpe_cap=8, adjustment=_adj(3, book))
    assert item.sets == 4


def test_rpe_cap_never_raised(tmp_path: Path) -> None:
    book = _rules_with(tmp_path, "rpe_cap_delta: -1", "rpe_cap_delta: 1")
    assert adjust_item(planned_sets=4, planned_rpe_cap=8, adjustment=_adj(3, book)).rpe_cap == 8


def test_rounding_half_up() -> None:
    assert adjust_item(planned_sets=5, planned_rpe_cap=8, adjustment=_adj(2)).sets == 3
