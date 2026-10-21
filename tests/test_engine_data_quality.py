"""异常值“待确认”的逐次判定（spec [2026-10-20] §12）。

成对判定的边界见黄金用例 tests/golden/data_quality_*.yaml。
"""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from ironm.contracts.events import EVENT_ADAPTER, WeightEvent
from ironm.engine.data_quality import effective_weights, pending_weights, weight_jump_pending
from ironm.rules.loader import RULES_DIR, RuleBook, load_rules
from sim.generate import generate
from sim.persona import PERSONAS_DIR, load_personas

BOOK = load_rules(RULES_DIR)
_n = 0


def _w(
    day: int, kg: float, stream: str | None = None, corrects: WeightEvent | None = None
) -> WeightEvent:
    global _n
    _n += 1
    data: dict[str, Any] = {
        "event_id": f"00000000-0000-4000-8000-{_n:012d}",
        "user_id": "u1",
        "type": "weight",
        "occurred_at": f"2026-10-{day:02d}T07:30:00+08:00",
        "recorded_at": f"2026-10-{day:02d}T07:31:00+08:00",
        "source": "user",
        "confirmed": True,
        "payload": {"weight_kg": kg},
    }
    if stream is not None:
        data |= {"source": "device", "stream_key": stream}
    if corrects is not None:
        data["corrects"] = str(corrects.event_id)
    e = EVENT_ADAPTER.validate_json(json.dumps(data))
    assert isinstance(e, WeightEvent)
    return e


def test_result_carries_rule_id() -> None:
    a, b = _w(1, 80.0), _w(3, 83.0)
    r = weight_jump_pending(
        rulebook=BOOK,
        previous_kg=80.0,
        previous_at=a.occurred_at,
        current_kg=83.0,
        current_at=b.occurred_at,
    )
    assert r.pending_confirm and r.rule_id == "R-DQ-WEIGHT-JUMP"


def test_threshold_comes_from_rule_table(tmp_path: Path) -> None:
    rules = tmp_path / "rules"
    rules.mkdir()
    text = (RULES_DIR / "data_quality.yaml").read_text(encoding="utf-8")
    (rules / "data_quality.yaml").write_text(
        text.replace("max_change_kg: 2.0", "max_change_kg: 3.0")
    )
    a, b = _w(1, 80.0), _w(3, 82.5)

    def check(book: RuleBook) -> bool:
        return weight_jump_pending(
            rulebook=book,
            previous_kg=80.0,
            previous_at=a.occurred_at,
            current_kg=82.5,
            current_at=b.occurred_at,
        ).pending_confirm

    assert check(BOOK)
    assert not check(load_rules(rules))


def test_spike_flags_only_itself() -> None:
    series = [_w(1, 80.0), _w(4, 79.8), _w(7, 82.6), _w(10, 79.5), _w(13, 79.3)]
    assert pending_weights(series, BOOK) == {series[2].event_id: "R-DQ-WEIGHT-JUMP"}


def test_streams_not_compared() -> None:
    series = [_w(1, 80.0, stream="scale:home"), _w(3, 82.8, stream="scale:gym")]
    assert pending_weights(series, BOOK) == {}


def test_correction_supersedes_and_counts_as_confirmed() -> None:
    a = _w(1, 80.0)
    spike = _w(3, 83.0)
    fix = _w(3, 83.0, corrects=spike)  # 用户确认这次读数没错
    effective = effective_weights([a, spike, fix])
    assert [e.event_id for e in effective] == [a.event_id, fix.event_id]
    assert pending_weights([a, spike, fix], BOOK) == {}


def test_correction_chain() -> None:
    a = _w(1, 80.0)
    first_fix = _w(1, 81.0, corrects=a)
    second_fix = _w(1, 80.5, corrects=first_fix)
    assert [e.event_id for e in effective_weights([second_fix, a, first_fix])] == [
        second_fix.event_id
    ]


PERSONAS = load_personas(PERSONAS_DIR)


@pytest.mark.parametrize("pid", sorted(PERSONAS))
def test_sim_traps_flagged_and_nothing_else(pid: str) -> None:
    sim = generate(PERSONAS[pid])
    weights = [e for d in sim.days for e in d.events if isinstance(e, WeightEvent)]
    fake = {i for t in sim.truth.traps if t.kind in {"spike", "unit_error"} for i in t.event_ids}
    assert set(pending_weights(weights, BOOK)) == fake


def test_window_boundary_uses_days() -> None:
    a = _w(1, 80.0)
    later = a.occurred_at + timedelta(days=7)
    assert weight_jump_pending(
        rulebook=BOOK,
        previous_kg=80.0,
        previous_at=a.occurred_at,
        current_kg=77.5,
        current_at=later,
    ).pending_confirm


def test_baseline_is_recent_level_not_last_reading() -> None:
    # 第 4 天读数带正噪声（80.4）：只和上一次比，尖峰 82.3 只高 1.9 会漏判，
    # 而随后的正常读数 80.0 又会和尖峰比而被误标。和近期水平（中位数 80.0）比则都对。
    series = [_w(1, 79.6), _w(4, 80.4), _w(6, 82.3), _w(8, 80.0)]
    assert pending_weights(series, BOOK) == {series[2].event_id: "R-DQ-WEIGHT-JUMP"}


def test_no_baseline_outside_window() -> None:
    series = [_w(1, 80.0), _w(12, 82.5)]  # 相隔 11 天，窗口内没有可比读数
    assert pending_weights(series, BOOK) == {}


def test_wrong_first_reading_is_the_one_flagged() -> None:
    series = [_w(1, 88.0), _w(3, 80.1), _w(5, 80.0), _w(7, 79.9), _w(9, 79.8)]
    assert pending_weights(series, BOOK) == {series[0].event_id: "R-DQ-WEIGHT-JUMP"}


def test_real_shift_against_corroborated_baseline_still_flagged() -> None:
    # 基准由两条彼此一致的读数构成：之后的整体跳变是“真实的水平变化”还是错误，要用户确认
    series = [_w(1, 80.0), _w(2, 80.2), _w(4, 83.0), _w(5, 83.1), _w(6, 83.0)]
    assert set(pending_weights(series, BOOK)) == {e.event_id for e in series[2:]}


def test_single_baseline_not_overturned_without_enough_followers() -> None:
    series = [_w(1, 88.0), _w(3, 80.1)]  # 只有 1 条后续读数，不足以推翻基准
    assert pending_weights(series, BOOK) == {series[1].event_id: "R-DQ-WEIGHT-JUMP"}


def test_spike_followed_by_return_does_not_overturn_true_baseline() -> None:
    # 80 → 82.1（尖峰）→ 80.3 → 80.4：后续读数与基准一致、与尖峰不一致，应标尖峰而不是 80
    series = [_w(1, 80.0), _w(3, 82.1), _w(4, 80.3), _w(5, 80.4)]
    assert pending_weights(series, BOOK) == {series[1].event_id: "R-DQ-WEIGHT-JUMP"}


def test_followers_close_to_baseline_do_not_overturn_it() -> None:
    # 后续 81.3、81.4 与 82.1 一致（差 ≤ 1），但与基准 80 相差不到 2：不算“基准有误”
    series = [_w(1, 80.0), _w(3, 82.1), _w(4, 81.3), _w(5, 81.4)]
    pending = pending_weights(series, BOOK)
    assert series[0].event_id not in pending
    assert series[1].event_id in pending


def test_followers_must_agree_with_current_within_tolerance() -> None:
    # 后续 84.6、82.4 都与基准 80 相差超过 2，但 84.6 与 83 相差 1.6（超过容差 1）：不推翻基准
    series = [_w(1, 80.0), _w(3, 83.0), _w(4, 84.6), _w(5, 82.4)]
    pending = pending_weights(series, BOOK)
    assert series[0].event_id not in pending
    assert series[1].event_id in pending


def test_rebaseline_uses_percentage_threshold() -> None:
    """120 千克时阈值为 3 千克：后续读数与基准只差 2.5 千克 → 不推翻基准（spec §26）。"""
    base, jump = _w(1, 120.0), _w(2, 116.9)
    followers = [_w(3, 117.5), _w(4, 117.4)]
    assert set(pending_weights([base, jump, *followers], BOOK)) == {jump.event_id}
