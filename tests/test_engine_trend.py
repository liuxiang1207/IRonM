"""体重趋势（spec [2026-10-20] §13）。回归数值的手算样例见 tests/golden/weight_trend.yaml。"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ironm.contracts.events import EVENT_ADAPTER, WeightEvent
from ironm.engine.stats import t_quantile
from ironm.engine.trend import fit_trend, weight_trend
from ironm.rules.loader import RULES_DIR, load_rules

BOOK = load_rules(RULES_DIR)
START = datetime(2026, 10, 1, tzinfo=UTC)
_n = 0


@pytest.mark.parametrize(
    ("p", "df", "expected"),
    [
        (0.975, 1, 12.706),
        (0.975, 2, 4.303),
        (0.975, 3, 3.182),
        (0.975, 5, 2.571),
        (0.975, 10, 2.228),
        (0.975, 30, 2.042),
        (0.975, 100, 1.984),
        (0.95, 3, 2.353),
    ],
)
def test_t_quantile_matches_table(p: float, df: int, expected: float) -> None:
    assert round(t_quantile(p, df), 3) == expected


def _w(
    day: float, kg: float, stream: str = "scale:home", corrects: WeightEvent | None = None
) -> WeightEvent:
    global _n
    _n += 1
    at = (START + timedelta(days=day)).isoformat()
    data: dict[str, Any] = {
        "event_id": f"00000000-0000-4000-8000-{_n:012d}",
        "user_id": "u1",
        "type": "weight",
        "occurred_at": at,
        "recorded_at": at,
        "source": "device",
        "stream_key": stream,
        "confirmed": True,
        "payload": {"weight_kg": kg},
    }
    if corrects is not None:
        data["corrects"] = str(corrects.event_id)
    e = EVENT_ADAPTER.validate_json(json.dumps(data))
    assert isinstance(e, WeightEvent)
    return e


def test_only_eligible_readings_used() -> None:
    too_old = _w(0, 85.0)
    a, b = _w(10, 82.4), _w(13, 82.1)
    spike = _w(15, 85.5)  # 待确认
    wrong = _w(17, 70.0)
    fixed = _w(17, 81.6, corrects=wrong)  # 被更正取代的 70.0 不参与
    c = _w(20, 81.7)
    d = _w(24, 81.0)
    old_scale_late = _w(23, 90.0, stream="scale:gym")  # 另一台秤，且不是最近一次读数的数据流
    events = [too_old, a, b, spike, wrong, fixed, c, d, old_scale_late]
    fit = weight_trend(events, as_of=START + timedelta(days=24), window_days=14, rulebook=BOOK)
    expected = fit_trend(
        [(10, 82.4), (13, 82.1), (17, 81.6), (20, 81.7), (24, 81.0)],
        ci_level=0.95,
        min_readings=3,
        min_span_days=3,
    )
    # 天数起点不同只带来 1e-12 级的浮点差异
    assert fit.model_dump() == pytest.approx(expected.model_dump())


def test_latest_stream_wins() -> None:
    events = [
        _w(1, 80.0),
        _w(4, 79.7),
        _w(8, 80.6, stream="scale:gym"),
        _w(11, 80.3, stream="scale:gym"),
    ]
    fit = weight_trend(events, as_of=START + timedelta(days=12), window_days=14, rulebook=BOOK)
    assert fit.n == 2 and fit.status == "insufficient"  # 只用新秤的 2 次读数


def test_ci_level_from_rule_table(tmp_path: Path) -> None:
    rules = tmp_path / "rules"
    rules.mkdir()
    for f in RULES_DIR.glob("*.yaml"):
        text = f.read_text(encoding="utf-8").replace("ci_level: 0.95", "ci_level: 0.90")
        (rules / f.name).write_text(text, encoding="utf-8")
    events = [_w(d, kg) for d, kg in ((0, 82.4), (3, 82.1), (7, 81.6), (10, 81.7), (14, 81.0))]
    as_of = START + timedelta(days=14)
    wide = weight_trend(events, as_of=as_of, window_days=14, rulebook=BOOK)
    narrow = weight_trend(events, as_of=as_of, window_days=14, rulebook=load_rules(rules))
    assert narrow.ci_level == 0.90
    assert isinstance(wide.ci_low, float) and isinstance(wide.ci_high, float)
    assert isinstance(narrow.ci_low, float) and isinstance(narrow.ci_high, float)
    assert narrow.ci_high - narrow.ci_low < wide.ci_high - wide.ci_low
