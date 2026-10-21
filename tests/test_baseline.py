"""基线测试方案（spec [2026-10-20] §31）。"""

import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from ironm.api.app import create_app
from ironm.contracts.events import EVENT_ADAPTER
from ironm.contracts.screening import Tier
from ironm.contracts.unknown import not_recorded
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.knowledge.baseline import (
    BASELINE_PATH,
    BaselineLoadError,
    default_baseline,
    load_baseline,
)
from tests.test_api_events import NOW

BOOK = default_baseline()


def _ids(tier: Any, **screen: bool) -> tuple[tuple[str, ...], str | None]:
    p = BOOK.protocol(
        tier, unwell_today=screen.get("unwell", False), recent_injury=screen.get("injury", False)
    )
    return p.test_ids, p.skipped_reason


def test_protocol_by_tier() -> None:
    assert _ids(Tier.A) == (("push_up@1", "sit_to_stand_30s@1", "plank@1", "resting_hr@1"), None)
    assert _ids(Tier.A_PRIME)[0] == _ids(Tier.A)[0]
    b = ("knee_push_up@1", "sit_to_stand_30s@1", "resting_hr@1")
    assert _ids(Tier.B) == (b, None)  # 不做平板支撑
    assert _ids(Tier.COLLABORATIVE) == (b, None)
    for tier in (Tier.C, Tier.D, Tier.PAUSED, Tier.NOT_APPLICABLE):
        assert _ids(tier) == (("resting_hr@1",), "tier_no_fitness_test")
    assert _ids(not_recorded()) == (("resting_hr@1",), "tier_unknown")


def test_pretest_screening_skips_fitness() -> None:
    assert _ids(Tier.A, unwell=True) == (("resting_hr@1",), "pretest_unwell")
    assert _ids(Tier.A, injury=True) == (("resting_hr@1",), "pretest_recent_injury")


def test_every_test_has_guidance() -> None:
    for t in BOOK.tests:
        assert t.instructions and t.common_errors and t.stop_when
    assert BOOK.review == "approved"


def test_resting_hr_recordable_in_bpm() -> None:
    event = {
        "event_id": "6f1c2a3e-1b2c-4d5e-8f90-123456789abc",
        "user_id": "u1",
        "type": "baseline_test",
        "occurred_at": "2026-10-21T07:30:00+08:00",
        "recorded_at": "2026-10-21T07:31:00+08:00",
        "source": "user",
        "confirmed": True,
        "payload": {"test_id": "resting_hr@1", "value": 68, "unit": "bpm"},
    }
    EVENT_ADAPTER.validate_json(json.dumps(event))


def _data() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(BASELINE_PATH.read_text(encoding="utf-8"))
    return data


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["tests"].append(dict(d["tests"][0])), "编号重复"),
        (lambda d: d["tests"][0].update(version=0), "version"),
        (lambda d: d["tests"][0].update(tiers=["z"]), "tiers"),
        (lambda d: d["tests"][0].update(stop_when=[]), "stop_when"),
        (lambda d: d.update(tests=[t for t in d["tests"] if t["kind"] != "vital"]), "生命体征"),
    ],
)
def test_bad_book_rejected(tmp_path: Path, mutate: Any, message: str) -> None:
    data = _data()
    mutate(data)
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(BaselineLoadError, match=message):
        load_baseline(path)


def test_startup_fails_on_bad_book(tmp_path: Path) -> None:
    data = _data()
    data["tests"][0]["version"] = 0
    bad = tmp_path / "baseline.yaml"
    bad.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})
    with pytest.raises(BaselineLoadError):
        create_app(settings, FixedClock(NOW), log_stream=io.StringIO(), baseline_path=bad)
