"""快照与计划版本（spec [2026-10-20] §4）。"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ironm.contracts.plan import PlanVersion
from ironm.contracts.snapshot import Snapshot

PLAN_V2: dict[str, Any] = {
    "version": 2,
    "parent": 1,
    "fields": {"daily_kcal_target": 1750, "training_days_per_week": 3},
    "diff": [{"field": "daily_kcal_target", "from_value": 1900, "to_value": 1750}],
    "rule_set_versions": {"diet": "2026.10.1"},
    "skill_versions": {"diet_review": "1.0.0"},
    "created_at": "2026-10-20T00:00:00+00:00",
}

METRIC: dict[str, Any] = {
    "name": "weight_trend",
    "domain": "body",
    "value": -0.42,
    "unit": "kg/week",
    "precision": "P2",
    "n_records": 9,
    "window_days": 28,
    "status": "ok",
    "last_record_at": "2026-10-19T23:30:00+00:00",
    "computed_at": "2026-10-20T00:00:00+00:00",
    "formula_id": "F-TREND-OLS-1",
}


def _snapshot(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "snapshot_id": "s1",
        "user_id": "u1",
        "as_of": "2026-10-20T00:00:00+00:00",
        "tier": "a",
        "precision_by_domain": {"training": "P1", "body": "P2", "diet": "P1", "recovery": "P1"},
        "profile": {"age": 34, "sex": "male", "equipment": ["dumbbell"]},
        "metrics": [METRIC],
        "current_plan": PLAN_V2,
        "constraints": [],
        "prior_verdicts": [],
    }
    data.update(overrides)
    return data


def _snap(data: dict[str, Any]) -> Snapshot:
    return Snapshot.model_validate_json(json.dumps(data))


def _plan(**overrides: Any) -> PlanVersion:
    return PlanVersion.model_validate_json(json.dumps(PLAN_V2 | overrides))


def test_valid_snapshot_round_trips() -> None:
    snap = _snap(_snapshot())
    assert Snapshot.model_validate_json(snap.model_dump_json()) == snap


def test_snapshot_before_first_plan() -> None:
    _snap(_snapshot(current_plan=None))


def test_snapshot_attributes_read_only() -> None:
    snap = _snap(_snapshot())
    with pytest.raises(ValidationError):
        snap.tier = "b"  # type: ignore[assignment]


def test_snapshot_mappings_read_only() -> None:
    snap = _snap(_snapshot())
    with pytest.raises(TypeError):
        snap.profile["age"] = 20  # type: ignore[index]
    with pytest.raises(TypeError):
        snap.precision_by_domain["diet"] = "P3"  # type: ignore[index]


def test_precision_must_cover_all_domains() -> None:
    with pytest.raises(ValidationError, match="recovery"):
        _snap(_snapshot(precision_by_domain={"training": "P1", "body": "P2", "diet": "P1"}))


def test_duplicate_metric_rejected() -> None:
    with pytest.raises(ValidationError):
        _snap(_snapshot(metrics=[METRIC, METRIC]))


def test_unknown_tier_rejected() -> None:
    with pytest.raises(ValidationError):
        _snap(_snapshot(tier="e"))


def test_first_version() -> None:
    plan = _plan(version=1, parent=None, diff=[])
    assert plan.parent is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": 1, "parent": 0},  # 第 1 版带上一版本号
        {"version": 3, "parent": 3},
        {"version": 3, "parent": None},
        {"diff": [{"field": "daily_kcal_target", "from_value": 1900, "to_value": 1800}]},
        {"diff": [{"field": "weekly_cardio_minutes", "from_value": 60, "to_value": 90}]},
        {"fields": {"daily_kcal_target": 1750, "favorite_color": "red"}},
    ],
)
def test_invalid_plan_versions_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _plan(**overrides)


def test_plan_fields_read_only() -> None:
    with pytest.raises(TypeError):
        _plan().fields["daily_kcal_target"] = 0  # type: ignore[index]
