"""记录契约（spec [2026-10-20] §3）。"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ironm.contracts.events import EVENT_ADAPTER, Event, WeightEvent
from ironm.contracts.unknown import Unknown, UnknownReason

EID = "6f1c2a3e-1b2c-4d5e-8f90-123456789abc"
EID2 = "7a2b3c4d-5e6f-4a1b-9c2d-3e4f5a6b7c8d"

PAYLOADS: dict[str, dict[str, Any]] = {
    "workout_check": {"status": "half", "effort": 4},
    "set_log": {"exercise_id": "squat", "set_index": 1, "reps": 8, "load_kg": 60, "rir": 2},
    "weight": {"weight_kg": 80.4},
    "checkin": {"fatigue": 3, "diet_adherence_days": 5, "clothes_fit": "looser"},
    "meal": {"slot": "lunch", "kcal": 650, "protein_g": {"unknown": "not_recorded"}},
    "symptom": {"text": "右膝训练后有点酸", "severity": 2},
    "baseline_test": {"test_id": "pushup_max", "value": 18, "unit": "reps"},
}


def _event(type_: str = "weight", **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "event_id": EID,
        "user_id": "u1",
        "type": type_,
        "occurred_at": "2026-10-20T07:30:00+08:00",
        "recorded_at": "2026-10-20T07:31:00+08:00",
        "source": "user",
        "confirmed": True,
        "payload": PAYLOADS[type_],
    }
    data.update(overrides)
    return data


def _parse(data: dict[str, Any]) -> Event:
    return EVENT_ADAPTER.validate_json(json.dumps(data))


@pytest.mark.parametrize("type_", sorted(PAYLOADS))
def test_each_type_round_trips(type_: str) -> None:
    event = _parse(_event(type_))
    assert event.type == type_
    assert EVENT_ADAPTER.validate_json(EVENT_ADAPTER.dump_json(event)) == event


def test_times_normalized_to_utc() -> None:
    event = _parse(_event())
    assert event.occurred_at.isoformat() == "2026-10-19T23:30:00+00:00"


def test_payload_must_match_type() -> None:
    with pytest.raises(ValidationError):
        _parse(_event("weight", payload=PAYLOADS["meal"]))


@pytest.mark.parametrize(
    "overrides",
    [
        {"occurred_at": "2026-10-20T08:00:00+08:00"},  # 晚于记录时间
        {"occurred_at": "2026-10-20T07:30:00"},  # 不带时区
        {"mood": "good"},  # 未定义字段
        {"type": "steps"},  # 未定义类型
        {"user_id": ""},
        {"corrects": EID},  # 更正自己
    ],
)
def test_invalid_envelope_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _parse(_event(**overrides))


def test_device_source_requires_stream() -> None:
    with pytest.raises(ValidationError, match="stream_key"):
        _parse(_event(source="device"))
    event = _parse(_event(source="device", stream_key="scale:home"))
    assert event.stream_key == "scale:home"


@pytest.mark.parametrize("weight", [15, 500, "80", None])
def test_weight_out_of_range_or_wrong_type_rejected(weight: Any) -> None:
    with pytest.raises(ValidationError):
        _parse(_event(payload={"weight_kg": weight}))


def test_nan_rejected() -> None:
    raw = json.dumps(_event()).replace("80.4", "NaN")
    with pytest.raises(ValidationError):
        EVENT_ADAPTER.validate_json(raw)


def test_unknown_payload_field_rejected() -> None:
    with pytest.raises(ValidationError):
        _parse(_event(payload={"weight_kg": 80, "bmi": 24}))


def test_correction_points_to_original() -> None:
    event = _parse(_event(event_id=EID2, corrects=EID))
    assert str(event.corrects) == EID


def test_optional_measure_defaults_to_unknown_not_recorded() -> None:
    event = _parse(_event("meal", payload={"slot": "dinner"}))
    assert event.payload.kcal == Unknown(unknown=UnknownReason.NOT_RECORDED)  # type: ignore[union-attr]


def test_null_instead_of_unknown_rejected() -> None:
    with pytest.raises(ValidationError):
        _parse(_event("meal", payload={"slot": "dinner", "kcal": None}))


def test_unknown_without_reason_rejected() -> None:
    with pytest.raises(ValidationError):
        _parse(_event("meal", payload={"slot": "dinner", "kcal": {"unknown": None}}))
    with pytest.raises(ValidationError):
        _parse(_event("meal", payload={"slot": "dinner", "kcal": {}}))


def test_events_are_immutable() -> None:
    event = _parse(_event())
    assert isinstance(event, WeightEvent)
    with pytest.raises(ValidationError):
        event.user_id = "u2"
