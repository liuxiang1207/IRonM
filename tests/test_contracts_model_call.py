"""模型调用记录（spec [2026-10-20] §4）。"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ironm.contracts.model_call import ModelCall


def _call(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "call_id": "c1",
        "request_id": "r1",
        "stage": "answer",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "prompt_template": "answer@1.2.0",
        "started_at": "2026-10-20T00:00:00+00:00",
        "duration_ms": 1830,
        "input_tokens": 1200,
        "output_tokens": 300,
        "status": "ok",
        "fallback_used": False,
    }
    data.update(overrides)
    return data


def _parse(data: dict[str, Any]) -> ModelCall:
    return ModelCall.model_validate_json(json.dumps(data))


def test_ok_call() -> None:
    _parse(_call())


def test_usage_may_be_unknown() -> None:
    _parse(_call(output_tokens={"unknown": "not_recorded"}))


def test_error_needs_error_type() -> None:
    with pytest.raises(ValidationError):
        _parse(_call(status="error", fallback_used=True))
    _parse(_call(status="error", error_type="HTTPStatusError", fallback_used=True))


def test_ok_must_not_carry_error_type() -> None:
    with pytest.raises(ValidationError):
        _parse(_call(error_type="Timeout"))


def test_no_raw_text_fields() -> None:
    for name in ModelCall.model_fields:
        assert not any(word in name for word in ("prompt_text", "response", "content", "message"))
    with pytest.raises(ValidationError):
        _parse(_call(response="今天体重 80 kg"))


def test_negative_duration_rejected() -> None:
    with pytest.raises(ValidationError):
        _parse(_call(duration_ms=-1))
