"""指标结果契约（spec [2026-10-20] §3）。"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ironm.contracts.metrics import MetricResult, MetricStatus


def _metric(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "weight_trend",
        "domain": "body",
        "value": -0.42,
        "unit": "kg/week",
        "ci": {"low": -0.61, "high": -0.23, "level": 0.95},
        "precision": "P2",
        "n_records": 9,
        "window_days": 28,
        "status": "ok",
        "needed": [],
        "last_record_at": "2026-10-19T23:30:00+00:00",
        "computed_at": "2026-10-20T00:00:00+00:00",
        "formula_id": "F-TREND-OLS-1",
    }
    data.update(overrides)
    return data


def _parse(data: dict[str, Any]) -> MetricResult:
    return MetricResult.model_validate_json(json.dumps(data))


UNKNOWN = {"unknown": "insufficient_data"}
NEED = [{"item": "weight_record", "count": 2}]


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"status": "insufficient", "value": UNKNOWN, "ci": None, "needed": NEED},
        {
            "status": "stale",
            "value": UNKNOWN,
            "ci": None,
            "last_record_at": {"unknown": "not_recorded"},
        },
        {"status": "insufficient", "ci": None, "needed": NEED},  # 有估计但不够确定
    ],
)
def test_valid_metrics(overrides: dict[str, Any]) -> None:
    metric = _parse(_metric(**overrides))
    assert MetricResult.model_validate_json(metric.model_dump_json()) == metric


def test_status_enum() -> None:
    assert _parse(_metric()).status is MetricStatus.OK


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        ({"value": UNKNOWN, "ci": None}, "正常状态但数值未知"),
        ({"status": "stale", "value": UNKNOWN}, "数值未知却带可信范围"),
        ({"ci": {"low": -0.3, "high": -0.1, "level": 0.95}}, "可信范围不含数值"),
        ({"ci": {"low": -0.2, "high": -0.6, "level": 0.95}}, "下限大于上限"),
        ({"status": "insufficient", "ci": None}, "数据不足却没写还差什么"),
        ({"ci": {"low": -0.61, "high": -0.23, "level": 1.5}}, "置信水平越界"),
        ({"precision": "P4"}, "精度等级越界"),
        ({"n_records": -1}, "记录条数为负"),
        ({"value": None}, "空值代替未知"),
        ({"last_record_at": None}, "最近记录时间用空值代替未知"),
    ],
)
def test_inconsistent_metrics_rejected(overrides: dict[str, Any], why: str) -> None:
    with pytest.raises(ValidationError):
        _parse(_metric(**overrides))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_value_rejected(literal: str) -> None:
    # value 没有上下限，能单独验证“拒绝非有限数字”，不会被范围校验顺带拦下
    raw = json.dumps(_metric(ci=None)).replace('"value": -0.42', f'"value": {literal}')
    assert literal in raw
    with pytest.raises(ValidationError):
        MetricResult.model_validate_json(raw)
