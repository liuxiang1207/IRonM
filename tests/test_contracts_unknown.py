"""“未知”缺失值（spec [2026-10-20] §3）。"""

import pytest

from ironm.contracts.unknown import Unknown, UnknownReason, is_known

U = Unknown(unknown=UnknownReason.DECLINED)


def test_arithmetic_raises() -> None:
    with pytest.raises(TypeError):
        _ = U + 1  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = 1 + U  # type: ignore[operator]


def test_truthiness_raises() -> None:
    with pytest.raises(TypeError, match="未知"):
        bool(U)


def test_is_known() -> None:
    assert is_known(0.0)
    assert not is_known(U)


def test_json_shape() -> None:
    assert U.model_dump_json() == '{"unknown":"declined"}'
