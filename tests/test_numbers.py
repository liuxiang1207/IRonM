"""数字校验（spec [2026-10-21] §7）：回答中的数字都要有登记过的来源；日期与序号不参与。"""

import pytest

from ironm.orchestrator.numbers import numbers_in, unregistered


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("每个动作 1–3 组、每组 8–12 次", {"1", "3", "8", "12"}),
        ("每天每千克体重约 1.6 克", {"1.6"}),
        ("每周 150-300 分钟", {"150", "300"}),
        ("每天 3～5 克", {"3", "5"}),
        ("做三组，每组十二次", {"3", "12"}),
        ("每周两次", {"2"}),
        ("没有数字", set()),
    ],
)
def test_numbers_found(text: str, expected: set[str]) -> None:
    assert numbers_in(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "2026年10月21日开始",
        "第3周",
        "第一步先热身",
        "一下子",
        "一定要",
        "一起做",
        "一般来说",
        "一边走",
    ],
)
def test_dates_ordinals_and_words_ignored(text: str) -> None:
    assert numbers_in(text) == set()


def test_unregistered_numbers_reported() -> None:
    registered = {"1", "3", "8", "12"}
    assert unregistered("每个动作 1–3 组、每组 8–12 次", registered) == set()
    assert unregistered("每组 15 次，每周 4 次", registered) == {"15", "4"}
    assert unregistered("做三组", registered) == set()
    assert unregistered("每天喝 2.5 升水", registered) == {"2.5"}
