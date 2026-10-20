"""红旗测试集（spec [2026-10-20] §7）。急症拦截实现前全部为预期失败。"""

from pathlib import Path

import pytest

from tests.casefiles import (
    NEGATIVE_CATEGORIES,
    POSITIVE_CATEGORIES,
    RedFlagCase,
    RedFlagFile,
    check_unique_ids,
    load_case_file,
    resolve,
)

FILE = load_case_file(Path(__file__).with_name("red_flags.yaml"), RedFlagFile)


def test_dataset_shape() -> None:
    cases = FILE.cases
    check_unique_ids(c.id for c in cases)
    assert len(cases) >= 40
    assert {c.expect for c in cases} == {"emergency", "red_flag", "none"}
    positives = {c.category for c in cases if c.expect != "none"}
    negatives = {c.category for c in cases if c.expect == "none"}
    assert positives == POSITIVE_CATEGORIES
    assert negatives == NEGATIVE_CATEGORIES
    assert sum(c.category == "negation" for c in cases) >= 4


@pytest.mark.parametrize("case", FILE.cases, ids=lambda c: c.id)
def test_red_flag(case: RedFlagCase, request: pytest.FixtureRequest) -> None:
    if FILE.status == "pending":
        request.applymarker(
            pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="急症拦截未实现")
        )
    assess = resolve(FILE.target, FILE.status)
    result = assess(case.text)
    assert getattr(result, "level", result) == case.expect
