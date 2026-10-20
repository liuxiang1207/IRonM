"""黄金用例：tests/golden/*.yaml 驱动（spec [2026-10-20] §7）。"""

import math
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, validate_call

from ironm.rules.loader import RULES_DIR, load_rules
from tests.casefiles import GoldenCase, GoldenFile, check_golden, load_case_file, resolve

FILES = {
    p.stem: load_case_file(p, GoldenFile) for p in sorted(Path(__file__).parent.glob("*.yaml"))
}
CASES = [(f, c) for f in FILES.values() for c in f.cases]
BOOK = load_rules(RULES_DIR)


def _close(a: object, b: object) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b and type(a) is type(b)
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    return a == b


def test_golden_files_consistent() -> None:
    assert FILES, "没有任何黄金用例文件"
    check_golden(FILES.values(), BOOK)


@pytest.mark.parametrize(("file", "case"), CASES, ids=[c.id for _, c in CASES])
def test_golden(file: GoldenFile, case: GoldenCase, request: pytest.FixtureRequest) -> None:
    if file.status == "pending":
        request.applymarker(
            pytest.mark.xfail(raises=NotImplementedError, strict=True, reason=file.reason or "")
        )
    fn = resolve(file.target, file.status)
    kwargs = dict(case.given)
    if "rulebook" in file.inject:
        kwargs["rulebook"] = BOOK
    # 按被测函数的参数类型转换 YAML 中的取值（如时间文字 → 带时区的时间）
    result = validate_call(config=ConfigDict(arbitrary_types_allowed=True))(fn)(**kwargs)
    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json")
    assert isinstance(result, dict)
    # 只比对用例写明的字段；结果可以多带信息（如规则编号）。小数允许 1e-6 的误差
    actual = {k: result.get(k) for k in case.expect}
    assert _close(actual, case.expect), f"{actual} != {case.expect}"
