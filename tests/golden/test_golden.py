"""黄金用例：tests/golden/*.yaml 驱动（spec [2026-10-20] §7）。"""

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
    # 只比对用例写明的字段；结果可以多带信息（如规则编号）
    assert {k: result.get(k) for k in case.expect} == case.expect
