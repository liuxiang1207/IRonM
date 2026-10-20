"""黄金用例：tests/golden/*.yaml 驱动（spec [2026-10-20] §7）。"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from ironm.rules.loader import RULES_DIR, load_rules
from tests.casefiles import GoldenCase, GoldenFile, check_golden, load_case_file, resolve

FILES = {
    p.stem: load_case_file(p, GoldenFile) for p in sorted(Path(__file__).parent.glob("*.yaml"))
}
CASES = [(f, c) for f in FILES.values() for c in f.cases]


def test_golden_files_consistent() -> None:
    assert FILES, "没有任何黄金用例文件"
    check_golden(FILES.values(), load_rules(RULES_DIR))


@pytest.mark.parametrize(("file", "case"), CASES, ids=[c.id for _, c in CASES])
def test_golden(file: GoldenFile, case: GoldenCase, request: pytest.FixtureRequest) -> None:
    if file.status == "pending":
        request.applymarker(
            pytest.mark.xfail(raises=NotImplementedError, strict=True, reason=file.reason or "")
        )
    result = resolve(file.target, file.status)(**case.given)
    if isinstance(result, BaseModel):
        result = result.model_dump(mode="json")
    assert result == case.expect
