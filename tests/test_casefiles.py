"""测试数据的一致性检查与“未实现”判定（spec [2026-10-20] §7）。"""

import sys
from pathlib import Path

import pytest

from ironm.rules.loader import RULES_DIR, load_rules
from tests.casefiles import CaseError, GoldenCase, GoldenFile, check_golden, resolve

BOOK = load_rules(RULES_DIR)


def _golden(*cases: GoldenCase, rules: tuple[str, ...] = ("R-DQ-KCAL-RANGE",)) -> GoldenFile:
    return GoldenFile(target="m:f", status="pending", rules=rules, cases=cases)


def _case(
    id_: str, given: dict[str, object], expect: dict[str, object], twin: str | None = None
) -> GoldenCase:
    return GoldenCase(id=id_, title="t", given=given, expect=expect, twin_of=twin)


A = _case("G-X-1", {"kcal": 499}, {"pending": True})


def test_valid_twin_passes() -> None:
    check_golden([_golden(A, _case("G-X-2", {"kcal": 500}, {"pending": False}, "G-X-1"))], BOOK)


def test_duplicate_id_across_files() -> None:
    with pytest.raises(CaseError, match="G-X-1"):
        check_golden([_golden(A), _golden(A)], BOOK)


def test_unknown_rule() -> None:
    with pytest.raises(CaseError, match="R-NOPE"):
        check_golden([_golden(A, rules=("R-NOPE",))], BOOK)


def test_twin_differs_in_two_conditions() -> None:
    twin = _case("G-X-2", {"kcal": 500, "day": 2}, {"pending": False}, "G-X-1")
    with pytest.raises(CaseError, match="只差一个条件"):
        check_golden([_golden(A, twin)], BOOK)


def test_twin_with_same_expectation() -> None:
    twin = _case("G-X-2", {"kcal": 498}, {"pending": True}, "G-X-1")
    with pytest.raises(CaseError, match="期望却相同"):
        check_golden([_golden(A, twin)], BOOK)


def test_missing_target_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        resolve("ironm.no_such_module:f", "pending")
    with pytest.raises(NotImplementedError):
        resolve("ironm.core.clock:no_such_function", "pending")


def test_other_import_errors_are_not_masked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "broken_target_mod.py").write_text("import no_such_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="no_such_dependency_xyz"):
        resolve("broken_target_mod:f", "pending")
    sys.modules.pop("broken_target_mod", None)


def test_active_target_must_exist() -> None:
    with pytest.raises(ModuleNotFoundError):
        resolve("ironm.no_such_module:f", "active")


def test_active_target_missing_function_is_error() -> None:
    with pytest.raises(CaseError, match="no_such_function"):
        resolve("ironm.core.clock:no_such_function", "active")
