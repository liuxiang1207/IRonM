"""契约 JSON Schema 导出（spec [2026-10-20] §3）。"""

from pathlib import Path

from ironm.contracts.schema import SCHEMAS, render_schema

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def test_schema_files_up_to_date() -> None:
    for name in SCHEMAS:
        path = SCHEMA_DIR / f"{name}.schema.json"
        assert path.exists(), f"缺少 {path.name}，请运行 make schemas"
        assert path.read_text(encoding="utf-8") == render_schema(name), (
            f"{path.name} 与契约不一致，请运行 make schemas"
        )


def test_no_stale_schema_files() -> None:
    expected = {f"{name}.schema.json" for name in SCHEMAS}
    assert {p.name for p in SCHEMA_DIR.glob("*.schema.json")} == expected


def test_expected_contracts_exported() -> None:
    assert {
        "event",
        "metric_result",
        "unknown",
        "agent_output",
        "snapshot",
        "plan_version",
        "model_call",
        "rule_set",
    } <= set(SCHEMAS)
