"""把契约导出为 JSON Schema 文件（schemas/ 目录）。用法：python -m ironm.contracts.schema。"""

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from ironm.contracts.agents import AGENT_OUTPUT_ADAPTER
from ironm.contracts.events import EVENT_ADAPTER
from ironm.contracts.metrics import MetricResult
from ironm.contracts.model_call import ModelCall
from ironm.contracts.plan import PlanVersion
from ironm.contracts.snapshot import Snapshot
from ironm.contracts.unknown import Unknown

SCHEMAS: dict[str, TypeAdapter[Any]] = {
    "event": EVENT_ADAPTER,
    "metric_result": TypeAdapter(MetricResult),
    "unknown": TypeAdapter(Unknown),
    "agent_output": AGENT_OUTPUT_ADAPTER,
    "snapshot": TypeAdapter(Snapshot),
    "plan_version": TypeAdapter(PlanVersion),
    "model_call": TypeAdapter(ModelCall),
}


def render_schema(name: str) -> str:
    schema = SCHEMAS[name].json_schema(mode="validation")
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def export(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.schema.json"):
        stale.unlink()
    for name in SCHEMAS:
        (out_dir / f"{name}.schema.json").write_text(render_schema(name), encoding="utf-8")


if __name__ == "__main__":
    export(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("schemas"))
