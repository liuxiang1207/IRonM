"""工程骨架的冒烟测试：子包齐全且可导入、版本号单一来源、应用可启动。"""

import importlib
import pkgutil
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

import ironm
from ironm.api.app import app

ROOT = Path(__file__).resolve().parent.parent

# 架构约定的子包。新增子包时同时加在这里，确保是有意为之而不是误建
EXPECTED_PACKAGES = {
    "agents",
    "api",
    "contracts",
    "engine",
    "knowledge",
    "llm",
    "memory",
    "orchestrator",
    "rules",
    "safety",
    "scheduler",
    "skills",
    "store",
}


def _subpackages() -> set[str]:
    return {m.name for m in pkgutil.iter_modules(ironm.__path__) if m.ispkg}


def test_subpackages_match_architecture() -> None:
    assert _subpackages() == EXPECTED_PACKAGES


def test_every_module_imports() -> None:
    for mod in pkgutil.walk_packages(ironm.__path__, prefix="ironm."):
        importlib.import_module(mod.name)


def test_version_single_source() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert ironm.__version__ == declared


def test_healthz() -> None:
    resp = TestClient(app).get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": ironm.__version__}
