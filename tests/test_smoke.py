"""工程骨架的冒烟测试：子包齐全且可导入、版本号单一来源、应用可启动。"""

import importlib
import pkgutil
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

import ironm
from ironm.api.app import app

ROOT = Path(__file__).resolve().parent.parent


def _registered_layers() -> set[str]:
    """依赖分层配置里登记的子包，是子包清单的唯一来源。"""
    with (ROOT / "pyproject.toml").open("rb") as f:
        [contract] = tomllib.load(f)["tool"]["importlinter"]["contracts"]
    return {name.strip() for layer in contract["layers"] for name in layer.split("|")}


def _subpackages() -> set[str]:
    return {m.name for m in pkgutil.iter_modules(ironm.__path__) if m.ispkg}


def test_subpackages_match_architecture() -> None:
    assert _subpackages() == _registered_layers()


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
