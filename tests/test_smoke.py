import importlib

import ironm

PACKAGES = [
    "contracts",
    "safety",
    "engine",
    "rules",
    "store",
    "memory",
    "orchestrator",
    "agents",
    "skills",
    "llm",
    "knowledge",
    "scheduler",
    "api",
]


def test_version() -> None:
    assert ironm.__version__ == "0.1.0"


def test_packages_import() -> None:
    for name in PACKAGES:
        importlib.import_module(f"ironm.{name}")
