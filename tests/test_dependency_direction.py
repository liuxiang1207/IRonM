"""模块依赖方向检查（spec [2026-10-20] §1）。

把 ironm 复制到临时目录，注入一条违规引用，再对副本跑依赖检查，确认能被拦下；
同时确认允许的依赖不被误报。
"""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LINT_IMPORTS = Path(sys.executable).parent / "lint-imports"


def _lint(tmp_path: Path, injections: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """复制 ironm 到 tmp_path，按 {相对路径: 追加的代码} 注入后运行依赖检查。"""
    shutil.copytree(ROOT / "ironm", tmp_path / "ironm")
    for rel, code in injections.items():
        target = tmp_path / "ironm" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(f"\n{code}\n")
    return subprocess.run(
        [str(LINT_IMPORTS), "--config", str(ROOT / "pyproject.toml"), "--no-cache"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_broken(result: subprocess.CompletedProcess[str], *names: str) -> None:
    out = result.stdout + result.stderr
    assert result.returncode != 0, out
    assert "broken" in out, out
    for name in names:
        assert name in out, out


def test_current_code_passes(tmp_path: Path) -> None:
    result = _lint(tmp_path, {})
    assert result.returncode == 0, result.stdout + result.stderr


def test_safety_cannot_import_other_business_module(tmp_path: Path) -> None:
    result = _lint(tmp_path, {"safety/__init__.py": "import ironm.agents"})
    _assert_broken(result, "ironm.safety", "ironm.agents")


def test_safety_cannot_import_rules(tmp_path: Path) -> None:
    result = _lint(tmp_path, {"safety/__init__.py": "import ironm.rules"})
    _assert_broken(result, "ironm.safety", "ironm.rules")


def test_safety_may_import_contracts(tmp_path: Path) -> None:
    result = _lint(tmp_path, {"safety/__init__.py": "import ironm.contracts"})
    assert result.returncode == 0, result.stdout + result.stderr


def test_lower_layer_cannot_import_upper(tmp_path: Path) -> None:
    result = _lint(tmp_path, {"engine/__init__.py": "import ironm.orchestrator"})
    _assert_broken(result, "ironm.engine", "ironm.orchestrator")


def test_agents_cannot_import_store(tmp_path: Path) -> None:
    result = _lint(tmp_path, {"agents/__init__.py": "import ironm.store"})
    _assert_broken(result, "ironm.agents", "ironm.store")


def test_skills_cannot_import_store(tmp_path: Path) -> None:
    result = _lint(tmp_path, {"skills/__init__.py": "import ironm.store"})
    _assert_broken(result, "ironm.skills", "ironm.store")


def test_unregistered_package_fails(tmp_path: Path) -> None:
    result = _lint(tmp_path, {"newpkg/__init__.py": ""})
    out = result.stdout + result.stderr
    assert result.returncode != 0, out
    assert "ironm.newpkg" in out, out


# ---- 独立审查后收紧：只有编排层与对外接口能直接用数据存储 ----


@pytest.mark.parametrize("module", ["scheduler", "memory", "rules", "safety", "llm"])
def test_only_orchestrator_and_api_use_store(tmp_path: Path, module: str) -> None:
    result = _lint(tmp_path, {f"{module}/__init__.py": "import ironm.store"})
    _assert_broken(result, f"ironm.{module}", "ironm.store")


@pytest.mark.parametrize("module", ["orchestrator", "api"])
def test_orchestrator_and_api_may_use_store(tmp_path: Path, module: str) -> None:
    result = _lint(tmp_path, {f"{module}/__init__.py": "import ironm.store"})
    assert result.returncode == 0, result.stdout + result.stderr


def test_store_ban_covers_every_other_package() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        contracts = tomllib.load(f)["tool"]["importlinter"]["contracts"]
    [layers] = [c for c in contracts if c["type"] == "layers"]
    [ban] = [c for c in contracts if c["type"] == "forbidden"]
    packages = {n.strip() for layer in layers["layers"] for n in layer.split("|")}
    banned = {m.removeprefix("ironm.") for m in ban["source_modules"]}
    assert banned == packages - {"api", "orchestrator", "store"}
