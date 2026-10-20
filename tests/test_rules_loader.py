"""规则表格式与加载（spec [2026-10-20] §6）。"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from ironm.rules.loader import RULES_DIR, RuleBook, RuleLoadError, RuleParamError, load_rules

GOOD = """\
rule_set: demo
version: 2026.10.1
rules:
  - id: R-DEMO-ONE
    title: 示例规则
    status: draft
    source:
      kind: design
    params:
      max_kg: 2.0
      days: 7
"""


class JumpParams(BaseModel):
    max_change_kg: float
    window_days: int


def _write(d: Path, name: str, text: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_text(text, encoding="utf-8")
    return path


def _load_one(tmp_path: Path, text: str, name: str = "demo.yaml") -> RuleBook:
    d = tmp_path / "rules"
    _write(d, name, text)
    return load_rules(d)


def test_repository_rules_load() -> None:
    book = load_rules(RULES_DIR)
    assert "data_quality" in book.versions()


def test_get_rule_and_typed_params() -> None:
    rule = load_rules(RULES_DIR).get("R-DQ-WEIGHT-JUMP")
    params = rule.params_as(JumpParams)
    assert params.max_change_kg == 2.0
    assert params.window_days == 7


def test_unknown_rule_id() -> None:
    with pytest.raises(KeyError, match="R-NOPE"):
        load_rules(RULES_DIR).get("R-NOPE")


def test_param_type_mismatch_names_rule(tmp_path: Path) -> None:
    text = GOOD.replace("days: 7", "days: 七天").replace("max_kg", "max_change_kg")
    text = text.replace("days:", "window_days:")
    rule = _load_one(tmp_path, text).get("R-DEMO-ONE")
    with pytest.raises(RuleParamError, match="R-DEMO-ONE"):
        rule.params_as(JumpParams)


def test_design_source_needs_no_ref(tmp_path: Path) -> None:
    assert _load_one(tmp_path, GOOD).versions() == {"demo": "2026.10.1"}


@pytest.mark.parametrize(
    ("text", "name"),
    [
        (GOOD.replace("    title: 示例规则\n", ""), "demo.yaml"),  # 缺标题
        (GOOD.replace("kind: design", "kind: guideline"), "demo.yaml"),  # 指南没写出处
        (GOOD.replace("    status: draft\n", "    status: draft\n    owner: me\n"), "demo.yaml"),
        (
            GOOD.replace("    status: draft\n", "    status: draft\n    status: approved\n"),
            "demo.yaml",
        ),
        (GOOD, "demo.yml"),  # 扩展名
        (GOOD, "other.yaml"),  # 规则集名与文件名不一致
        (GOOD.replace("version: 2026.10.1", "version: 2026-10-20"), "demo.yaml"),  # 未加引号的日期
        (GOOD.replace("id: R-DEMO-ONE", "id: demo-one"), "demo.yaml"),  # 编号格式
        (GOOD.replace("rules:\n", "rules: []\nx:\n"), "demo.yaml"),
        ("rule_set: [", "demo.yaml"),  # 不是合法 YAML
    ],
)
def test_bad_file_fails_naming_file(tmp_path: Path, text: str, name: str) -> None:
    with pytest.raises(RuleLoadError, match=name.replace(".", r"\.")):
        _load_one(tmp_path, text, name)


def test_duplicate_id_across_files(tmp_path: Path) -> None:
    d = tmp_path / "rules"
    _write(d, "demo.yaml", GOOD)
    _write(d / "sub", "demo2.yaml", GOOD.replace("rule_set: demo", "rule_set: demo2"))
    with pytest.raises(RuleLoadError, match="R-DEMO-ONE"):
        load_rules(d)


def test_duplicate_rule_set_name(tmp_path: Path) -> None:
    d = tmp_path / "rules"
    _write(d, "demo.yaml", GOOD)
    _write(d / "sub", "demo.yaml", GOOD.replace("R-DEMO-ONE", "R-DEMO-TWO"))
    with pytest.raises(RuleLoadError, match="demo"):
        load_rules(d)


def test_empty_dir_fails(tmp_path: Path) -> None:
    (tmp_path / "rules").mkdir()
    with pytest.raises(RuleLoadError, match="没有"):
        load_rules(tmp_path / "rules")


def test_missing_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(RuleLoadError):
        load_rules(tmp_path / "nope")


def test_app_startup_fails_on_bad_rules(tmp_path: Path) -> None:
    d = tmp_path / "rules"
    _write(d, "demo.yaml", GOOD.replace("kind: design", "kind: guideline"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("IRONM_")}
    env["IRONM_RULES_DIR"] = str(d)
    result = subprocess.run(
        [sys.executable, "-c", "import ironm.api.app"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "demo.yaml" in result.stderr


# ---- 独立审查后补充 ----


@pytest.mark.parametrize("name", ["extra.YAML", "x.yaml.bak", "notes.txt"])
def test_unexpected_files_fail(tmp_path: Path, name: str) -> None:
    d = tmp_path / "rules"
    _write(d, "demo.yaml", GOOD)
    _write(d, name, GOOD)
    with pytest.raises(RuleLoadError, match=re.escape(name)):
        load_rules(d)


def test_docs_and_hidden_files_allowed(tmp_path: Path) -> None:
    d = tmp_path / "rules"
    _write(d, "demo.yaml", GOOD)
    _write(d, "README.md", "# 说明")
    _write(d, ".DS_Store", "x")
    assert load_rules(d).versions() == {"demo": "2026.10.1"}


@pytest.mark.parametrize(
    ("literal", "kept"), [("1:30", "1:30"), ("010", "010"), ("yes", "yes"), ("on", "on")]
)
def test_yaml11_literals_stay_text(tmp_path: Path, literal: str, kept: str) -> None:
    rule = _load_one(tmp_path, GOOD.replace("days: 7", f"days: {literal}")).get("R-DEMO-ONE")
    assert rule.params["days"] == kept


def test_standard_literals_still_parsed(tmp_path: Path) -> None:
    text = GOOD.replace("days: 7", "days: 7\n      flag: true\n      ratio: -0.5\n      big: 1e3")
    params = _load_one(tmp_path, text).get("R-DEMO-ONE").params
    assert (params["days"], params["flag"], params["ratio"], params["big"]) == (
        7,
        True,
        -0.5,
        1000.0,
    )
