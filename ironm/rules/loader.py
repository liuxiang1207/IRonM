"""加载规则目录下的全部 *.yaml。任何格式问题都直接报错，应用因此启动失败。"""

import json
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from ironm.contracts.rules import Rule, RuleParamError, RuleSet
from ironm.core.yamlfile import load_yaml

__all__ = ["RULES_DIR", "RuleBook", "RuleLoadError", "RuleParamError", "load_rules", "load_yaml"]

RULES_DIR = Path(__file__).resolve().parents[2] / "rules"


class RuleLoadError(ValueError):
    pass


class UnknownRuleError(KeyError):
    pass


class RuleBook:
    def __init__(self, rule_sets: Sequence[RuleSet]) -> None:
        self._rules = {r.id: r for rs in rule_sets for r in rs.rules}
        self._versions = {rs.rule_set: rs.version for rs in rule_sets}

    def get(self, rule_id: str) -> Rule:
        try:
            return self._rules[rule_id]
        except KeyError:
            raise UnknownRuleError(f"规则不存在：{rule_id}") from None

    def versions(self) -> dict[str, str]:
        """规则集名 → 版本，写入计划版本的 rule_set_versions。"""
        return dict(self._versions)


def _load_file(path: Path) -> RuleSet:
    try:
        data = load_yaml(path)
    except yaml.YAMLError as e:
        raise RuleLoadError(f"{path}：YAML 格式错误：{e}") from e
    try:
        raw = json.dumps(data)
    except TypeError as e:
        raise RuleLoadError(f"{path}：含有无法按纯文本数据解析的值（如未加引号的日期）：{e}") from e
    try:
        return RuleSet.model_validate_json(raw)
    except ValidationError as e:
        raise RuleLoadError(f"{path}：{e}") from e


def load_rules(rules_dir: Path) -> RuleBook:
    if not rules_dir.is_dir():
        raise RuleLoadError(f"规则目录不存在：{rules_dir}")
    files: list[Path] = []
    for path in sorted(p for p in rules_dir.rglob("*") if p.is_file()):
        if any(part.startswith(".") for part in path.relative_to(rules_dir).parts):
            continue  # 隐藏文件（如 .DS_Store）
        if path.suffix == ".md":
            continue  # 说明文档
        if path.suffix != ".yaml":
            raise RuleLoadError(f"{path}：规则目录中只能有 .yaml 规则文件或 .md 说明")
        files.append(path)
    if not files:
        raise RuleLoadError(f"规则目录里没有任何规则表：{rules_dir}")

    rule_sets: list[RuleSet] = []
    set_files: dict[str, Path] = {}
    rule_files: dict[str, Path] = {}
    for path in files:
        rs = _load_file(path)
        if rs.rule_set != path.stem:
            raise RuleLoadError(f"{path}：规则集名 {rs.rule_set} 与文件名不一致")
        if rs.rule_set in set_files:
            raise RuleLoadError(f"{path}：规则集 {rs.rule_set} 与 {set_files[rs.rule_set]} 重名")
        set_files[rs.rule_set] = path
        for rule in rs.rules:
            if rule.id in rule_files:
                raise RuleLoadError(f"{path}：规则编号 {rule.id} 与 {rule_files[rule.id]} 重复")
            rule_files[rule.id] = path
        rule_sets.append(rs)
    return RuleBook(rule_sets)
