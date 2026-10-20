"""加载规则目录下的全部 *.yaml。任何格式问题都直接报错，应用因此启动失败。"""

import json
from collections.abc import Hashable, Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from ironm.contracts.rules import Rule, RuleParamError, RuleSet

__all__ = ["RULES_DIR", "RuleBook", "RuleLoadError", "RuleParamError", "load_rules"]

RULES_DIR = Path(__file__).resolve().parents[2] / "rules"


class RuleLoadError(ValueError):
    pass


class UnknownRuleError(KeyError):
    pass


class _StrictLoader(yaml.SafeLoader):
    """同一层级出现重复的键时报错（PyYAML 默认静默覆盖）。"""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode) -> dict[Hashable, object]:
    seen: set[Hashable] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"重复的键：{key}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


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
        # _StrictLoader 继承 SafeLoader，不会构造任意 Python 对象
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
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
    for path in sorted(rules_dir.rglob("*.yml")):
        raise RuleLoadError(f"{path}：规则文件扩展名必须是 .yaml")
    files = sorted(rules_dir.rglob("*.yaml"))
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
