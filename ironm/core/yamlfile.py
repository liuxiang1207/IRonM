"""严格读取 YAML：重复的键报错，只认 YAML 1.2 的数字与真假值写法。规则表、词表、测试数据共用。"""

import re
from collections.abc import Hashable
from pathlib import Path

import yaml

__all__ = ["load_yaml", "load_yaml_text"]


class _StrictLoader(yaml.SafeLoader):
    """同一层级出现重复的键时报错（PyYAML 默认静默覆盖）；数字与真假值只认现行标准写法。"""


# PyYAML 沿用 YAML 1.1：1:30 读成 90、010 读成 8、yes / on 读成真，都会悄悄改值。
# 换成 YAML 1.2 核心写法，旧式写法一律当文字，再由类型校验报错。
_BOOL, _INT, _FLOAT = (f"tag:yaml.org,2002:{t}" for t in ("bool", "int", "float"))
_StrictLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag not in {_BOOL, _INT, _FLOAT}]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictLoader.add_implicit_resolver(_BOOL, re.compile(r"^(?:true|false)$"), list("tf"))
_StrictLoader.add_implicit_resolver(
    _INT, re.compile(r"^[-+]?(?:0|[1-9][0-9]*)$"), list("-+0123456789")
)
_StrictLoader.add_implicit_resolver(
    _FLOAT,
    re.compile(
        r"^[-+]?(?:(?:0|[1-9][0-9]*)\.[0-9]+(?:[eE][-+]?[0-9]+)?|(?:0|[1-9][0-9]*)[eE][-+]?[0-9]+)$"
    ),
    list("-+0123456789"),
)


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


def load_yaml_text(text: str) -> object:
    """从文字读取 YAML（规则同 load_yaml）。"""
    return yaml.load(text, Loader=_StrictLoader)


def load_yaml(path: Path) -> object:
    """读取 YAML；同一层级出现重复的键时报错。"""
    # _StrictLoader 继承 SafeLoader，不会构造任意 Python 对象
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
