"""数字校验（spec [2026-10-21] §7）：找出回答中的数字，与登记过的数字比对。

登记过的数字来自问答卡、规则表与用户自己的数据。

- 阿拉伯数字（含小数、区间两端）与带量词的中文数字（“三组”“两次”“十二次”）都算。
- 日期（年月日）与序号（第 3 周、第一步）不算建议，不参与校验。
"""

import re

__all__ = ["numbers_in", "unregistered"]

_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_UNITS = "组次个分钟秒小时天周月克千卡公斤斤升毫升杯步层圈公里米"
_ARABIC_RE = re.compile(r"(?<![第\d.])(\d+(?:\.\d+)?)(?![\d.]*\s*[年月日号])")
_CHINESE_RE = re.compile(rf"(?<!第)([零一二两三四五六七八九十百]+)(?=[{_UNITS}])")


def _chinese(text: str) -> int | None:
    """一百以内（含“百”）的中文数字。"""
    if text == "十":
        return 10
    if "百" in text:
        head, _, tail = text.partition("百")
        hundreds = _DIGITS.get(head, 1 if head == "" else -1)
        rest = _chinese(tail) if tail else 0
        return None if hundreds < 0 or rest is None else hundreds * 100 + rest
    if "十" in text:
        head, _, tail = text.partition("十")
        tens = _DIGITS.get(head, 1 if head == "" else -1)
        ones = _DIGITS.get(tail, 0 if tail == "" else -1)
        return None if tens < 0 or ones < 0 else tens * 10 + ones
    if len(text) == 1:
        return _DIGITS.get(text)
    return None


def _norm(value: str) -> str:
    return str(int(float(value))) if float(value).is_integer() else str(float(value))


def numbers_in(text: str) -> set[str]:
    found = {_norm(m.group(1)) for m in _ARABIC_RE.finditer(text)}
    for m in _CHINESE_RE.finditer(text):
        value = _chinese(m.group(1))
        if value is not None:
            found.add(str(value))
    return found


def unregistered(text: str, registered: set[str]) -> set[str]:
    known = {_norm(r) for r in registered}
    return numbers_in(text) - known
