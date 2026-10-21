"""数字校验（spec [2026-10-21] §7）：找出回答中的数字，与登记过的数字比对。

登记过的数字来自问答卡、规则表与用户自己的数据。

- 阿拉伯数字（含小数、区间两端）与带量词的中文数字（“三组”“两次”“十二次”“一千五百千卡”）都算；
  区间（“三到五组”）与约数（“两三组”）的每个数都算；“百分之八十”算 80。
- 日期（年月日）与序号（第 3 周、第一步）不算建议，不参与校验；“7日”只有跟在“月”后面才是日期。
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
# 量词（按词匹配，长的在前）：中文数字后面跟着这些词才算数字，避免“一下子”“一定”被当成 1
_UNIT_WORDS = sorted(
    [
        *("组", "次", "个", "分钟", "秒", "小时", "天", "日", "周", "月", "岁", "倍"),
        *("克", "千卡", "大卡", "公斤", "千克", "斤", "升", "毫升", "杯"),
        *("步", "层", "圈", "公里", "米", "餐", "顿", "碗", "片", "粒", "份", "遍", "趟", "成"),
    ],
    key=len,
    reverse=True,
)
_UNIT = "(?:" + "|".join(_UNIT_WORDS) + ")"
_CN = "零一二两三四五六七八九十百千万"
_RANGE = r"\s*(?:到|至|~|～|-|—|–|或)\s*"
_ABOUT = "(?:几|来|多)?"  # 约数的尾巴：十几组、二十来次、三十多岁，记前面的整数
_PERCENT_RE = re.compile(rf"百分之([{_CN}]+|\d+(?:\.\d+)?)")
_ARABIC_RE = re.compile(r"(?<![第\d.])(\d+(?:\.\d+)?)(?![\d.]*\s*[年月号])")
# 中文数字（可带“点”）：后面跟量词，或是区间的前一半（“三到五组”）；前面不能是“第”或另一个数字。
# 取最短的一段：“三千克”是 3 千克，不是 3000 克
_CHINESE_RE = re.compile(
    rf"(?<![第{_CN}点\d])([{_CN}点]+?){_ABOUT}"
    rf"(?={_UNIT}|{_RANGE}[{_CN}点\d]+{_ABOUT}\s*{_UNIT})"
)
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")  # 1,500 → 1500
_RUN_SCALE_RE = re.compile(r"([一二两三四五六七八九]{2,})([十百千万])")
_SCALE = {"十": 10, "百": 100, "千": 1000}


def _below_wan(text: str) -> int | None:
    """一万以内：一千五百、三百零五、十二、两百。"""
    total, digit, last_scale = 0, None, 0
    for ch in text:
        if ch == "零":
            digit = None
        elif ch in _DIGITS:
            if digit is not None:
                return None  # 两个数字连写（如“两三”）不是一个数
            digit = _DIGITS[ch]
        elif ch in _SCALE:
            total += (1 if digit is None else digit) * _SCALE[ch]
            digit, last_scale = None, _SCALE[ch]
        else:
            return None
    if digit is not None and len(text) > 1 and text[-2] in _SCALE and last_scale >= 100:
        return total + digit * last_scale // 10  # 口语省略末位：一千五 = 1500，一百五 = 150
    return total + (digit or 0)


def _chinese(text: str) -> int | None:
    if "万" in text:
        head, _, tail = text.partition("万")
        high = _below_wan(head) if head else 1
        # 两万三 = 23000（口语省略“千”）
        low = _DIGITS[tail] * 1000 if tail in _DIGITS else (_below_wan(tail) if tail else 0)
        return None if high is None or low is None else high * 10000 + low
    return _below_wan(text)


def _chinese_numbers(token: str) -> set[str]:
    if token[0] in "百千万点" or token.endswith("点"):
        return set()  # “每千克”“千卡”里的千是单位的一部分，不是数字
    if "点" in token:
        whole, _, frac = token.partition("点")
        value = _chinese(whole)
        if value is None or not frac or any(ch not in _DIGITS for ch in frac):
            return set()
        return {_norm(f"{value}.{''.join(str(_DIGITS[ch]) for ch in frac)}")}
    if m := _RUN_SCALE_RE.fullmatch(token):
        # 三四十分钟、一两千卡：约数，每个都算
        scale = 10000 if m.group(2) == "万" else _SCALE[m.group(2)]
        return {str(_DIGITS[ch] * scale) for ch in m.group(1)}
    value = _chinese(token)
    if value is not None:
        return {str(value)}
    if all(ch in _DIGITS and ch != "零" for ch in token):
        return {str(_DIGITS[ch]) for ch in token}  # “两三组”“三五次”：约数，每个都算
    return set()


def _norm(value: str) -> str:
    return str(int(float(value))) if float(value).is_integer() else str(float(value))


def _is_date_day(text: str, start: int, end: int) -> bool:
    """“12日”只有跟在“月”后面（3月12日）才是日期；“每周练7日”是数字。"""
    return text[end:].lstrip().startswith("日") and text[:start].endswith("月")


def numbers_in(text: str) -> set[str]:
    text = _THOUSANDS_RE.sub("", text)
    found: set[str] = set()
    for m in _PERCENT_RE.finditer(text):
        raw = m.group(1)
        found |= {_norm(raw)} if raw[0].isdigit() else _chinese_numbers(raw)
    for m in _ARABIC_RE.finditer(text):
        if not _is_date_day(text, m.start(1), m.end(1)):
            found.add(_norm(m.group(1)))
    for m in _CHINESE_RE.finditer(text):
        values = _chinese_numbers(m.group(1))
        if text.startswith("千卡", m.end()) and all(float(v) < 10 for v in values):
            # 口语：“两千卡”是 2000 千卡，而不是 2 千卡
            values = {str(int(float(v) * 1000)) for v in values}
        found |= values
    return found


def unregistered(text: str, registered: set[str]) -> set[str]:
    known = {_norm(r) for r in registered}
    return numbers_in(text) - known
