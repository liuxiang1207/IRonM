"""说的是本人还是旁人（spec [2026-10-21] §3）。只看文字；拿不准时按本人处理。

判法：先去掉不指人的词（“其他”“妈呀”）与“我看到、我叫”这类旁观者的说法，再看：
- 还剩“我、自己、本人”：说到了自己 → 本人；
- 没有说到自己、但有“有人、他、她、我爸、同事”等 → 旁人；
- 都没有 → 拿不准，按本人。
"""

import re
from typing import Literal

from ironm.safety.matching import clauses

type Subject = Literal["self", "other"]

# 不指人的词：先去掉，避免“其他”里的“他”、“妈呀”里的“妈”被当成旁人
_NOT_PEOPLE = ("其他", "其它", "妈呀", "我的妈", "我的天")
# 指旁人的说法；前面可带“我、我的、我们”（“我爸”“我的同事”）
_OTHERS = (
    "有人", "别人", "有个", "一个人", "旁边的", "他", "她", "爸", "妈", "爷爷", "奶奶", "外公",
    "外婆", "姥姥", "姥爷", "老公", "老婆", "丈夫", "妻子", "儿子", "女儿", "孩子", "朋友",
    "同事", "队友", "室友", "同学", "会员", "学员", "教练", "大哥", "大姐", "阿姨", "叔叔",
    "对象", "邻居", "客户",
)  # fmt: skip
_OTHER_RE = re.compile("(?:我的|我们|我)?(?:" + "|".join(map(re.escape, _OTHERS)) + ")")
# 旁观者的说法：“我（今天在健身房）看到”“我叫”“我们”
_WITNESS_RE = re.compile(r"我[^，]{0,8}?(?:看到|看见|发现|听说|叫|喊|扶|陪)|我们")  # 不跨分句
_SELF = ("我", "自己", "本人")


def subject_of(text: str) -> Subject:
    joined = "，".join(clauses(text))
    for word in _NOT_PEOPLE:
        joined = joined.replace(word, "")
    has_other = _OTHER_RE.search(joined) is not None
    rest = _WITNESS_RE.sub("", _OTHER_RE.sub("", joined))
    if any(w in rest for w in _SELF):
        return "self"
    return "other" if has_other else "self"
