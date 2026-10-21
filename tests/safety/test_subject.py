"""本人与旁人（spec [2026-10-21] §3）：旁人的急症只给急救指引；拿不准时按本人处理。"""

import pytest

from ironm.safety.subject import subject_of

OTHER = [
    "看到有人在跑步机上晕倒了",
    "我爸突然倒地",
    "他突然倒了，叫他也不答应",
    "我今天在健身房看到有人晕倒了，好吓人",
    "我妈有糖尿病，现在手抖出冷汗",
    "我同事练完腿在更衣室倒了，叫了半天没反应",
    "队友现在胡言乱语，身上发烫",
    "我叫他他没反应，人躺在地上",
]

SELF = [
    "我胸口疼，现在还在疼",
    "突然倒了，叫不醒",  # 没说是谁：按本人处理
    "胸口疼，出冷汗",
    "我胸口疼，我爸说让我去医院",  # 同时说到自己：按本人处理
    "教练让我深蹲，我膝盖疼得不能着地",
    "其他时候胸口不疼，今天跑步时疼",  # “其他”不是旁人
    "妈呀胸口好疼",  # 语气词不是旁人
    "他说是肌肉问题，但我还是胸口疼",
    "我胸口疼，叫我老公来了",  # “叫”不跨分句去掉前面的“我”
]


@pytest.mark.parametrize("text", OTHER)
def test_other_person(text: str) -> None:
    assert subject_of(text) == "other"


@pytest.mark.parametrize("text", SELF)
def test_self_or_unclear_counts_as_self(text: str) -> None:
    assert subject_of(text) == "self"
