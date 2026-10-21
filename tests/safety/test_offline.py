"""App 离线版急症判定（spec [2026-10-20] §22）：与服务端逐句比对。

离线脚本是 TypeScript（app/src/safety/emergency.ts），用 Node 22.18 以上直接运行。
"""

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from ironm.safety.decision import load_kit
from ironm.safety.emergency import TERMS_PATH, assess, load_terms
from ironm.safety.matching import PUNCTUATION, clauses
from ironm.safety.offline import OFFLINE_PATH, render
from ironm.safety.replies import four_section
from tests.safety.test_emergency import (
    BLOOD_PRESSURE,
    NEGATIONS,
    ORDINARY,
    PARAPHRASES,
    TRADEOFFS,
    UNIFIED,
)

ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run_offline.mjs")


def test_offline_data_up_to_date() -> None:
    assert OFFLINE_PATH.exists(), "缺少 App 离线词表，请运行 make offline"
    assert OFFLINE_PATH.read_text(encoding="utf-8") == render(load_kit()), (
        "App 离线数据与急症词表或回答模板不一致，请运行 make offline"
    )


def _random_texts(n: int) -> list[str]:
    """把词表里的说法、否定词、“过”、分句符号、血压数字等随机拼接，覆盖各条分支。"""
    book = load_terms(TERMS_PATH)
    neg = book.negation
    words = [t for s in book.symptoms for t in s.terms]
    words += [t for s in book.symptoms for c in s.combos for g in c for t in g]
    pieces = [
        *words,
        *neg.before,
        *neg.exceptions,
        *neg.recurrence,
        *neg.recent,
        *neg.body_parts,
        *neg.remote,
        neg.past_after,
        *neg.past_not_followed_by,
        *"，。！？；、,.!? \n～~\uff61\ufe50\ufe52\ufe54\ufe56\ufe57",
        *["我", "今天", "跑步", "有点", "很", "了", "的", "也", "但是", "ＭＭＨＧ", "Ｂ"],
        *["血压", "高压", "低压", "190/120", "１８０／９０", "179/119", "1900/80", "12", "/"],
    ]
    spaces = [" ", "\u3000", "\u00a0", "\u200b", "\t"]
    fillers = ["", "我", "但", "也", "还是", "今天", "，"]
    rng = random.Random(20261020)

    def spaced(word: str) -> str:  # 说法中间插入各种空白
        return "".join(ch + (rng.choice(spaces) if rng.random() < 0.3 else "") for ch in word)

    def digits() -> str:
        return str(rng.randint(1, 3000))

    texts = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.15:  # 同一说法出现两处，前一处被否定
            w = rng.choice(words)
            texts.append(f"{rng.choice(neg.before)}{w}{rng.choice(fillers)}{w}")
        elif kind < 0.3:  # 血压：各种位数
            ctx = rng.choice(["血压", "高压", "低压", "mmhg", "ＭＭＨＧ", ""])
            texts.append(f"{ctx}{digits()}/{digits()}{rng.choice(['', '，头痛', '头晕'])}")
        elif kind < 0.4:
            texts.append(spaced(rng.choice(words)) + rng.choice(fillers) + rng.choice(words))
        else:
            texts.append("".join(rng.choice(pieces) for _ in range(rng.randint(1, 10))))
    return texts


def _cases() -> list[str]:
    red_flags = yaml.safe_load((ROOT / "tests/safety/red_flags.yaml").read_text("utf-8"))
    texts = [c["text"] for c in red_flags["cases"]]
    texts += [t for t, _ in PARAPHRASES] + ORDINARY + [t for t, _ in NEGATIONS] + UNIFIED
    texts += [t for t, _ in BLOOD_PRESSURE] + [t for t, _ in TRADEOFFS]
    texts += [f"胸口痛{chr(c)}还在疼" for c in PUNCTUATION]  # 每个标点映射两边都要一致
    return texts + _random_texts(2000)


def _offline(texts: list[str]) -> list[dict[str, object]]:
    node = shutil.which("node")
    if node is None:
        pytest.fail("需要 Node.js 22.18 以上才能检查 App 离线版")
    payload = json.dumps({"book": str(OFFLINE_PATH), "texts": texts}, ensure_ascii=False)
    done = subprocess.run(
        [node, str(RUNNER)], input=payload, capture_output=True, text=True, check=True
    )
    result: list[dict[str, object]] = json.loads(done.stdout)
    return result


def test_offline_matches_server() -> None:
    texts = _cases()
    offline = _offline(texts)
    replies = load_kit().replies
    server: list[dict[str, object]] = []
    for t in texts:
        a = assess(t)
        sections = [s.model_dump(mode="json") for s in four_section(replies, a)]
        server.append(
            {"assessment": a.model_dump(mode="json"), "sections": sections, "clauses": clauses(t)}
        )
    mismatches = [(t, s, o) for t, s, o in zip(texts, server, offline, strict=True) if s != o]
    assert not mismatches, f"共 {len(mismatches)} 处不一致，例如：{mismatches[:3]}"
    # 随机句子确实覆盖到三种级别，比对才有意义
    levels = {assess(t).level for t in texts}
    assert levels == {"emergency", "red_flag", "none"}


def test_offline_merges_like_server_when_categories_share_text(tmp_path: Path) -> None:
    """两个类别的停止文字、科室、可带信息相同时，两边都去重（真实模板里暂未出现，单独构造）。"""
    data = json.loads(OFFLINE_PATH.read_text(encoding="utf-8"))
    cats = data["replies"]["categories"]
    cats["syncope"] = dict(cats["cardiovascular"])
    book = tmp_path / "book.json"
    book.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    replies = load_kit().replies
    shared = replies.categories["cardiovascular"]
    replies = replies.model_copy(update={"categories": {**replies.categories, "syncope": shared}})

    text = "我爸刚才跑步时说胸痛，然后就倒下了"
    a = assess(text)
    assert set(a.categories) == {"cardiovascular", "syncope"}
    expected = [s.model_dump(mode="json") for s in four_section(replies, a)]
    node = shutil.which("node")
    assert node is not None
    payload = json.dumps({"book": str(book), "texts": [text]}, ensure_ascii=False)
    done = subprocess.run(
        [node, str(RUNNER)], input=payload, capture_output=True, text=True, check=True
    )
    [offline] = json.loads(done.stdout)
    assert offline["sections"] == expected
    assert expected[1]["text"].count(shared.stop) == 1
