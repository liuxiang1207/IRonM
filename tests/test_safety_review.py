"""急症判定的模型复核（spec [2026-10-21] §12）：只升不降，类别须在模板内，模型不可用时只用词表。"""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from ironm.contracts.model_call import ModelCall, ModelStage
from ironm.contracts.screening import SafetyFlag
from ironm.core.clock import FixedClock
from ironm.core.ids import SeededIdGenerator
from ironm.llm.config import MODELS_PATH, load_models
from ironm.llm.gateway import Gateway
from ironm.orchestrator.chat import respond
from tests.llm.test_gateway import ENV, FakeTransport, _reply
from tests.test_chat_safety import _deps

NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)
SLUMP = "跑着跑着突然整个人软下去了，半天才缓过来"  # 词表判为“无”
CHEST = "胸口有点闷，停下来就好"  # 词表判为红旗
ONGOING = "我胸口疼，现在还在疼"  # 词表判为急症
INTENT = _reply(json.dumps({"intent": "training_knowledge", "confidence": 0.9, "concerns": []}))


def _review(level: str, categories: list[str]) -> dict[str, object]:
    return _reply(json.dumps({"level": level, "categories": categories}))


def _respond(text: str, *results: object) -> tuple[Any, list[ModelCall], list[SafetyFlag]]:
    calls: list[ModelCall] = []
    flags: list[tuple[str, str, SafetyFlag]] = []
    gateway = Gateway(
        load_models(MODELS_PATH),
        environ=ENV,
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(8),
        transport=FakeTransport(*results, INTENT, INTENT),
        record=calls.append,
    )
    deps = _deps(flags)
    deps = type(deps)(**{**vars(deps), "gateway": gateway})
    reply = respond("u1", text, deps)
    return reply, calls, [f for _, _, f in flags]


def test_review_raises_none_to_emergency_and_flags() -> None:
    reply, calls, flags = _respond(SLUMP, _review("emergency", ["syncope"]))
    assert reply.level == "emergency" and reply.layer == "safety" and reply.flagged
    assert flags[0].rule_ids == ("EM-MODEL-REVIEW",) and flags[0].categories == ("syncope",)
    assert calls[0].stage is ModelStage.SAFETY_REVIEW


def test_review_raises_red_flag_to_emergency_keeping_lexicon_rules() -> None:
    reply, _, flags = _respond(CHEST, _review("emergency", ["cardiovascular"]))
    assert reply.level == "emergency"
    assert "RF-CHEST" in flags[0].rule_ids and "EM-MODEL-REVIEW" in flags[0].rule_ids


def test_review_cannot_lower() -> None:
    reply, _, _ = _respond(CHEST, _review("none", []))
    assert reply.level == "red_flag"


def test_lexicon_emergency_skips_review() -> None:
    reply, calls, _ = _respond(ONGOING)
    assert reply.level == "emergency" and calls == []


@pytest.mark.parametrize(
    "review",
    [
        _review("emergency", ["made_up_category"]),  # 不在模板里
        _review("emergency", []),  # 没给类别
        _review("emergency", ["syncope", "made_up_category"]),  # 部分不在模板里
        _reply("不是 JSON"),  # 输出不合规
        ConnectionError(),  # 模型不可用（重试后仍失败）
    ],
)
def test_unusable_review_keeps_lexicon_result(review: object) -> None:
    results = (review, ConnectionError()) if isinstance(review, Exception) else (review,)
    reply, _, flags = _respond(SLUMP, *results)
    assert reply.level == "none" and flags == []


def test_no_gateway_means_lexicon_only() -> None:
    assert respond("u1", SLUMP, _deps([])).level == "none"
