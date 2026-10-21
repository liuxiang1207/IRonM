"""对话中的意图识别（spec [2026-10-21] §4）：追问、闲聊、超范围与模型调用记录。"""

import json
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.orchestrator.chat import respond
from ironm.store.db import connect
from ironm.store.safety import ModelCallStore
from tests.test_chat_safety import NOW, _deps


def test_unclear_message_gets_one_clarify_question() -> None:
    reply = respond("u1", "嗯嗯这个那个", _deps([]))
    assert reply.layer == "clarify" and [p.kind for p in reply.answer] == ["clarify"]


def test_short_message_not_clarified() -> None:
    reply = respond("u1", "好", _deps([]))
    assert reply.layer == "template" and reply.intent == "chitchat"


def test_out_of_scope_gets_fixed_answer() -> None:
    reply = respond("u1", "推荐一只股票", _deps([]))
    assert reply.layer == "template" and reply.intent == "out_of_scope"


def test_fitness_question_recognized_with_concern() -> None:
    reply = respond("u1", "深蹲时膝盖疼怎么办", _deps([]))
    assert reply.intent == "exercise_technique" and reply.concerns == ("discomfort",)
    assert reply.layer == "pending"


def test_emergency_skips_intent() -> None:
    reply = respond("u1", "我胸口疼，现在还在疼", _deps([]))
    assert reply.intent is None and reply.answer == ()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    db = tmp_path / "db.sqlite"
    settings = load_settings({"IRONM_DB_PATH": str(db)})
    with TestClient(
        create_app(settings, FixedClock(NOW), environ={}), headers={"X-IronM-User": "u1"}
    ) as c:
        yield c, db


def _chat(c: TestClient, text: str) -> Any:
    return c.post(
        "/chat", content=json.dumps({"text": text}), headers={"Content-Type": "application/json"}
    )


def test_model_unavailable_still_answers_and_is_recorded(client: tuple[TestClient, Path]) -> None:
    c, db = client
    body = _chat(c, "增肌一周练几次比较好").json()
    assert body["intent"] == "training_knowledge"
    with closing(connect(db)) as conn:
        [call] = ModelCallStore(conn).for_request(body["request_id"])
    assert call.status == "error" and call.error_type == "MissingApiKey" and call.fallback_used


def test_emergency_makes_no_model_call(client: tuple[TestClient, Path]) -> None:
    c, db = client
    body = _chat(c, "我胸口疼，现在还在疼").json()
    with closing(connect(db)) as conn:
        assert ModelCallStore(conn).for_request(body["request_id"]) == []
