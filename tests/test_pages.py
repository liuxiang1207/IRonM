"""功能页提问与带上下文（spec [2026-10-21] §1、§3、§10）。"""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.contracts.roles import AgentId
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.orchestrator.chat import respond
from ironm.orchestrator.pages import PageContext
from ironm.orchestrator.qa import RunRecord
from tests.test_chat_safety import _deps

NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)
SET_CONTEXT = PageContext(page="training", item="today.exercise:3.set:2")


def _records() -> tuple[list[RunRecord], Any]:
    records: list[RunRecord] = []
    deps = _deps([])
    deps = type(deps)(**{**vars(deps), "record_run": lambda u, r: records.append(r)})
    return records, deps


def test_vague_question_with_set_context_needs_no_clarify() -> None:
    records, deps = _records()
    reply = respond("u1", "这个动作我做得怎么样", deps, page="training", page_context=SET_CONTEXT)
    assert reply.layer != "clarify" and reply.page == "training"
    [record] = records
    assert record.primary_domain is AgentId.EXERCISE_COACH and record.page == "training"
    assert record.context_item == "today.exercise:3.set:2"


def test_page_question_only_looks_up_page_domain() -> None:
    records, deps = _records()
    respond("u1", "增肌一周练几次比较好", deps, page="training")
    [record] = records
    # 只查本功能的领域（训练）与安全类的卡
    assert record.primary_domain is AgentId.TRAINING_PLANNER and record.related_domains == ()


def test_chest_pain_with_training_context_still_safety() -> None:
    records, deps = _records()
    reply = respond("u1", "胸口疼", deps, page="training", page_context=SET_CONTEXT)
    assert reply.level == "emergency" and reply.sections[0].kind == "conclusion"
    assert reply.layer == "safety" and records == [] and reply.flagged
    # 不在训练页时仍是红旗：上下文只会提高判定
    assert respond("u1", "胸口疼", _deps([])).level == "red_flag"
    # 非心血管类（如关节）不因训练页升级
    knee = respond("u1", "膝盖肿了，摸着发烫", _deps([]), page="training")
    assert knee.level == "red_flag"


def test_other_page_question_is_handed_off_without_other_agents() -> None:
    records, deps = _records()
    reply = respond("u1", "我这周吃多少蛋白质合适", deps, page="training", page_context=SET_CONTEXT)
    assert reply.layer == "handoff" and reply.handoff == SET_CONTEXT
    assert records == []  # 没有调用营养相关 Agent


def test_plan_change_from_page_is_only_noted() -> None:
    records, deps = _records()
    reply = respond("u1", "帮我把每周改成5天训练计划", deps, page="training")
    assert reply.layer == "plan_note" and records[0].page == "training"


def test_same_question_same_safety_and_numbers_from_both_entries() -> None:
    text = "身高165，体重45公斤，想再减点，饮食怎么吃"
    page_reply = respond("u1", text, _deps([]), page="diet")
    chat_reply = respond("u2", text, _deps([]))
    assert page_reply.level == chat_reply.level
    page_text = "".join(p.text for p in page_reply.answer)
    chat_text = "".join(p.text for p in chat_reply.answer)
    assert "16.5" in page_text and "16.5" in chat_text


def test_context_page_must_match_asked_page() -> None:
    with pytest.raises(ValueError, match="不一致"):
        respond("u1", "怎么做", _deps([]), page="diet", page_context=SET_CONTEXT)


@pytest.mark.parametrize("item", ["今日训练", "Today", "a b", "x" * 81])
def test_context_item_is_an_identifier_not_text(item: str) -> None:
    with pytest.raises(ValueError):
        PageContext(page="training", item=item)


# ---- 接口 ----


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})
    with TestClient(
        create_app(settings, FixedClock(NOW), environ={}), headers={"X-IronM-User": "u1"}
    ) as c:
        yield c


def test_page_ask_endpoint_and_replay_without_user_text(client: TestClient) -> None:
    context = {"page": "training", "item": "today.exercise:3.set:2"}
    body = client.post(
        "/pages/training/ask", json={"text": "这个动作我做得怎么样独特XYZ", "context": context}
    ).json()
    record = client.get(f"/chat/runs/{body['request_id']}")
    assert record.json()["page"] == "training" and "XYZ" not in record.text


def test_page_ask_rejects_unknown_page_and_mismatched_context(client: TestClient) -> None:
    assert client.post("/pages/sleep/ask", json={"text": "你好"}).status_code == 404
    bad = {"text": "怎么做", "context": {"page": "diet"}}
    assert client.post("/pages/training/ask", json=bad).status_code == 422


def test_chat_accepts_context_from_page(client: TestClient) -> None:
    context = {"page": "training", "item": "today.exercise:3.set:2"}
    body = client.post("/chat", json={"text": "这个动作我做得怎么样", "context": context}).json()
    assert body["layer"] != "clarify" and body["intent"] == "exercise_technique"


def test_recording_does_not_run_agents(client: TestClient, tmp_path: Path) -> None:
    """记一条体重只写记录、触发计算，不产生 Agent 运行记录。"""
    from contextlib import closing

    from ironm.store.db import connect
    from tests.test_api_events import _event

    event = _event(user_id="u1")
    resp = client.post("/events", json=event)
    assert resp.status_code in (200, 201)
    with closing(connect(tmp_path / "db.sqlite")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chat_runs").fetchone()[0] == 0
