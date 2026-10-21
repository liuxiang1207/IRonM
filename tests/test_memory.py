"""对话记忆（spec [2026-10-21] §8）：上下文、压缩、画像候选与确认。"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.contracts.screening import Screening, Tier
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.core.ids import SeededIdGenerator
from ironm.llm.config import MODELS_PATH, load_models
from ironm.llm.gateway import Gateway
from ironm.memory.conversation import compress, context
from ironm.memory.profile import ProfileService, load_terms
from ironm.orchestrator.chat import after_reply, respond
from ironm.orchestrator.first_plan import SCREENING_FIELDS
from ironm.rules.loader import RULES_DIR, load_rules
from ironm.safety.tiers import classify
from ironm.store.memory import MemoryStore
from sim.first_plan_demo import PERSONAS
from tests.llm.test_gateway import ENV, FakeTransport, _reply
from tests.memory_helpers import memory_db
from tests.test_chat_safety import NOW, _deps

RULES = load_rules(RULES_DIR)
GYM = next(iter(PERSONAS.values()))


def _say(memory: MemoryStore, text: str, user: str = "u1", gateway: Gateway | None = None) -> Any:
    deps = _deps([], memory)
    if gateway is not None:
        deps = type(deps)(**{**vars(deps), "gateway": gateway})
    reply = respond(user, text, deps)
    after_reply(user, text, reply, deps, now=FixedClock(NOW).now)
    return reply


def _tier(profile: dict[str, Any]) -> Tier | object:
    data = {k: profile[k] for k in SCREENING_FIELDS if k in profile}
    screening = Screening.model_validate_json(json.dumps(data))
    return classify(screening, as_of=NOW, tz=ZoneInfo("Asia/Shanghai"), rulebook=RULES).tier


# ---- 上下文与压缩 ----


def test_turns_are_saved_and_read_back_as_context() -> None:
    memory = memory_db()
    _say(memory, "肌酸要怎么吃")
    ctx = context(memory, "u1")
    assert [m.role for m in ctx.recent] == ["user", "assistant"]
    assert ctx.recent[0].text == "肌酸要怎么吃" and ctx.recent[0].intent == "supplements_drugs"
    assert ctx.summary is None


def test_compressed_at_fifteen_messages_keeping_recent_raw() -> None:
    memory = memory_db()
    for i in range(7):
        _say(memory, f"肌酸要怎么吃{i}")  # 每轮 2 条：14 条，还不压缩
    assert memory.latest_summary("u1") is None
    _say(memory, "一天要吃多少蛋白质")  # 第 15、16 条
    summary = memory.latest_summary("u1")
    assert summary is not None and summary.source == "rule"
    assert "补剂与药物" in summary.text and "肌酸" not in summary.text  # 兜底摘要不含原话
    ctx = context(memory, "u1")
    assert len(ctx.recent) == 6 and ctx.recent[-2].text == "一天要吃多少蛋白质"


def test_model_summary_used_when_available() -> None:
    memory = memory_db()
    for i in range(7):
        _say(memory, f"肌酸要怎么吃{i}")
    gateway = Gateway(
        load_models(MODELS_PATH),
        environ=ENV,
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(9),
        transport=FakeTransport(_reply(json.dumps({"summary": "用户在问肌酸的用法。"}))),
    )
    assert (
        compress(memory, "u1", rulebook=RULES, gateway=None, request_id="r", now=lambda: NOW)
        is False
    )
    memory.add_message("u1", "r", "user", "再问一句", None, NOW)
    assert compress(memory, "u1", rulebook=RULES, gateway=gateway, request_id="r", now=lambda: NOW)
    summary = memory.latest_summary("u1")
    assert (
        summary is not None and summary.source == "model" and summary.text == "用户在问肌酸的用法。"
    )


def test_storage_failure_after_reply_does_not_raise() -> None:
    memory = memory_db()
    deps = _deps([], memory)
    reply = respond("u1", "肌酸要怎么吃", deps)
    memory._conn.close()  # 模拟数据库不可用
    after_reply("u1", "肌酸要怎么吃", reply, deps, now=FixedClock(NOW).now)


# ---- 画像候选 ----


def _service(memory: MemoryStore) -> ProfileService:
    return ProfileService(memory, terms=load_terms(), ids=SeededIdGenerator(2), now=lambda: NOW)


def test_training_condition_is_candidate_until_confirmed() -> None:
    memory = memory_db()
    reply = _say(memory, "我家里只有哑铃，深蹲怎么练")
    [cand] = reply.candidates
    assert cand.field == "environment" and cand.value == "home" and not cand.immediate
    assert memory.profile("u1") == (0, {})  # 确认前画像不变
    assert reply.answer[-1].kind == "notice"
    _service(memory).decide("u1", cand.candidate_id, accept=True)
    assert memory.profile("u1")[1] == {"environment": "home"}


def test_rejected_candidate_never_reaches_profile() -> None:
    memory = memory_db()
    [cand] = _say(memory, "每周只能练3天").candidates
    _service(memory).decide("u1", cand.candidate_id, accept=False)
    assert memory.profile("u1") == (0, {})
    assert memory.candidates("u1", "pending") == []


def test_new_heart_disease_makes_tier_stricter_immediately() -> None:
    memory = memory_db()
    memory.save_profile("u1", 0, dict(GYM), "initial", NOW)
    assert _tier(memory.profile("u1")[1]) is Tier.A
    reply = _say(memory, "我刚查出心脏病，还能练吗")
    [cand] = reply.candidates
    assert cand.immediate and cand.value == "heart_disease"
    profile = memory.profile("u1")[1]
    assert "heart_disease" in profile["conditions"] and _tier(profile) is Tier.C
    assert "special_population" in reply.concerns  # 本次回答已按特殊人群处理
    # 用户否认：撤回
    _service(memory).decide("u1", cand.candidate_id, accept=False)
    assert _tier(memory.profile("u1")[1]) is Tier.A


def test_profile_conditions_make_later_answers_cautious() -> None:
    memory = memory_db()
    memory.save_profile("u1", 0, {"conditions": ["pregnant"]}, "initial", NOW)
    reply = _say(memory, "一周练几次比较好")
    assert "special_population" in reply.concerns


def test_other_persons_condition_not_recorded() -> None:
    memory = memory_db()
    assert _say(memory, "我爸有心脏病，他能跑步吗").candidates == ()
    assert memory.profile("u1") == (0, {})


def test_repeated_statement_not_duplicated() -> None:
    memory = memory_db()
    _say(memory, "我有高血压")
    assert _say(memory, "我有高血压，能举重吗").candidates == ()


def test_profile_failure_still_answers() -> None:
    memory = memory_db()
    deps = _deps([], memory)
    broken = type(deps.profiles)(
        memory, terms=load_terms(), ids=SeededIdGenerator(1), now=lambda: NOW
    )
    broken.note = lambda *a: (_ for _ in ()).throw(RuntimeError())  # type: ignore[method-assign]
    reply = respond("u1", "肌酸要怎么吃", type(deps)(**{**vars(deps), "profiles": broken}))
    assert reply.layer == "l0" and reply.candidates == ()


def test_decide_only_own_pending_candidates() -> None:
    memory = memory_db()
    [cand] = _say(memory, "每周只能练3天").candidates
    service = _service(memory)
    assert service.decide("u2", cand.candidate_id, accept=True) is None
    assert service.decide("u1", "missing", accept=True) is None
    assert service.decide("u1", cand.candidate_id, accept=True) is not None
    assert service.decide("u1", cand.candidate_id, accept=False) is None  # 已处理过


def test_concurrent_profile_write_rejected() -> None:
    memory = memory_db()
    memory.save_profile("u1", 0, {"a": 1}, "initial", NOW)
    with pytest.raises(ValueError, match="已被更新"):
        memory.save_profile("u1", 0, {"a": 2}, "late", NOW)


# ---- 接口 ----


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})
    with TestClient(
        create_app(settings, FixedClock(NOW), environ={}), headers={"X-IronM-User": "u1"}
    ) as c:
        yield c


def test_api_candidate_flow(client: TestClient) -> None:
    body = client.post("/chat", json={"text": "我家里只有哑铃"}).json()
    cand_id = body["candidates"][0]["candidate_id"]
    pending = client.get("/profile/candidates").json()["candidates"]
    assert [c["candidate_id"] for c in pending] == [cand_id]
    assert client.get("/profile").json() == {"version": 0, "data": {}}
    assert client.post(f"/profile/candidates/{cand_id}/confirm").json()["status"] == "confirmed"
    assert client.get("/profile").json()["data"] == {"environment": "home"}
    assert client.post(f"/profile/candidates/{cand_id}/reject").status_code == 404
    assert client.post(f"/profile/candidates/{cand_id}/maybe").status_code == 404


def test_api_saves_turns_in_background(client: TestClient, tmp_path: Path) -> None:
    client.post("/chat", json={"text": "肌酸要怎么吃"})
    from contextlib import closing

    from ironm.store.db import connect

    with closing(connect(tmp_path / "db.sqlite")) as conn:
        assert [m.role for m in MemoryStore(conn).messages_after("u1", 0)] == ["user", "assistant"]
