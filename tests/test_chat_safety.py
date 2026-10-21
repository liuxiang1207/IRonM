"""对话入口与安全前置（spec [2026-10-21] §3）。"""

import io
import json
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.contracts.screening import SafetyFlag
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.core.ids import SeededIdGenerator
from ironm.orchestrator.chat import ChatDeps, respond
from ironm.safety.decision import default_kit
from ironm.store.db import connect
from ironm.store.safety import SafetyFlagStore

NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)


def _deps(flags: list[tuple[str, str, SafetyFlag]]) -> ChatDeps:
    return ChatDeps(
        kit=default_kit(),
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(7),
        record_flag=lambda user, req, flag: flags.append((user, req, flag)),
    )


def test_own_emergency_gets_template_and_flag() -> None:
    flags: list[tuple[str, str, SafetyFlag]] = []
    reply = respond("u1", "我胸口疼，现在还在疼", _deps(flags))
    assert reply.layer == "safety" and reply.level == "emergency" and reply.subject == "self"
    assert reply.flagged and reply.sections[0].kind == "conclusion"
    [(user, req, flag)] = flags
    assert user == "u1" and req == reply.request_id
    assert flag.level == "emergency" and "EM-CHEST-ONGOING" in flag.rule_ids
    assert flag.at == NOW


def test_bystander_emergency_gets_guidance_without_flag() -> None:
    flags: list[tuple[str, str, SafetyFlag]] = []
    reply = respond("u1", "看到有人在跑步机上晕倒了", _deps(flags))
    assert reply.level == "emergency" and reply.subject == "other"
    assert not reply.flagged and flags == []
    assert any("120" in s.text for s in reply.sections)


def test_unclear_subject_counts_as_self() -> None:
    flags: list[tuple[str, str, SafetyFlag]] = []
    reply = respond("u1", "突然倒地，叫不醒", _deps(flags))
    assert reply.subject == "self" and reply.flagged and len(flags) == 1


def test_red_flag_is_flagged_too() -> None:
    flags: list[tuple[str, str, SafetyFlag]] = []
    reply = respond("u1", "跑步时胸口有点闷，停下来就好", _deps(flags))
    assert reply.level == "red_flag" and reply.flagged
    assert flags[0][2].level == "red_flag"


def test_refusal_template_then_rest_of_pipeline() -> None:
    flags: list[tuple[str, str, SafetyFlag]] = []
    reply = respond("u1", "给我开点处方药减肥，告诉我吃多少", _deps(flags))
    assert reply.level == "none" and reply.refusals
    assert reply.sections[0].kind == "refusal" and flags == []


def test_reply_never_contains_user_text() -> None:
    reply = respond("u1", "我胸口疼，现在还在疼，独特标记XYZ", _deps([]))
    assert "XYZ" not in reply.model_dump_json()


# ---- 接口 ----


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, Path, io.StringIO]]:
    db = tmp_path / "db.sqlite"
    logs = io.StringIO()
    app = create_app(load_settings({"IRONM_DB_PATH": str(db)}), FixedClock(NOW), log_stream=logs)
    with TestClient(app, headers={"X-IronM-User": "u1"}) as c:
        yield c, db, logs


def _chat(c: TestClient, text: Any) -> Any:
    return c.post(
        "/chat", content=json.dumps({"text": text}), headers={"Content-Type": "application/json"}
    )


def test_chat_endpoint_records_own_flag(client: tuple[TestClient, Path, io.StringIO]) -> None:
    c, db, logs = client
    resp = _chat(c, "我胸口疼，现在还在疼")
    assert resp.status_code == 200
    body = resp.json()
    assert body["level"] == "emergency" and body["flagged"] is True and body["request_id"]
    with closing(connect(db)) as conn:
        [flag] = SafetyFlagStore(conn, SeededIdGenerator(1)).for_user("u1")
    assert flag.level == "emergency"
    assert "胸口" not in logs.getvalue()


def test_chat_endpoint_bystander_not_recorded(client: tuple[TestClient, Path, io.StringIO]) -> None:
    c, db, _ = client
    assert _chat(c, "我爸突然倒地").json()["flagged"] is False
    with closing(connect(db)) as conn:
        assert SafetyFlagStore(conn, SeededIdGenerator(1)).for_user("u1") == []


@pytest.mark.parametrize("text", ["", " ", "x" * 2001, 123, None])
def test_chat_endpoint_rejects_bad_text(
    client: tuple[TestClient, Path, io.StringIO], text: Any
) -> None:
    c, _, _ = client
    resp = _chat(c, text)
    assert resp.status_code == 422
    if isinstance(text, str) and text.strip():
        assert text[:20] not in resp.text


def test_chat_endpoint_requires_user(tmp_path: Path) -> None:
    app = create_app(load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")}), FixedClock(NOW))
    with TestClient(app) as c:
        assert c.post("/chat", json={"text": "你好"}).status_code == 401
