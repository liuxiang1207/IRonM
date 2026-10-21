"""冲刺 3 验收（spec [2026-10-21] §1、§9）。

模型全部不可用时的完整对话、运行记录回放、默认健身场景。
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings

NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)

# 建档后提问 → 追问 → 报告不适 → 旁人急症 → 闲聊
CONVERSATION = [
    ("我家里只有哑铃，每周只能练3天", None),
    ("增肌一周练几次比较好", "l0"),
    ("那每个动作做几组几次合适", "l0"),
    ("深蹲时膝盖疼怎么办", "l0"),
    ("跑步时胸口有点闷，停下来就好", "safety"),
    ("看到有人在跑步机上晕倒了", "safety"),
    ("谢谢你", "template"),
]


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})
    # environ 为空：所有模型都拿不到密钥，相当于全部不可用
    app = create_app(settings, FixedClock(NOW), environ={})
    with TestClient(app, headers={"X-IronM-User": "u1"}) as c:
        yield c


def test_full_conversation_without_models(client: TestClient) -> None:
    for text, layer in CONVERSATION:
        resp = client.post("/chat", json={"text": text})
        assert resp.status_code == 200, text
        body = resp.json()
        assert body["sections"] or body["answer"], f"没有回答：{text}"
        if layer is not None:
            assert body["layer"] == layer, text
    # 只有本人的红旗记了标记；旁人的急症没有
    flags = [client.post("/chat", json={"text": t}).json()["flagged"] for t, _ in CONVERSATION[4:6]]
    assert flags == [True, False]


def test_every_answer_can_be_replayed(client: TestClient) -> None:
    ids = [
        client.post("/chat", json={"text": t}).json()["request_id"]
        for t in ("增肌一周练几次比较好", "肌酸要怎么吃")
    ]
    for request_id in ids:
        record = client.get(f"/chat/runs/{request_id}").json()
        assert record["request_id"] == request_id and record["steps"]
        assert "safety_officer" in record["assistants"]


def test_fitness_default_has_no_calorie_deficit(client: TestClient) -> None:
    """没选减脂目标的用户问训练安排，回答里不出现热量缺口。"""
    body = client.post("/chat", json={"text": "增肌一周练几次比较好"}).json()
    text = "".join(p["text"] for p in body["answer"])
    assert "缺口" not in text and "千卡" not in text


def test_emergency_unaffected_when_qa_breaks(monkeypatch: pytest.MonkeyPatch) -> None:
    from ironm.orchestrator import chat
    from tests.test_chat_safety import _deps

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError

    monkeypatch.setattr(chat, "answer_question", broken)
    monkeypatch.setattr(chat, "recognize", broken)
    reply = chat.respond("u1", "我胸口疼，现在还在疼", _deps([]))
    assert reply.level == "emergency" and reply.sections[0].kind == "conclusion"
