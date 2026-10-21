"""问答：查知识与回答（spec [2026-10-21] §6、§7、§12）。"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ironm.api.app import create_app
from ironm.contracts.model_call import ModelCall
from ironm.contracts.roles import AgentId
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.core.ids import SeededIdGenerator
from ironm.knowledge.qa_cards import default_cards
from ironm.llm.config import MODELS_PATH, load_models
from ironm.llm.gateway import Gateway
from ironm.orchestrator.chat import load_answer_texts
from ironm.orchestrator.intent import Intent, IntentResult
from ironm.orchestrator.qa import answer_question
from tests.llm.test_gateway import ENV, FakeTransport, _reply

CARDS = default_cards()
TEXTS = load_answer_texts()
NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)


def _intent(intent: Intent, *concerns: str) -> IntentResult:
    return IntentResult.model_validate(
        {
            "intent": intent,
            "concerns": concerns,
            "confidence": 0.9,
            "source": "rule",
            "clarify": False,
        }
    )


def _gateway(*results: object) -> tuple[Gateway, list[ModelCall]]:
    gw, calls, _ = _gateway_with_transport(*results)
    return gw, calls


def _gateway_with_transport(*results: object) -> tuple[Gateway, list[ModelCall], FakeTransport]:
    calls: list[ModelCall] = []
    transport = FakeTransport(*results)
    gw = Gateway(
        load_models(MODELS_PATH),
        environ=ENV,
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(5),
        transport=transport,
        record=calls.append,
    )
    return gw, calls, transport


def _ask(text: str, intent: IntentResult, gateway: Gateway | None = None, **kw: Any) -> Any:
    return answer_question(
        text, intent, cards=CARDS, texts=TEXTS, gateway=gateway, request_id="req-1", **kw
    )


def _l1(**parts: object) -> dict[str, object]:
    return _reply(json.dumps({"conclusion": "先减轻重量。", **parts}, ensure_ascii=False))


KNEE = "深蹲时膝盖疼怎么办"
KNEE_INTENT = _intent(Intent.EXERCISE_TECHNIQUE, "discomfort")


# ---- 查知识（spec [2026-10-21] §12）----


def test_knee_pain_looks_up_technique_cards_and_adds_caution() -> None:
    record = _ask(KNEE, KNEE_INTENT).record
    assert record.primary_domain is AgentId.EXERCISE_COACH
    assert record.matched_cards[0] == "QA-SQUAT-KNEE-PAIN"
    assert record.concerns == ("discomfort",)


def test_safety_cards_always_looked_up_and_first() -> None:
    """意图是训练知识，但“两周后婚礼一天只吃一顿”命中安全类的卡：安全卡排在最前并直接使用。"""
    record = _ask("两周后婚礼，怎么一天只吃一顿快速掉秤", _intent(Intent.TRAINING_KNOWLEDGE)).record
    assert record.matched_cards[0] == "QA-RAPID-LOSS" and record.card_ids == ("QA-RAPID-LOSS",)
    assert record.layer == "l0" and not record.degraded


def test_cross_domain_question_finds_both_domains() -> None:
    intent = IntentResult.model_validate(
        {
            "intent": Intent.TRAINING_KNOWLEDGE,
            "concerns": (),
            "confidence": 0.85,
            "source": "rule",
            "clarify": False,
            "domains": (Intent.TRAINING_KNOWLEDGE, Intent.NUTRITION),
        }
    )
    record = _ask("增肌一周练几次，蛋白质一天吃多少", intent).record
    assert record.primary_domain is AgentId.TRAINING_PLANNER
    assert AgentId.NUTRITIONIST in record.related_domains
    assert {"QA-STRENGTH-FREQUENCY", "QA-PROTEIN"} <= set(record.matched_cards)


def test_unrelated_domain_cards_not_used() -> None:
    """问动作时不会查到饮食领域的卡（“蛋白”只在饮食卡里）。"""
    record = _ask("俯卧撑标准吗，吃蛋白多少", _intent(Intent.EXERCISE_TECHNIQUE)).record
    assert "QA-PROTEIN" not in record.matched_cards


def test_page_only_looks_up_its_domain_and_safety() -> None:
    record = _ask(
        "增肌一周练几次，蛋白质一天吃多少",
        _intent(Intent.TRAINING_KNOWLEDGE),
        page="training",
        page_domain=AgentId.TRAINING_PLANNER,
    ).record
    assert record.matched_cards == ("QA-STRENGTH-FREQUENCY",) and record.related_domains == ()


# ---- 回答分层 ----


def test_card_hit_uses_l0_without_model() -> None:
    gw, calls = _gateway()
    result = _ask("肌酸要怎么吃", _intent(Intent.SUPPLEMENTS_DRUGS), gw)
    assert result.record.layer == "l0" and not result.record.degraded and calls == []
    assert [p.kind for p in result.parts] == ["conclusion", "why", "how", "caution"]


def test_plan_change_is_explained_not_applied() -> None:
    gw, calls = _gateway()
    result = _ask("帮我把每周训练改成5天", _intent(Intent.PLAN), gw)
    assert result.record.layer == "plan_note" and calls == []
    assert result.parts[0].text == TEXTS.plan_change


def test_caution_goes_through_l1_with_structure() -> None:
    gw, calls = _gateway(_l1(how="改用自重深蹲。", caution="疼痛明显时先停下并咨询医生。"))
    result = _ask(KNEE, KNEE_INTENT, gw)
    assert result.record.layer == "l1" and [c.status for c in calls] == ["ok"]
    assert [p.kind for p in result.parts] == ["conclusion", "how", "caution"]
    assert result.record.card_ids[0] == "QA-SQUAT-KNEE-PAIN"


def test_unregistered_number_rewritten_once() -> None:
    gw, calls, transport = _gateway_with_transport(
        _l1(how="每组做 15 次。"), _l1(how="改用自重深蹲。")
    )
    result = _ask(KNEE, KNEE_INTENT, gw)
    assert result.record.layer == "l1" and result.record.number_rewrites == 1
    assert "15" not in "".join(p.text for p in result.parts) and len(calls) == 2
    first, second = (r[2]["messages"][0]["content"] for r in transport.requests)
    assert "只能使用这些数字" not in first and "只能使用这些数字" in second


def test_numbers_still_unregistered_fall_back_to_card() -> None:
    gw, _ = _gateway(_l1(how="每组做 15 次。"), _l1(how="每周 9 次。"))
    result = _ask(KNEE, KNEE_INTENT, gw)
    assert result.record.layer == "l0" and result.record.degraded
    assert result.parts[0].kind == "notice" and result.parts[0].text == TEXTS.degraded
    assert result.record.number_rewrites == 2


def test_model_failure_degrades_to_card_and_says_so() -> None:
    gw, _ = _gateway(ConnectionError(), ConnectionError())
    result = _ask(KNEE, KNEE_INTENT, gw)
    assert result.record.layer == "l0" and result.record.degraded
    assert result.parts[0].kind == "notice"
    assert any(p.kind == "caution" and p.text == TEXTS.caution_discomfort for p in result.parts)


def test_no_knowledge_gives_general_answer() -> None:
    gw, calls = _gateway()
    result = _ask("动感单车和椭圆机哪个好", _intent(Intent.TRAINING_KNOWLEDGE), gw)
    assert result.record.layer == "general" and result.record.degraded and calls == []


def test_card_numbers_count_as_registered() -> None:
    """肌酸卡里有“3–5 克”：回答引用这两个数字不算未登记。"""
    gw, _ = _gateway(_l1(how="每天 3–5 克，随餐服用。"))
    result = _ask("肌酸要怎么吃，我膝盖有点疼", _intent(Intent.SUPPLEMENTS_DRUGS, "discomfort"), gw)
    assert result.record.layer == "l1" and result.record.number_rewrites == 0


def test_record_has_no_user_text() -> None:
    result = _ask("肌酸要怎么吃独特标记XYZ", _intent(Intent.SUPPLEMENTS_DRUGS))
    assert "XYZ" not in result.record.model_dump_json()


def test_unrouted_intent_rejected() -> None:
    with pytest.raises(ValueError, match="不经过问答"):
        _ask("你好", _intent(Intent.CHITCHAT))


# ---- 接口：回放 ----


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})
    with TestClient(create_app(settings, FixedClock(NOW), environ={})) as c:
        yield c


def test_run_can_be_replayed_by_owner_only(client: TestClient) -> None:
    body = client.post(
        "/chat", json={"text": "肌酸要怎么吃"}, headers={"X-IronM-User": "u1"}
    ).json()
    assert body["layer"] == "l0"
    run = client.get(f"/chat/runs/{body['request_id']}", headers={"X-IronM-User": "u1"})
    assert run.status_code == 200
    record = run.json()
    assert record["primary_domain"] == "nutritionist" and record["card_ids"] == ["QA-CREATINE"]
    assert "肌酸" not in run.text
    other = client.get(f"/chat/runs/{body['request_id']}", headers={"X-IronM-User": "u2"})
    assert other.status_code == 404


def test_detox_fasting_is_corrected_with_alternative() -> None:
    result = _ask("断食排毒有用吗", _intent(Intent.WELLNESS))
    assert result.record.card_ids == ("QA-DETOX-FASTING",)
    text = "".join(p.text for p in result.parts)
    assert "没有可靠证据" in text and any(p.kind == "how" for p in result.parts)
