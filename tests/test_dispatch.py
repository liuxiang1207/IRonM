"""调度（spec [2026-10-21] §5）：快速通道、管家调度与规则校验、恢复先于训练、来源写进运行记录。"""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from ironm.contracts.model_call import ModelCall, ModelStage
from ironm.contracts.roles import AgentId
from ironm.core.clock import FixedClock
from ironm.core.ids import SeededIdGenerator
from ironm.knowledge.qa_cards import default_cards
from ironm.llm.config import MODELS_PATH, load_models
from ironm.llm.gateway import Gateway
from ironm.orchestrator.chat import load_answer_texts
from ironm.orchestrator.dispatch import _Butler, check_butler, stages_for
from ironm.orchestrator.intent import Intent, IntentResult
from ironm.orchestrator.qa import answer_question
from ironm.rules.loader import RULES_DIR, load_rules
from tests.llm.test_gateway import ENV, FakeTransport, _reply

_A = AgentId
RULES = load_rules(RULES_DIR)
NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)
CROSS = "增肌一周练几次，蛋白质一天吃多少"


def _intent(
    intent: Intent, *domains: Intent, confidence: float = 0.85, concerns: tuple[str, ...] = ()
) -> IntentResult:
    return IntentResult.model_validate(
        {
            "intent": intent,
            "concerns": concerns,
            "confidence": confidence,
            "source": "rule",
            "clarify": False,
            "domains": domains or (intent,),
        }
    )


CROSS_INTENT = _intent(Intent.TRAINING_KNOWLEDGE, Intent.TRAINING_KNOWLEDGE, Intent.NUTRITION)


def _gateway(*results: object) -> tuple[Gateway, list[ModelCall]]:
    calls: list[ModelCall] = []
    gw = Gateway(
        load_models(MODELS_PATH),
        environ=ENV,
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(4),
        transport=FakeTransport(*results),
        record=calls.append,
    )
    return gw, calls


def _butler(
    primary: str,
    assistants: list[str],
    reason: str = "training_and_nutrition",
    actions: list[str] | None = None,
) -> dict[str, object]:
    body: dict[str, Any] = {"primary": primary, "assistants": assistants, "reason_code": reason}
    if actions is not None:
        body["actions"] = actions
    return _reply(json.dumps(body))


L1 = _reply(
    json.dumps(
        {"conclusion": "每周至少 2 天力量训练，蛋白质每千克体重约 1.6 克。"}, ensure_ascii=False
    )
)


def _ask(text: str, intent: IntentResult, gateway: Gateway | None) -> Any:
    return answer_question(
        text,
        intent,
        cards=default_cards(),
        texts=load_answer_texts(),
        gateway=gateway,
        request_id="req-1",
        rulebook=RULES,
    )


def test_cross_domain_question_is_dispatched_by_butler() -> None:
    gw, calls = _gateway(_butler("training_planner", ["nutritionist", "safety_officer"]), L1)
    record = _ask(CROSS, CROSS_INTENT, gw).record
    assert record.dispatch == "butler" and record.dispatch_reason == "training_and_nutrition"
    assert record.primary is _A.TRAINING_PLANNER
    assert record.assistants == (_A.NUTRITIONIST, _A.SAFETY_OFFICER)
    assert [c.stage for c in calls] == [ModelStage.DISPATCH, ModelStage.ANSWER]
    assert record.layer == "l1"  # 两个领域的卡合在一起回答


@pytest.mark.parametrize(
    ("butler", "reason"),
    [
        (_butler("training_planner", ["nutritionist"]), "missing_safety_officer"),
        (
            _butler("training_planner", ["nutritionist", "recovery", "safety_officer"]),
            "too_many_assistants",
        ),
        (
            _butler("training_planner", ["evidence_reviewer", "safety_officer"]),
            "reviewer_not_allowed",
        ),
        (
            _butler("training_planner", ["safety_officer"], actions=["answer", "write_plan"]),
            "forbidden_action",
        ),
        (
            _butler("training_planner", ["safety_officer"], actions=["write_profile"]),
            "forbidden_action",
        ),
        (_butler("dance_teacher", ["safety_officer"]), "unknown_agent"),
        (_butler("training_planner", ["training_planner", "safety_officer"]), "duplicate_agent"),
    ],
)
def test_invalid_butler_dispatch_falls_back_to_rule_table(
    butler: dict[str, object], reason: str
) -> None:
    gw, _ = _gateway(butler, L1)
    record = _ask(CROSS, CROSS_INTENT, gw).record
    assert (
        record.dispatch == "rule_fallback"
        and record.dispatch_reason == f"dispatch_invalid:{reason}"
    )
    assert record.primary is _A.TRAINING_PLANNER and record.assistants == (_A.SAFETY_OFFICER,)


def test_recovery_runs_before_training() -> None:
    gw, _ = _gateway(
        _butler("training_planner", ["recovery", "safety_officer"], "sleep_and_training"), L1
    )
    intent = _intent(Intent.TRAINING_KNOWLEDGE, Intent.TRAINING_KNOWLEDGE, Intent.RECOVERY_SLEEP)
    record = _ask("最近睡不好，一周练几次合适", intent, gw).record
    stage_of = {s.agent: s.stage_index for s in record.steps}
    assert stage_of[_A.RECOVERY] == 0
    assert stage_of[_A.TRAINING_PLANNER] == 1 and stage_of[_A.SAFETY_OFFICER] == 1


def test_later_stage_sees_earlier_facts_only_as_ids() -> None:
    assert stages_for(_A.TRAINING_PLANNER, (_A.RECOVERY, _A.SAFETY_OFFICER)) == (
        (_A.RECOVERY,),
        (_A.TRAINING_PLANNER, _A.SAFETY_OFFICER),
    )
    assert stages_for(_A.NUTRITIONIST, (_A.RECOVERY, _A.SAFETY_OFFICER)) == (
        (_A.NUTRITIONIST, _A.RECOVERY, _A.SAFETY_OFFICER),
    )


def test_clear_single_domain_question_takes_fast_lane() -> None:
    gw, calls = _gateway()
    record = _ask("高脚杯深蹲怎么做", _intent(Intent.EXERCISE_TECHNIQUE), gw).record
    assert record.dispatch == "fast_lane" and calls == []


def test_low_confidence_single_domain_goes_to_butler() -> None:
    gw, calls = _gateway(_butler("exercise_coach", ["safety_officer"], "technique"))
    record = _ask("深蹲", _intent(Intent.EXERCISE_TECHNIQUE, confidence=0.45), gw).record
    assert record.dispatch == "butler" and calls[0].stage is ModelStage.DISPATCH


def test_models_unavailable_use_rule_table_and_still_answer() -> None:
    gw, calls = _gateway(ConnectionError(), ConnectionError(), ConnectionError(), ConnectionError())
    result = _ask(CROSS, CROSS_INTENT, gw)
    assert result.record.dispatch == "rule_table" and result.parts
    dispatch_calls = [c for c in calls if c.stage is ModelStage.DISPATCH]
    assert len(dispatch_calls) == 2 and dispatch_calls[-1].fallback_used


def test_no_gateway_uses_rule_table() -> None:
    assert _ask(CROSS, CROSS_INTENT, None).record.dispatch == "rule_table"


def test_check_butler_accepts_safety_officer_as_primary() -> None:
    out = _Butler(primary="safety_officer", assistants=[], reason_code="safety")
    assert check_butler(out) == (_A.SAFETY_OFFICER, ())
