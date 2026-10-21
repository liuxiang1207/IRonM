"""查知识的领域打分（spec [2026-10-21] §12；沿用 §5 领域打分的用例）。"""

from typing import Any

import pytest

from ironm.contracts.roles import AgentId
from ironm.orchestrator.domains import (
    DOMAIN_RULE,
    MAX_RELATED,
    DomainParams,
    select_domains,
)
from ironm.orchestrator.intent import Intent, IntentResult
from ironm.rules.loader import RULES_DIR, load_rules

_A = AgentId
RULES = load_rules(RULES_DIR)


def _intent(intent: Intent, *domains: Intent, confidence: float = 0.85) -> IntentResult:
    return IntentResult.model_validate(
        {
            "intent": intent,
            "concerns": (),
            "confidence": confidence,
            "source": "rule",
            "clarify": False,
            "domains": domains or (intent,),
        }
    )


@pytest.mark.parametrize(
    ("intent", "primary"),
    [
        (Intent.NUTRITION, _A.NUTRITIONIST),
        (Intent.RECOVERY_SLEEP, _A.RECOVERY),
        (Intent.EXERCISE_TECHNIQUE, _A.EXERCISE_COACH),
        (Intent.HABIT_MOTIVATION, _A.BEHAVIOR_COACH),
    ],
)
def test_single_domain_goes_to_its_owner(intent: Intent, primary: AgentId) -> None:
    assert select_domains(_intent(intent), RULES).primary is primary


def test_cross_domain_question_adds_related_domain() -> None:
    d = select_domains(
        _intent(Intent.TRAINING_KNOWLEDGE, Intent.TRAINING_KNOWLEDGE, Intent.NUTRITION), RULES
    )
    assert d.primary is _A.TRAINING_PLANNER and _A.NUTRITIONIST in d.related


def test_weak_related_domain_below_floor_is_left_out() -> None:
    """营养题里训练规划的固有分只有 0.3，低于门槛，不作为相关领域。"""
    d = select_domains(_intent(Intent.NUTRITION), RULES)
    assert d.related == ()


def test_low_confidence_keeps_primary_but_drops_related() -> None:
    d = select_domains(
        _intent(
            Intent.TRAINING_KNOWLEDGE, Intent.TRAINING_KNOWLEDGE, Intent.NUTRITION, confidence=0.2
        ),
        RULES,
    )
    assert d.primary is _A.TRAINING_PLANNER and d.related == ()


def test_related_capped_and_safety_never_scored() -> None:
    d = select_domains(
        _intent(
            Intent.PLAN,
            Intent.PLAN,
            Intent.NUTRITION,
            Intent.RECOVERY_SLEEP,
            Intent.HABIT_MOTIVATION,
        ),
        RULES,
    )
    assert len(d.related) <= MAX_RELATED
    assert _A.SAFETY_OFFICER not in {r for r, _ in d.scores}


def test_unknown_intent_has_no_domain() -> None:
    d = select_domains(_intent(Intent.UNKNOWN), RULES)
    assert d.primary is None and d.related == ()


@pytest.mark.parametrize("params", [{"related_min_score": 1.5}, {"related_min_ratio": -0.1}])
def test_bad_params_rejected(params: dict[str, Any]) -> None:
    base = RULES.get(DOMAIN_RULE).params_as(DomainParams).model_dump()
    with pytest.raises(ValueError):
        DomainParams.model_validate({**base, **params})
