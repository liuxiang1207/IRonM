"""人群分层与实时重新分层（spec [2026-10-20] §23）。各层边界见 tests/golden/safety_tiers.yaml。"""

import io
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from ironm.api.app import create_app
from ironm.contracts.rules import RuleParamError
from ironm.contracts.screening import Condition, SafetyFlag, Screening, Sex, Tier
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.rules.loader import RULES_DIR, load_rules
from ironm.safety.emergency import assess
from ironm.safety.tiers import TIER_RULE, TierParams, classify, note_message, tier_change

BOOK = load_rules(RULES_DIR)
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 10, 20, 4, tzinfo=UTC)


def _screening(**changes: Any) -> Screening:
    base: dict[str, Any] = {
        "birth_date": date(1990, 6, 15),
        "sex": Sex.FEMALE,
        "height_cm": 165.0,
        "weight_kg": 72.0,  # BMI 26.4
        "waist_cm": 86.0,
        "conditions": (),
    }
    base.update(changes)
    return Screening(**base)


def _tier(s: Screening, as_of: datetime = NOW) -> Any:
    return classify(s, as_of=as_of, tz=TZ, rulebook=BOOK)


# ---- 实时重新分层 ----


def test_red_flag_message_pauses_immediately() -> None:
    before = _tier(_screening())
    assert before.tier is Tier.A
    s = note_message(_screening(), assess("跑步的时候胸口突然很痛，现在还在疼"), NOW)
    after = _tier(s)
    assert after.tier is Tier.PAUSED
    assert tier_change(before, after) == "stricter"
    [flag] = s.flags
    assert flag.level == "emergency" and flag.categories == ("cardiovascular",)
    assert set(SafetyFlag.model_fields) == {"at", "level", "rule_ids", "categories"}  # 不含原话


def test_purge_message_pauses_then_stays_in_c_after_clearance() -> None:
    s = note_message(_screening(), assess("最近吃完饭总想去吐掉，已经吐了好几次"), NOW)
    paused = _tier(s)
    assert paused.tier is Tier.PAUSED
    cleared = s.model_copy(update={"clearances": (NOW + timedelta(hours=2),)})
    later = _tier(cleared, NOW + timedelta(days=1))
    assert later.tier is Tier.C
    assert tier_change(paused, later) == "looser"
    assert tier_change(_tier(_screening()), later) == "stricter"


def test_ordinary_message_changes_nothing() -> None:
    s = _screening()
    assert note_message(s, assess("今天练腿，好累"), NOW) is s


def test_tier_change_unknown_when_either_side_unknown() -> None:
    known = _tier(_screening())
    unknown = _tier(_screening(conditions=Screening().conditions))
    assert tier_change(known, unknown) == "unknown"
    assert tier_change(known, known) == "same"


# ---- 安全性质：多知道一条健康状况或红旗，只会更严或不变 ----

screenings = st.builds(
    _screening,
    birth_date=st.dates(date(1940, 1, 1), date(2015, 1, 1)),
    sex=st.sampled_from(Sex),
    height_cm=st.floats(140, 200),
    weight_kg=st.floats(35, 160),
    waist_cm=st.floats(55, 140),
    conditions=st.lists(st.sampled_from(Condition), unique=True, max_size=3).map(tuple),
)


@given(screenings, st.sampled_from(Condition))
def test_adding_condition_never_loosens(s: Screening, extra: Condition) -> None:
    assert isinstance(s.conditions, tuple)
    if extra in s.conditions:
        return
    more = s.model_copy(update={"conditions": (*s.conditions, extra)})
    assert tier_change(_tier(s), _tier(more)) in {"same", "stricter"}


@given(screenings)
def test_uncleared_flag_always_pauses(s: Screening) -> None:
    flag = SafetyFlag(at=NOW - timedelta(minutes=1), level="red_flag", rule_ids=("RF-DIZZY",))
    assert _tier(s.model_copy(update={"flags": (flag,)})).tier is Tier.PAUSED


# ---- 输入与规则参数校验 ----


def test_duplicate_condition_rejected() -> None:
    with pytest.raises(ValidationError, match="健康状况重复"):
        _screening(conditions=(Condition.PREGNANT, Condition.PREGNANT))


def _params(**changes: Any) -> dict[str, Any]:
    data = dict(BOOK.get(TIER_RULE).params)
    data.update(changes)
    return data


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"adult_age": "18"}, "adult_age"),
        ({"senior_age": 17}, "年龄界值"),
        ({"bmi_overweight": 33.0}, "BMI 界值"),
        ({"waist_male_cm": 0.0}, "腰围界值"),
        ({"b_conditions": ("hypertension", "recent_injury", "pregnant")}, "只能归入一层"),
        ({"b_conditions": ("hypertension",)}, "没有归入任何一层"),
        ({"b_conditions": ("hypertension", "recent_injury", "hypertensoin")}, "不认识"),
    ],
)
def test_bad_params_rejected(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        TierParams.model_validate(_params(**changes), strict=True)


def test_startup_fails_on_bad_tier_rule(tmp_path: Path) -> None:
    rules = tmp_path / "rules"
    shutil.copytree(RULES_DIR, rules)
    path = rules / "safety.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    doc["rules"][0]["params"]["b_conditions"] = ["hypertension"]
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    settings = load_settings(
        {"IRONM_DB_PATH": str(tmp_path / "db.sqlite"), "IRONM_RULES_DIR": str(rules)}
    )
    with pytest.raises(RuleParamError, match=TIER_RULE):
        create_app(settings, FixedClock(NOW), log_stream=io.StringIO())
