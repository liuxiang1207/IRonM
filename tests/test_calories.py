"""热量计算（spec [2026-10-20] §25）。手算样例见 tests/golden/calories.yaml。"""

import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from ironm.contracts.calories import Activity
from ironm.contracts.rules import RuleParamError
from ironm.contracts.screening import Screening, Sex, Tier, TierResult
from ironm.contracts.unknown import is_known
from ironm.engine.calories import PLAN_TIERS, calorie_target
from ironm.engine.rule_params import validate_rule_params
from ironm.rules.loader import RULES_DIR, RuleBook, load_rules
from ironm.safety.decision import plan_allowed

BOOK = load_rules(RULES_DIR)
TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 10, 20, 4, tzinfo=UTC)


def _tier(tier: Tier) -> TierResult:
    return TierResult(
        tier=tier, status="ok", matched=(tier,), reasons=(), missing=(), bmi=24.0, rule_id="R"
    )


def _target(s: Screening, tier: Tier = Tier.A, book: RuleBook = BOOK, **kw: Any) -> Any:
    return calorie_target(
        s, kw.pop("activity", Activity.SEDENTARY), _tier(tier), as_of=NOW, tz=TZ, rulebook=book
    )


def _book(tmp_path: Path, rule_set: str, rule_id: str, **params: Any) -> RuleBook:
    rules = tmp_path / "rules"
    shutil.copytree(RULES_DIR, rules)
    path = rules / f"{rule_set}.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    [rule] = [r for r in doc["rules"] if r["id"] == rule_id]
    rule["params"].update(params)
    path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    return load_rules(rules)


# 女 30 岁、160 厘米、61.1 千克：静息恰为 1300，久坐每日总消耗 1820，A′ 缺口 300 → 1520
# （结果是整数，便于测底线边界）
EXACT = Screening(
    birth_date=date(1996, 1, 1),
    sex=Sex.FEMALE,
    height_cm=160.0,
    weight_kg=61.1,
    conditions=(),
)


def test_floor_boundary(tmp_path: Path) -> None:
    at = _target(EXACT, Tier.A_PRIME, _book(tmp_path / "a", "calories", "R-CAL-FLOOR", female=1520))
    assert (at.target_kcal, at.floor_applied, at.deficit_kcal) == (1520, False, 300)
    above = _target(
        EXACT, Tier.A_PRIME, _book(tmp_path / "b", "calories", "R-CAL-FLOOR", female=1521)
    )
    assert (above.target_kcal, above.floor_applied, above.deficit_kcal) == (1521, True, 299)


def test_plan_tiers_match_safety_decision() -> None:
    assert set(PLAN_TIERS) == {tier for tier in Tier if plan_allowed(tier)}


@pytest.mark.parametrize(
    ("rule_set", "rule_id", "params", "message"),
    [
        ("calories", "R-CAL-MACRO", {"carb": 60}, "之和应为 100"),
        ("calories", "R-CAL-MACRO", {"high_protein_protein": 30}, "高蛋白"),
        ("calories", "R-CAL-MACRO", {"meals": [30, 40, 20]}, "三餐"),
        ("calories", "R-CAL-FLOOR", {"female": 700}, "绝对底线"),
        ("calories", "R-CAL-DEFICIT", {"b": -100}, "R-CAL-DEFICIT"),
        ("calories", "R-CAL-PAL", {"moderate": 1.3}, "依次增大"),
        ("calories", "R-CAL-PROTEIN", {"max_pct": 35}, "不能超过 100"),
        ("calories", "R-CAL-PROTEIN", {"g_per_kg_high_protein": 2.2}, "不能超过上限"),
        ("calories", "R-CAL-REE", {"age_coef": "5"}, "R-CAL-REE"),
    ],
)
def test_bad_rules_fail_at_startup(
    tmp_path: Path, rule_set: str, rule_id: str, params: dict[str, Any], message: str
) -> None:
    book = _book(tmp_path, rule_set, rule_id, **params)
    with pytest.raises(RuleParamError, match=message):
        validate_rule_params(book)


# ---- 属性 ----

people = st.builds(
    Screening,
    # 65 岁以下：老年“只给维持热量”的界值会让体重增加时缺口变大，另由黄金用例覆盖
    birth_date=st.dates(date(1962, 1, 1), date(2008, 1, 1)),
    sex=st.sampled_from(Sex),
    height_cm=st.floats(140, 200).map(lambda h: round(h, 1)),
    weight_kg=st.floats(40, 160).map(lambda w: round(w, 1)),
    conditions=st.just(()),
)

everyone = st.builds(
    Screening,
    birth_date=st.dates(date(1940, 1, 1), date(2008, 1, 1)),
    sex=st.sampled_from(Sex),
    height_cm=st.floats(140, 200).map(lambda h: round(h, 1)),
    weight_kg=st.floats(40, 160).map(lambda w: round(w, 1)),
    conditions=st.just(()),
)
plan_tiers = st.sampled_from(sorted(PLAN_TIERS))
activities = st.sampled_from(Activity)


@given(people, plan_tiers, activities, st.floats(0.1, 20).map(lambda d: round(d, 1)))
def test_heavier_never_lowers_target(s: Screening, tier: Tier, act: Activity, more: float) -> None:
    assert is_known(s.weight_kg)
    heavier = s.model_copy(update={"weight_kg": min(400.0, s.weight_kg + more)})
    assert (
        _target(heavier, tier, activity=act).target_kcal
        >= _target(s, tier, activity=act).target_kcal
    )


@given(everyone, plan_tiers, activities)
def test_target_never_below_floor_and_meals_add_up(s: Screening, tier: Tier, act: Activity) -> None:
    result = _target(s, tier, activity=act)
    floor = 1500 if s.sex is Sex.MALE else 1200
    assert result.target_kcal >= floor
    assert result.target_kcal >= result.ree_kcal  # 不低于静息能量消耗
    assert sum(result.meals_kcal) == result.target_kcal
    m = result.macros
    kcal = m.carb_g * 4 + m.fat_g * 9 + m.protein_g * 4
    assert abs(kcal - result.target_kcal) <= 10  # 克数取整的误差
    assert m.protein_g * 4 <= result.target_kcal * 0.3 + 2  # 蛋白质不超过 30%
    assert result.deficit_kcal >= 0
