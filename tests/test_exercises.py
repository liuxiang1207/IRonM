"""动作库（spec [2026-10-20] §27）。"""

import io
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from ironm.api.app import create_app
from ironm.contracts.exercises import Equipment, Pattern, Restriction
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.knowledge.exercises import (
    ENVIRONMENTS,
    LIBRARY_PATH,
    LibraryLoadError,
    default_library,
    load_library,
)
from tests.test_api_events import NOW

LIB = default_library()
HOTEL, HOME, GYM = (LIB.environment(e) for e in ENVIRONMENTS)


def test_library_size_and_patterns() -> None:
    assert len(LIB.exercises) == 50
    assert {e.pattern for e in LIB.exercises} == set(Pattern)
    assert LIB.review == "approved"


@pytest.mark.parametrize("env", ENVIRONMENTS)
@pytest.mark.parametrize("pattern", list(Pattern))
def test_every_slot_has_substitutes(env: Any, pattern: Pattern) -> None:
    """验收：每个槽位在每种环境下都有替代；任一身体限制下仍有候选。"""
    equipment = LIB.environment(env)
    assert len(LIB.candidates(pattern, equipment=equipment)) >= 2
    for r in Restriction:
        assert LIB.candidates(pattern, equipment=equipment, restrictions=[r])


@pytest.mark.parametrize("pattern", list(Pattern))
def test_one_safe_default_per_pattern(pattern: Pattern) -> None:
    d = LIB.default_for(pattern)
    assert d.difficulty == 1 and not d.avoid_if and set(d.equipment) <= HOTEL


def test_filter_by_equipment_restriction_difficulty() -> None:
    hotel_squats = {c.exercise_id for c in LIB.candidates(Pattern.SQUAT, equipment=HOTEL)}
    assert "sq_goblet" not in hotel_squats  # 没有哑铃
    assert "sq_goblet" in {c.exercise_id for c in LIB.candidates(Pattern.SQUAT, equipment=HOME)}

    knee = LIB.candidates(Pattern.LUNGE, equipment=GYM, restrictions=[Restriction.KNEE_PAIN])
    ids = [c.exercise_id for c in knee]
    assert "lg_bulgarian" not in ids  # 膝盖疼时避免
    caution = LIB.restrictions[Restriction.KNEE_PAIN].caution
    assert all(caution in c.cautions for c in knee)  # 其余单腿动作都标为谨慎，附提示

    easy = LIB.candidates(Pattern.PUSH, equipment=GYM, max_difficulty=1)
    assert easy and all(c.difficulty == 1 for c in easy)
    every = LIB.candidates(Pattern.PULL, equipment=GYM)
    assert [c.difficulty for c in every] == sorted(c.difficulty for c in every)


def test_no_caution_without_matching_restriction() -> None:
    assert all(not c.cautions for c in LIB.candidates(Pattern.SQUAT, equipment=GYM))


def test_substitutes_same_pattern_without_self() -> None:
    subs = LIB.substitutes("sq_goblet", equipment=HOME)
    assert subs and "sq_goblet" not in {c.exercise_id for c in subs}
    assert {LIB.get(c.exercise_id).pattern for c in subs} == {Pattern.SQUAT}


def test_regress_follows_easier_chain() -> None:
    # 杠铃后蹲 → 高脚杯深蹲（家里有哑铃）
    assert LIB.regress("sq_back_barbell", equipment=HOME).exercise_id == "sq_goblet"
    # 酒店没有哑铃也没有箱子：一路降到徒手深蹲
    assert LIB.regress("sq_back_barbell", equipment=HOTEL).exercise_id == "sq_bodyweight"
    # 引体向上，肩膀疼：反向划船、门框划船都可用但需谨慎 → 第一个可用的是反向划船（健身房）
    up = LIB.regress("pl_pull_up", equipment=GYM, restrictions=[Restriction.SHOULDER_PAIN])
    assert up.exercise_id == "pl_inverted_row" and up.cautions


def test_regress_falls_back_to_default() -> None:
    # 默认动作本身没有更容易的动作：返回它自己
    assert LIB.regress("co_dead_bug", equipment=HOTEL).exercise_id == "co_dead_bug"
    # 跳绳 → 慢跑（脚踝疼时避免）→ 快走
    r = LIB.regress("cd_jump_rope", equipment=HOTEL, restrictions=[Restriction.ANKLE_PAIN])
    assert r.exercise_id == "cd_brisk_walk"


def test_texts_have_no_english_words() -> None:
    """说明为原创中文：不应出现连续 3 个以上英文字母（机器翻译残留）。"""
    for e in LIB.exercises:
        for text in (e.name, *e.key_points, *e.common_errors, *e.self_check):
            assert not re.search(r"[A-Za-z]{3,}", text), (e.id, text)


# ---- 校验 ----


def _data() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(LIBRARY_PATH.read_text(encoding="utf-8"))
    return data


def _ex(data: dict[str, Any], exercise_id: str) -> dict[str, Any]:
    found: dict[str, Any] = next(e for e in data["exercises"] if e["id"] == exercise_id)
    return found


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["exercises"].append(dict(_ex(d, "sq_box"))), "编号重复"),
        (lambda d: _ex(d, "sq_box").update(equipment=["sofa"]), "equipment"),
        (lambda d: _ex(d, "sq_box").update(avoid_if=["neck_pain"]), "avoid_if"),
        (lambda d: _ex(d, "sq_box").update(avoid_if=["knee_pain"]), "既“避免”又“谨慎”"),
        (lambda d: _ex(d, "sq_bodyweight").update(easier="sq_goblet"), "难度方向不对"),
        (lambda d: _ex(d, "sq_goblet").update(harder="sq_box"), "难度方向不对"),
        (
            lambda d: (
                _ex(d, "sq_box").update(easier="sq_wall_sit")
                or _ex(d, "sq_wall_sit").update(easier="sq_box")
            ),
            "绕回",
        ),
        (lambda d: _ex(d, "sq_box").update(easier="sq_nothing"), "不存在的动作"),
        (lambda d: _ex(d, "sq_box").update(easier="hg_glute_bridge"), "不是同一动作模式"),
        (lambda d: _ex(d, "sq_box").update(default=True), "恰好有一个默认"),
        (lambda d: _ex(d, "sq_sit_to_stand").update(default=False), "恰好有一个默认"),
        (lambda d: _ex(d, "sq_sit_to_stand").update(avoid_if=["ankle_pain"]), "默认安全动作"),
        (lambda d: _ex(d, "sq_box").update(key_points=["只有一条"]), "key_points"),
        (lambda d: _ex(d, "sq_box").update(easier="sq_box"), "不能是自己"),
        (lambda d: d["restrictions"].pop("wrist_pain"), "缺少说明"),
        (lambda d: d["environments"]["gym"].remove("jump_rope"), "健身房包含全部器材"),
    ],
)
def test_bad_library_rejected(tmp_path: Path, mutate: Any, message: str) -> None:
    data = _data()
    mutate(data)
    path = tmp_path / "exercises.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(LibraryLoadError, match=message):
        load_library(path)


def test_startup_fails_on_bad_library(tmp_path: Path) -> None:
    data = _data()
    _ex(data, "sq_box").update(easier="sq_nothing")
    bad = tmp_path / "exercises.yaml"
    bad.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})
    with pytest.raises(LibraryLoadError):
        create_app(settings, FixedClock(NOW), log_stream=io.StringIO(), library_path=bad)


def test_equipment_enum_covered_by_library_text() -> None:
    assert set(LIB.environment("gym")) == set(Equipment)
