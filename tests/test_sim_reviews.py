"""快进模拟：5 类模拟用户各 8 周，每次复盘与调整都符合决策表（spec [2026-10-21] §17，计划 D40）。"""

from pathlib import Path

import pytest

from sim.reviews import GOALS, PersonaReport, run_all


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> dict[str, PersonaReport]:
    workdir: Path = tmp_path_factory.mktemp("sim-reviews")
    return {r.persona: r for r in run_all(workdir)}


def test_every_adjustment_follows_the_decision_tables(reports: dict[str, PersonaReport]) -> None:
    assert set(reports) == set(GOALS)
    assert [v for r in reports.values() for v in r.violations] == []


def test_simulation_actually_exercises_the_tables(reports: dict[str, PersonaReport]) -> None:
    """不是空跑：每类用户都按周复盘，进阶、恢复性下调都真的发生过。"""
    assert all(r.reviews >= 7 for r in reports.values())
    assert sum(r.adjusted for r in reports.values()) >= 10
    rows = [row for r in reports.values() for row in r.rows]
    assert "progress" in rows and "recovery" in rows


def test_fatigue_build_up_triggers_recovery(reports: dict[str, PersonaReport]) -> None:
    """疲劳累积的用户：第 5 周起疲劳上升、睡眠变差 → 恢复性下调（依赖签到被复盘用上）。"""
    assert reports["fatigue"].rows[4:].count("recovery") >= 3


def test_diet_reviews_only_for_fat_loss(reports: dict[str, PersonaReport]) -> None:
    for r in reports.values():
        if r.goal == "fat_loss":
            assert len(r.diet_rows) >= 3
        else:
            assert r.diet_rows == ()


def test_long_recovery_stops_cutting_after_two_weeks(reports: dict[str, PersonaReport]) -> None:
    """完成度低的用户连续 6 周睡得差：只在前两周下调，之后不再减量（spec §17 修正）。"""
    r = reports["low_completion"]
    assert r.rows[:6] == ("recovery",) * 6
    assert r.adjusted == 2


def test_diet_review_decides_in_second_month(reports: dict[str, PersonaReport]) -> None:
    """28 天体重趋势、门槛 0.4：每周称两次、记饮食的减脂用户第二个月起得到饮食判断。"""
    for name in ("adherent", "fatigue"):
        assert [row for row in reports[name].diet_rows[1:] if row != "insufficient"]
    assert set(reports["weekly_weigher"].diet_rows) == {"insufficient"}  # 每周只称一次、不记饮食
