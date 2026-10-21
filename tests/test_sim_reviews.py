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
