"""Agent 基类与 Skill 加载器（spec [2026-10-20] §28）。"""

import io
from pathlib import Path
from typing import Any

import pytest
import yaml

from ironm.agents.base import NO_SKILL, Agent, AgentContractError
from ironm.api.app import create_app
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.screening import Tier, TierResult
from ironm.contracts.skills import ScenePack
from ironm.contracts.snapshot import Snapshot
from ironm.contracts.unknown import Unknown, UnknownReason
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.rules.loader import RULES_DIR, load_rules
from ironm.skills.loader import LoadedSkill, SkillLoadError, load_skills, select
from ironm.skills.view import SnapshotView, UndeclaredDataError
from tests.test_api_events import NOW
from tests.test_contracts_snapshot_plan import _snap, _snapshot

NUTRI = AgentId.NUTRITIONIST


def _manifest(skill_id: str, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": skill_id,
        "owner_agent": "nutritionist",
        "version": "0.1.0",
        "description": "测试用",
        "triggers": {"intents": ["nutrition.plateau"]},
        "applies_to": {"scene_packs": ["fat_loss"], "population_tiers": ["a", "a_prime", "b"]},
        "requires_data": ["metric.weight_trend"],
        "produces": [],
        "priority": 100,
        "prescriptive": True,
        "review": "approved",
        "reviewed_by": "测试",
        "tests": "tests/test_skills.py",
    }
    data.update(overrides)
    return data


# 测试用 Skill 的实现：返回“本次无意见”，读一项数据；SKILL_AGENT / SKILL_NAME 可替换成别的值来测校验
RUN_SOURCE = """
from ironm.contracts.agents import NoOpinionOutput
from ironm.contracts.roles import AgentId, OutputKind

READ = {read!r}
SKILL_AGENT = {agent!r}
SKILL_NAME = {name!r}


def run(view, context):
    for key in READ:
        view.get(key)
    m = context.skill.manifest
    return NoOpinionOutput(
        agent=AgentId(SKILL_AGENT) if SKILL_AGENT else m.owner_agent,
        skill=SKILL_NAME or m.versioned,
        kind=OutputKind.NO_OPINION,
        reason_code="test",
    )
"""


def _write(
    root: Path,
    manifest: dict[str, Any],
    body: str = "说明正文。",
    *,
    read: tuple[str, ...] = (),
    agent: str = "",
    name: str = "",
) -> Path:
    folder: Path = root / str(manifest["owner_agent"]) / str(manifest["id"])
    folder.mkdir(parents=True, exist_ok=True)
    head = yaml.safe_dump(manifest, allow_unicode=True)
    path = folder / "SKILL.md"
    path.write_text(f"---\n{head}---\n\n{body}\n", encoding="utf-8")
    source = RUN_SOURCE.format(read=read, agent=agent, name=name)
    (folder / "skill.py").write_text(source, encoding="utf-8")
    return path


def _tier(tier: Tier | None) -> TierResult:
    value: Tier | Unknown = tier if tier else Unknown(unknown=UnknownReason.INSUFFICIENT_DATA)
    return TierResult(
        tier=value,
        status="ok" if tier else "insufficient",
        matched=(tier,) if tier else (),
        reasons=(),
        missing=(),
        bmi=25.0,
        rule_id="R",
    )


def _select(skills: tuple[LoadedSkill, ...], tier: Tier | None, **kw: Any) -> Any:
    return select(
        skills,
        agent=kw.get("agent", NUTRI),
        intent=kw.get("intent", "nutrition.plateau"),
        scene=kw.get("scene", ScenePack.FAT_LOSS),
        tier=_tier(tier),
    )


@pytest.fixture
def library(tmp_path: Path) -> tuple[LoadedSkill, ...]:
    _write(tmp_path, _manifest("plateau_adjust", produces=["daily_kcal_target"]))
    _write(
        tmp_path,
        _manifest(
            "food_log_summary",
            prescriptive=False,
            applies_to={
                "scene_packs": ["fat_loss"],
                "population_tiers": ["a", "a_prime", "b", "collaborative", "c"],
            },
        ),
    )
    _write(tmp_path, _manifest("intake_questions", owner_agent="assessor", prescriptive=False))
    return load_skills(tmp_path)


# ---- 选择 ----


@pytest.mark.parametrize("tier", [Tier.C, Tier.D, Tier.PAUSED, Tier.NOT_APPLICABLE, None])
def test_no_prescriptive_skill_when_no_plan(
    library: tuple[LoadedSkill, ...], tier: Tier | None
) -> None:
    sel = _select(library, tier)
    loaded = {s.manifest.id for s in sel.loaded}
    assert "plateau_adjust" not in loaded
    reason = dict(sel.skipped)["plateau_adjust"]
    assert reason == ("tier_unknown" if tier is None else "tier_no_plan")
    assert ("food_log_summary" in loaded) == (tier is Tier.C)  # 非处方类按适用分层加载


def test_a_tier_loads_both(library: tuple[LoadedSkill, ...]) -> None:
    sel = _select(library, Tier.A)
    assert [s.manifest.id for s in sel.loaded] == ["food_log_summary", "plateau_adjust"]
    assert sel.versions == ("food_log_summary@0.1.0", "plateau_adjust@0.1.0")
    assert "intake_questions" not in dict(sel.skipped)  # 别的 Agent 的 Skill 不参与


def test_mismatch_reasons(library: tuple[LoadedSkill, ...]) -> None:
    assert dict(_select(library, Tier.A, intent="goal.create").skipped) == {
        "food_log_summary": "intent_mismatch",
        "plateau_adjust": "intent_mismatch",
    }
    assert dict(_select(library, Tier.A, scene=ScenePack.STRENGTH).skipped)["plateau_adjust"] == (
        "scene_mismatch"
    )
    assert dict(_select(library, Tier.COLLABORATIVE).skipped)["plateau_adjust"] == "tier_mismatch"


def test_unknown_tier_needs_explicit_opt_in(tmp_path: Path) -> None:
    applies = {"scene_packs": ["fat_loss"], "population_tiers": ["a"], "unknown_tier": True}
    _write(tmp_path, _manifest("intake", prescriptive=False, applies_to=applies))
    _write(tmp_path, _manifest("summary", prescriptive=False))
    sel = _select(load_skills(tmp_path), None)
    assert [s.manifest.id for s in sel.loaded] == ["intake"]
    assert dict(sel.skipped) == {"summary": "tier_unknown"}


def test_same_field_only_from_higher_priority(tmp_path: Path) -> None:
    _write(tmp_path, _manifest("first", priority=10, produces=["daily_kcal_target"]))
    _write(tmp_path, _manifest("second", priority=20, produces=["daily_kcal_target", "diet_mode"]))
    sel = _select(load_skills(tmp_path), Tier.A)
    assert [s.manifest.id for s in sel.loaded] == ["first"]
    assert dict(sel.skipped) == {"second": "field_conflict"}


# ---- 裁剪快照 ----


def _snapshot_obj() -> Snapshot:
    return _snap(_snapshot())


def test_view_reads_only_declared() -> None:
    view = SnapshotView(
        _snapshot_obj(), {"tier", "metric.weight_trend", "metric.missing", "profile.sex"}
    )
    assert view.get("tier") is Tier.A
    trend = view.get("metric.weight_trend")
    assert getattr(trend, "value", None) == -0.42
    assert view.get("profile.sex") == "male"
    assert isinstance(view.get("metric.missing"), Unknown)  # 声明了但没有
    for key in ("profile.age", "precision", "current_plan", "metric.diet_kcal_mean_7d"):
        with pytest.raises(UndeclaredDataError):
            view.get(key)


# ---- Agent ----

BOOK = load_rules(RULES_DIR)


def test_agent_no_skill_returns_single_no_opinion(library: tuple[LoadedSkill, ...]) -> None:
    [out] = Agent(NUTRI).execute(
        _snapshot_obj(), _select(library, Tier.A, intent="goal.create"), BOOK
    )
    assert out.kind is OutputKind.NO_OPINION and out.skill == NO_SKILL
    assert out.reason_code == "no_matching_skill"


def test_one_output_per_skill_each_with_own_view(tmp_path: Path) -> None:
    _write(tmp_path, _manifest("first", requires_data=["tier"]), read=("tier",))
    _write(
        tmp_path,
        _manifest("second", requires_data=["metric.weight_trend"]),
        read=("metric.weight_trend",),
    )
    outputs = Agent(NUTRI).execute(_snapshot_obj(), _select(load_skills(tmp_path), Tier.A), BOOK)
    assert [o.skill for o in outputs] == ["first@0.1.0", "second@0.1.0"]


def test_skill_cannot_read_other_skills_data(tmp_path: Path) -> None:
    """每个 Skill 只能读自己声明的数据，即使同一 Agent 的另一个 Skill 声明了。"""
    _write(tmp_path, _manifest("first", requires_data=["tier"]), read=("metric.weight_trend",))
    _write(tmp_path, _manifest("second", requires_data=["metric.weight_trend"]))
    with pytest.raises(UndeclaredDataError):
        Agent(NUTRI, strict=True).execute(
            _snapshot_obj(), _select(load_skills(tmp_path), Tier.A), BOOK
        )


@pytest.mark.parametrize(
    ("agent", "name", "message"),
    [("assessor", "", "标成了 assessor"), ("", "first@9.9.9", "别的 Skill 版本")],
)
def test_output_must_match_running_skill(
    tmp_path: Path, agent: str, name: str, message: str
) -> None:
    _write(tmp_path, _manifest("first"), agent=agent, name=name)
    with pytest.raises(AgentContractError, match=message):
        Agent(NUTRI, strict=True).execute(
            _snapshot_obj(), _select(load_skills(tmp_path), Tier.A), BOOK
        )


def test_selection_for_other_agent_rejected(library: tuple[LoadedSkill, ...]) -> None:
    with pytest.raises(AgentContractError):
        Agent(NUTRI).execute(
            _snapshot_obj(), _select(library, Tier.A, agent=AgentId.ASSESSOR), BOOK
        )


# ---- 头信息校验 ----


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (_manifest("x", owner_agent="chef"), "头信息有误"),
        (_manifest("x", version="1.0"), "头信息有误"),
        (_manifest("x", produces=["session_slots"]), "越权"),
        (
            _manifest(
                "x", applies_to={"scene_packs": ["fat_loss"], "population_tiers": ["a", "c"]}
            ),
            "不可给计划",
        ),
        (
            _manifest(
                "x",
                applies_to={
                    "scene_packs": ["fat_loss"],
                    "population_tiers": ["a"],
                    "unknown_tier": True,
                },
            ),
            "分层未知",
        ),
        (_manifest("x", requires_data=["password"]), "头信息有误"),
        (_manifest("Bad-Id"), "头信息有误"),
    ],
)
def test_bad_manifest_rejected(tmp_path: Path, manifest: dict[str, Any], message: str) -> None:
    folder = tmp_path / "nutritionist" / manifest["id"]
    folder.mkdir(parents=True)
    head = yaml.safe_dump(manifest, allow_unicode=True)
    (folder / "SKILL.md").write_text(f"---\n{head}---\n\n正文\n", encoding="utf-8")
    (folder / "skill.py").write_text(
        RUN_SOURCE.format(read=(), agent="", name=""), encoding="utf-8"
    )
    with pytest.raises(SkillLoadError, match=message):
        load_skills(tmp_path)


def test_skill_py_required(tmp_path: Path) -> None:
    path = _write(tmp_path, _manifest("x"))
    (path.parent / "skill.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(SkillLoadError, match="run"):
        load_skills(tmp_path)
    (path.parent / "skill.py").unlink()
    with pytest.raises(SkillLoadError, match=r"缺少 skill\.py"):
        load_skills(tmp_path)


def test_skill_md_shape_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, _manifest("x"))
    path.write_text("没有头信息\n", encoding="utf-8")
    with pytest.raises(SkillLoadError, match="--- 开头"):
        load_skills(tmp_path)
    path.write_text("---\nid: x\n", encoding="utf-8")
    with pytest.raises(SkillLoadError, match="结束的 ---"):
        load_skills(tmp_path)
    _write(tmp_path, _manifest("x"), body="")
    with pytest.raises(SkillLoadError, match="说明正文"):
        load_skills(tmp_path)


def test_folder_must_match_manifest(tmp_path: Path) -> None:
    path = _write(tmp_path, _manifest("x"))
    wrong = tmp_path / "assessor" / "x"
    wrong.mkdir(parents=True)
    (wrong / "SKILL.md").write_text(path.read_text("utf-8"), encoding="utf-8")
    (wrong / "skill.py").write_text((path.parent / "skill.py").read_text("utf-8"), encoding="utf-8")
    with pytest.raises(SkillLoadError, match="目录应为"):
        load_skills(tmp_path)


def test_startup_fails_on_bad_skill(tmp_path: Path) -> None:
    _write(tmp_path / "skills", _manifest("x", produces=["session_slots"]))
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})
    with pytest.raises(SkillLoadError):
        create_app(
            settings, FixedClock(NOW), log_stream=io.StringIO(), skills_dir=tmp_path / "skills"
        )


def test_bundled_skills_reviewed_and_tested() -> None:
    """内置 Skill 全部可加载、已审定、各自的测试文件存在。"""
    root = Path(__file__).resolve().parents[1]
    skills = load_skills()
    assert {s.manifest.id for s in skills} >= {
        "intake_questions",
        "tier_limits",
        "kcal_floor",
        "eating_relationship",
    }
    for s in skills:
        assert s.manifest.review == "approved", s.manifest.id
        assert (root / s.manifest.tests).is_file(), s.manifest.tests


def test_skill_error_isolated_to_no_opinion(tmp_path: Path) -> None:
    """非严格模式：出错的 Skill 降级为本次无意见，其他 Skill 照常（spec §32）。"""
    _write(tmp_path, _manifest("broken", requires_data=["tier"]), read=("metric.weight_trend",))
    _write(tmp_path, _manifest("fine", requires_data=["tier"]), read=("tier",))
    outputs = Agent(NUTRI).execute(_snapshot_obj(), _select(load_skills(tmp_path), Tier.A), BOOK)
    by_skill = {o.skill: getattr(o, "reason_code", None) for o in outputs}
    assert by_skill["broken@0.1.0"] == "skill_error:UndeclaredDataError"
    assert by_skill["fine@0.1.0"] == "test"


def test_non_prescriptive_with_produces_rejected(tmp_path: Path) -> None:
    _write(tmp_path, _manifest("x", prescriptive=False, produces=["diet_mode"]))
    with pytest.raises(SkillLoadError, match="必须是处方类"):
        load_skills(tmp_path)
