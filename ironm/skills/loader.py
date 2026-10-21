"""Skill 加载与选择（spec §28）。

目录结构：<根目录>/<所属 Agent>/<Skill 编号>/SKILL.md 与 skill.py。
SKILL.md 以 YAML 头信息开头（两行 --- 之间），其后是说明正文；skill.py 提供 run(view, context)。
选择只看头信息：所属 Agent、意图、场景包、人群分层同时匹配才加载；
处方类 Skill 在不可给计划的分层一律不加载。
"""

import importlib.util
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from ironm.contracts.agents import AgentOutput
from ironm.contracts.plan import PlanField
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import TierResult
from ironm.contracts.skills import ScenePack, SkillManifest
from ironm.contracts.unknown import is_known
from ironm.core.yamlfile import load_yaml_text
from ironm.rules.loader import RuleBook
from ironm.safety.decision import plan_allowed
from ironm.skills.view import SnapshotView

__all__ = [
    "SKILLS_DIR",
    "LoadedSkill",
    "Selection",
    "SkillLoadError",
    "load_skills",
    "select",
    "validate_skill_rules",
]

SKILLS_DIR = Path(__file__).with_name("library")
_FENCE = "---"


class SkillLoadError(ValueError):
    pass


@dataclass(frozen=True)
class SkillContext:
    """运行一个 Skill 时除快照外能用的东西：Skill 自身、规则簿，以及本次请求。

    text / context_item 只在问答链路里有值；计划链路为空。
    """

    skill: "LoadedSkill"
    rulebook: RuleBook
    text: str = ""
    context_item: str | None = None


type SkillRun = Callable[[SnapshotView, SkillContext], AgentOutput]


@dataclass(frozen=True)
class LoadedSkill:
    manifest: SkillManifest
    body: str  # 说明正文
    path: Path
    run: SkillRun
    # skill.py 中的 RULE_PARAMS：用到的规则编号 → 参数类型；服务启动时逐条校验
    rule_params: Mapping[str, type[BaseModel]] = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    """一个 Agent 本次加载的 Skill（按优先级）与未加载的原因（编号 → 原因代码）。"""

    agent: AgentId
    loaded: tuple[LoadedSkill, ...]
    skipped: tuple[tuple[str, str], ...]

    @property
    def versions(self) -> tuple[str, ...]:
        return tuple(s.manifest.versioned for s in self.loaded)

    @property
    def requires_data(self) -> frozenset[str]:
        return frozenset(k for s in self.loaded for k in s.manifest.requires_data)


def parse_skill_md(path: Path) -> LoadedSkill:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FENCE:
        raise SkillLoadError(f"{path}：SKILL.md 必须以 --- 开头的头信息开始")
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == _FENCE)
    except StopIteration:
        raise SkillLoadError(f"{path}：头信息缺少结束的 ---") from None
    try:
        data = load_yaml_text("\n".join(lines[1:end]))
        manifest = SkillManifest.model_validate_json(json.dumps(data))
    except (ValidationError, yaml.YAMLError, TypeError, ValueError) as e:
        raise SkillLoadError(f"{path}：头信息有误：{e}") from e
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise SkillLoadError(f"{path}：缺少说明正文")
    run, params = _load_module(path.parent)
    return LoadedSkill(manifest=manifest, body=body, path=path, run=run, rule_params=params)


def _load_module(folder: Path) -> tuple[SkillRun, Mapping[str, type[BaseModel]]]:
    """读取 Skill 目录下 skill.py 的 run 函数与 RULE_PARAMS（可省略）。"""
    source = folder / "skill.py"
    if not source.is_file():
        raise SkillLoadError(f"{folder}：缺少 skill.py")
    name = f"ironm_skill_{folder.parent.name}_{folder.name}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise SkillLoadError(f"{source}：无法加载")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise SkillLoadError(f"{source}：缺少 run(view, context) 函数")
    skill_run: SkillRun = run
    params = getattr(module, "RULE_PARAMS", {})
    if not isinstance(params, dict) or not all(
        isinstance(k, str) and isinstance(v, type) and issubclass(v, BaseModel)
        for k, v in params.items()
    ):
        raise SkillLoadError(f"{source}：RULE_PARAMS 应为 {{规则编号: 参数类型}}")
    return skill_run, params


def validate_skill_rules(skills: Iterable[LoadedSkill], rulebook: RuleBook) -> None:
    """服务启动时调用：每个 Skill 用到的规则都存在、参数符合声明的类型，否则报错。"""
    for skill in skills:
        for rule_id, model in skill.rule_params.items():
            try:
                rulebook.get(rule_id).params_as(model)
            except (KeyError, ValueError) as e:
                raise SkillLoadError(f"{skill.manifest.versioned}：规则 {rule_id} 有误：{e}") from e


def load_skills(root: Path = SKILLS_DIR) -> tuple[LoadedSkill, ...]:
    """读取并校验全部 Skill；任何问题都抛 SkillLoadError。"""
    skills: list[LoadedSkill] = []
    seen: set[tuple[AgentId, str]] = set()
    for path in sorted(root.glob("*/*/SKILL.md")) if root.is_dir() else []:
        skill = parse_skill_md(path)
        m = skill.manifest
        if path.parent.name != m.id or path.parent.parent.name != m.owner_agent:
            raise SkillLoadError(f"{path}：目录应为 {m.owner_agent}/{m.id}/SKILL.md")
        if (m.owner_agent, m.id) in seen:
            raise SkillLoadError(f"{path}：{m.owner_agent} 的 Skill 编号重复：{m.id}")
        seen.add((m.owner_agent, m.id))
        blocked = [t for t in m.applies_to.population_tiers if not plan_allowed(t)]
        if m.prescriptive and blocked:
            raise SkillLoadError(f"{path}：处方类 Skill 不能适用于不可给计划的分层：{blocked}")
        skills.append(skill)
    return tuple(skills)


def select(
    skills: Iterable[LoadedSkill],
    *,
    agent: AgentId,
    intent: str,
    scene: ScenePack,
    tier: TierResult,
) -> Selection:
    """按所属 Agent、意图、场景包、人群分层选出本次加载的 Skill。

    Skill 只用于计划链路，问答不加载 Skill（spec [2026-10-21] §12）。同一计划字段只由优先的
    Skill 产出。
    """
    candidates = sorted(
        (s for s in skills if s.manifest.owner_agent == agent),
        key=lambda s: (s.manifest.priority, s.manifest.id),
    )
    loaded: list[LoadedSkill] = []
    skipped: list[tuple[str, str]] = []
    taken: set[PlanField] = set()
    for s in candidates:
        reason = _mismatch(s.manifest, intent, scene, tier)
        if reason is None and taken & set(s.manifest.produces):
            reason = "field_conflict"
        if reason is not None:
            skipped.append((s.manifest.id, reason))
            continue
        loaded.append(s)
        taken |= set(s.manifest.produces)
    return Selection(agent=agent, loaded=tuple(loaded), skipped=tuple(skipped))


def _mismatch(
    m: SkillManifest,
    intent: str,
    scene: ScenePack,
    tier: TierResult,
) -> str | None:
    if intent not in m.triggers.intents:
        return "intent_mismatch"
    if scene not in m.applies_to.scene_packs:
        return "scene_mismatch"
    if not is_known(tier.tier):
        # 分层未知（如还没建档）：处方类一律不加载；非处方类须声明 unknown_tier 才加载
        return None if m.applies_to.unknown_tier and not m.prescriptive else "tier_unknown"
    if m.prescriptive and not plan_allowed(tier.tier):
        return "tier_no_plan"
    if tier.tier not in m.applies_to.population_tiers:
        return "tier_mismatch"
    return None


def versions(selections: Sequence[Selection]) -> tuple[str, ...]:
    """多个 Agent 本次使用的 Skill 版本（写入运行记录）。"""
    return tuple(v for sel in selections for v in sel.versions)
