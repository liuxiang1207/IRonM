"""Skill 加载与选择（spec §28）。

目录结构：<根目录>/<所属 Agent>/<Skill 编号>/SKILL.md。
SKILL.md 以 YAML 头信息开头（两行 --- 之间），其后是说明正文。
选择只看头信息：所属 Agent、意图、场景包、人群分层同时匹配才加载；
处方类 Skill 在不可给计划的分层一律不加载。
"""

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from ironm.contracts.plan import PlanField
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import TierResult
from ironm.contracts.skills import ScenePack, SkillManifest
from ironm.contracts.unknown import is_known
from ironm.core.yamlfile import load_yaml_text
from ironm.safety.decision import plan_allowed

__all__ = ["SKILLS_DIR", "LoadedSkill", "Selection", "SkillLoadError", "load_skills", "select"]

SKILLS_DIR = Path(__file__).with_name("library")
_FENCE = "---"


class SkillLoadError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedSkill:
    manifest: SkillManifest
    body: str  # 说明正文
    path: Path


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
    return LoadedSkill(manifest=manifest, body=body, path=path)


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
    """按所属 Agent、意图、场景包、人群分层选出本次加载的 Skill；同一字段只由优先的 Skill 产出。"""
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


def _mismatch(m: SkillManifest, intent: str, scene: ScenePack, tier: TierResult) -> str | None:
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
