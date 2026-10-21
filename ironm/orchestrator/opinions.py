"""问答链路的意见收集（spec [2026-10-21] 统一链路）：选出 Agent、加载问答类 Skill、并行收集意见。

与计划链路的区别只有两点：Skill 按 answer 链路筛选；产出是意见而不是提案。
其余规则相同：每个 Agent 只读自己声明的数据、只看本次问题，互不可见；
单个 Agent 出错或没有匹配的 Skill 时为“本次无意见”，不影响其他 Agent。
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ironm.agents.base import Agent
from ironm.contracts.agents import OpinionOutput
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.skills import ScenePack, SkillMode
from ironm.contracts.snapshot import Snapshot
from ironm.rules.loader import RuleBook
from ironm.skills.loader import LoadedSkill, select

__all__ = ["Collected", "Opinion", "collect_opinions"]


@dataclass(frozen=True)
class Opinion:
    """一个 Agent 的一条意见，带上它是谁说的。"""

    agent: AgentId
    skill: str
    item: Any  # contracts.agents.Opinion


@dataclass(frozen=True)
class Collected:
    opinions: tuple[Opinion, ...]
    ran: tuple[AgentId, ...]  # 实际产出意见的 Agent
    silent: tuple[tuple[AgentId, str], ...]  # 没有意见的 Agent 与原因
    skills: tuple[str, ...]  # 本次加载的 Skill 版本


def collect_opinions(
    *,
    agents: Iterable[AgentId],
    snapshot: Snapshot,
    skills: Iterable[LoadedSkill],
    rulebook: RuleBook,
    intent: str,
    scene: ScenePack,
    tier: Any,
    text: str,
    context_item: str | None = None,
) -> Collected:
    """按给出的 Agent 顺序逐个收集意见（Agent 之间互不可见，顺序只影响排列）。"""
    opinions: list[Opinion] = []
    ran: list[AgentId] = []
    silent: list[tuple[AgentId, str]] = []
    used: list[str] = []
    for agent in dict.fromkeys(agents):
        selection = select(
            skills, agent=agent, intent=intent, scene=scene, tier=tier, mode=SkillMode.ANSWER
        )
        if not selection.loaded:
            silent.append((agent, "no_matching_skill"))
            continue
        used.extend(selection.versions)
        outputs = Agent(agent).execute(
            snapshot, selection, rulebook, text=text, context_item=context_item
        )
        got = [
            Opinion(agent=agent, skill=o.skill, item=item)
            for o in outputs
            if isinstance(o, OpinionOutput)
            for item in o.items
        ]
        if got:
            opinions.extend(got)
            ran.append(agent)
        else:
            reason = next(
                (o.reason_code for o in outputs if o.kind is OutputKind.NO_OPINION),
                None,
            )
            silent.append((agent, reason or "no_opinion"))
    return Collected(
        opinions=tuple(opinions),
        ran=tuple(ran),
        silent=tuple(silent),
        skills=tuple(dict.fromkeys(used)),
    )
