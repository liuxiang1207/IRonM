"""问答链路的意见收集（spec [2026-10-21] 统一链路）：选出 Agent、加载问答类 Skill、并行收集意见。

与计划链路的区别只有两点：Skill 按 answer 链路筛选；产出是意见而不是提案。
其余规则相同：每个 Agent 只读自己声明的数据、只看本次问题，互不可见；
单个 Agent 出错或没有匹配的 Skill 时为“本次无意见”，不影响其他 Agent。
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ironm.agents.base import Agent
from ironm.agents.knowledge import card_opinion, cards_for
from ironm.contracts.agents import OpinionOutput
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.skills import ScenePack, SkillMode
from ironm.contracts.snapshot import Snapshot
from ironm.knowledge.qa_cards import CardBook
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
    cards: CardBook | None = None,
) -> Collected:
    """按给出的 Agent 顺序逐个收集意见（Agent 之间互不可见，顺序只影响排列）。

    Skill 是增强，不是前提：没有匹配的 Skill 时，Agent 回退到查本领域的问答卡。
    cards 为空时才真的“本次无意见”。
    """
    opinions: list[Opinion] = []
    ran: list[AgentId] = []
    silent: list[tuple[AgentId, str]] = []
    used: list[str] = []
    for agent in dict.fromkeys(agents):
        selection = select(
            skills, agent=agent, intent=intent, scene=scene, tier=tier, mode=SkillMode.ANSWER
        )
        if not selection.loaded:
            # 没有 Skill 不等于答不了：回退到 Agent 自带的知识库查询
            fallback = _from_cards(agent, cards, text)
            if fallback is None:
                silent.append((agent, "no_matching_knowledge"))
                continue
            opinions.append(fallback)
            ran.append(agent)
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
            # Skill 明确说“不知道”（如动作库里没这个动作）时，不再用卡片硬凑
            reason = next(
                (o.reason_code for o in outputs if o.kind is OutputKind.NO_OPINION),
                None,
            )
            fallback = None if reason else _from_cards(agent, cards, text)
            if fallback is None:
                silent.append((agent, reason or "no_opinion"))
            else:
                opinions.append(fallback)
                ran.append(agent)
    return Collected(
        opinions=tuple(opinions),
        ran=tuple(ran),
        silent=tuple(silent),
        skills=tuple(dict.fromkeys(used)),
    )


def _from_cards(agent: AgentId, cards: CardBook | None, text: str) -> Opinion | None:
    """Agent 自带的兜底：查本领域的问答卡并转成意见。"""
    if cards is None:
        return None
    found = cards_for(cards, agent, text)
    opinion = card_opinion(agent, found)
    if opinion is None:
        return None
    return Opinion(agent=agent, skill="knowledge@1.0.0", item=opinion)
