"""Agent 的基础问答能力（spec [2026-10-21] 统一链路）：不装 Skill 也能查知识库回答。

分工：
- **Agent 自带**：查本领域的问答卡（审定过的标准答案）。任何 Agent 都能做，
  不需要加载任何 Skill。
- **Skill 才有**：更专门的做法，例如动作讲解（查动作库并结合用户限制给降阶）、
  能量目标（按公式算个人数值）。有 Skill 时优先用 Skill。

所以「没有匹配的 Skill」不等于「答不了」，而是「回退到查知识库」。
"""

from collections.abc import Iterable

from ironm.contracts.agents import Citation, Opinion
from ironm.contracts.roles import AgentId
from ironm.knowledge.qa_cards import CardBook, QACard

__all__ = ["card_opinion", "cards_for"]


def cards_for(cards: CardBook, agent: AgentId, text: str, *, limit: int = 2) -> tuple[QACard, ...]:
    """本 Agent 名下与这句话相关的问答卡，按具体程度排序。"""
    found = [(c, s) for c, s in cards.matches(text) if c.agent == agent]
    return tuple(c for c, _ in found[:limit])


def card_opinion(agent: AgentId, found: Iterable[QACard]) -> Opinion | None:
    """把问答卡转成一份意见：结论、做法、注意都来自卡，依据是卡的编号与出处。

    多张卡时取最具体的一张作为结论，其余作为补充，避免把不相干的答案拼在一起。
    """
    chosen = list(found)
    if not chosen:
        return None
    best, *rest = chosen
    # 卡里的数字本身就是登记过的，但要出现在依据里才通得过意见层的数字校验
    citations = [
        Citation(
            kind="knowledge",
            ref=card.id,
            label=f"问答卡 {card.id}：{card.conclusion}",
            # 注意与不确定里的数字也要登记，否则这张卡兜底时会被数字校验挡掉
            detail="；".join(
                v for v in (card.how, card.why, card.caution, card.uncertain, *card.sources) if v
            ),
        )
        for card in chosen
    ]
    how = [best.how, *(c.how for c in rest if c.how)]
    caution = [best.caution, *(c.caution for c in rest if c.caution)]
    return Opinion(
        conclusion=best.conclusion,
        citations=tuple(citations),
        reasoning=f"按审定过的问答卡 {best.id} 回答（依据见下）。",
        how="；".join(h for h in how if h) or None,
        caution="；".join(c for c in caution if c) or None,
        uncertain=best.uncertain,
    )
