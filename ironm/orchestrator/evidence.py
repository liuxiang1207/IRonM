"""依据卡生成（spec §31）：每条提案一张；依据在规则表与知识库中都找不到时报错。"""

from collections.abc import Iterable

from ironm.contracts.agents import AgentOutput, Proposal, ProposalOutput
from ironm.contracts.evidence import EvidenceCard, EvidenceSource
from ironm.knowledge.exercises import default_library
from ironm.rules.loader import RuleBook
from ironm.skills.loader import LoadedSkill

__all__ = ["EvidenceError", "build_cards", "resolve_source"]

KNOWLEDGE_SOURCES = {"KB-EXERCISES"}  # 动作库


class EvidenceError(ValueError):
    pass


def resolve_source(ref: str, rulebook: RuleBook) -> EvidenceSource:
    if ref in KNOWLEDGE_SOURCES:
        lib = default_library()
        return EvidenceSource(
            ref=ref,
            title="动作库",
            kind="knowledge",
            citation=f"ironm/knowledge/exercises.yaml 版本 {lib.version}",
            review=lib.review,
        )
    try:
        rule = rulebook.get(ref)
    except KeyError:
        raise EvidenceError(f"提案引用的依据不存在：{ref}") from None
    return EvidenceSource(
        ref=rule.id,
        title=rule.title,
        kind=rule.source.kind.value,
        citation=rule.source.ref,
        review=rule.status.value,
    )


def _card(
    output: ProposalOutput, p: Proposal, skill: LoadedSkill, rulebook: RuleBook
) -> EvidenceCard:
    return EvidenceCard(
        card_id=p.proposal_id,
        field=p.field,
        from_value=p.from_value,
        to_value=p.to_value,
        agent=output.agent,
        skill=output.skill,
        data_used=skill.manifest.requires_data,
        facts=p.reason_facts,
        source=resolve_source(p.rule_id, rulebook),
        expected_effect=p.expected_effect,
        rollback_condition=p.rollback_condition,
        review_date=p.review_date,
    )


def build_cards(
    outputs: Iterable[AgentOutput],
    skills: Iterable[LoadedSkill],
    rulebook: RuleBook,
    *,
    only: Iterable[str] | None = None,
) -> tuple[EvidenceCard, ...]:
    """outputs 中每条提案一张依据卡；only 给出时只为这些提案编号生成（如复核通过的）。"""
    by_version = {s.manifest.versioned: s for s in skills}
    wanted = None if only is None else set(only)
    cards: list[EvidenceCard] = []
    for output in outputs:
        if not isinstance(output, ProposalOutput):
            continue
        skill = by_version.get(output.skill)
        if skill is None:
            raise EvidenceError(f"提案来自未知的 Skill 版本：{output.skill}")
        cards += [
            _card(output, p, skill, rulebook)
            for p in output.items
            if wanted is None or p.proposal_id in wanted
        ]
    return tuple(cards)
