"""问答链路的意见路径（spec [2026-10-21] 统一链路）：问答类 Skill 产出意见后优先于查卡。"""

from datetime import UTC, datetime

from ironm.contracts.skills import ScenePack
from ironm.core.clock import FixedClock
from ironm.core.ids import SeededIdGenerator
from ironm.knowledge.qa_cards import default_cards
from ironm.memory.profile import ProfileService
from ironm.memory.profile import load_terms as load_profile_terms
from ironm.orchestrator.chat import ChatDeps, ChatReply, load_answer_texts, respond
from ironm.orchestrator.intent import load_intent_book
from ironm.rules.loader import RULES_DIR, load_rules
from ironm.safety.decision import default_kit
from ironm.skills.loader import load_skills
from tests.memory_helpers import memory_db
from tests.skills.helpers import tier_result

NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)
SKILLS = load_skills()


def _deps(*, with_skills: bool) -> ChatDeps:
    memory = memory_db()
    ids = SeededIdGenerator(7)
    deps = ChatDeps(
        kit=default_kit(),
        clock=FixedClock(NOW),
        ids=ids,
        record_flag=lambda user, req, flag: None,
        record_run=lambda user, record: None,
        intents=load_intent_book(),
        cards=default_cards(),
        rulebook=load_rules(RULES_DIR),
        texts=load_answer_texts(),
        memory=memory,
        profiles=ProfileService(
            memory, terms=load_profile_terms(), ids=ids, now=FixedClock(NOW).now
        ),
    )
    if not with_skills:
        return deps
    return ChatDeps(
        **{
            **deps.__dict__,
            "skills": SKILLS,
            "scene": ScenePack.FAT_LOSS,
            "tier": tier_result(__import__("ironm.contracts.screening", fromlist=["Tier"]).Tier.A),
        }
    )


def _ask(deps: ChatDeps, text: str) -> ChatReply:
    return respond("u1", text, deps)


def test_opinion_answer_carries_citations() -> None:
    reply = _ask(_deps(with_skills=True), "高脚杯深蹲怎么做")
    assert reply.layer == "l1"
    assert reply.citations, "意见路径应带出依据"
    assert any(c["kind"] == "knowledge" for c in reply.citations)
    assert any("高脚杯深蹲" in p.text for p in reply.answer)


def test_without_skills_falls_back_to_cards() -> None:
    reply = _ask(_deps(with_skills=False), "高脚杯深蹲怎么做")
    assert reply.citations == ()


def test_unknown_exercise_marks_silent_not_fabricated() -> None:
    reply = _ask(_deps(with_skills=True), "那个转体抛接动作怎么做")
    assert any("unknown_exercise" in s for s in reply.silent)


def test_agent_without_skill_still_answers_from_knowledge_base() -> None:
    """没有问答类 Skill 的领域，Agent 直接查本领域的问答卡，不该是“无意见”。"""
    reply = _ask(_deps(with_skills=True), "蛋白质一天吃多少")
    assert reply.layer == "l1"
    assert reply.citations, "查知识库的回答也要带依据"
    assert any(c["kind"] == "knowledge" for c in reply.citations)


def test_skill_reason_is_respected_over_cards() -> None:
    """Skill 明确说不知道时（动作库无此动作），不用不相干的卡硬凑。"""
    reply = _ask(_deps(with_skills=True), "那个转体抛接动作怎么做")
    assert any("unknown_exercise" in s for s in reply.silent)
