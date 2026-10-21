"""问答链路独立审查后的修正（spec [2026-10-21] §12）。

数字校验覆盖意见的每个部分、中文数字的常见写法；带关注点的闲聊走专业流程；工具只在问题相关时
算作依据；“针对你”的计算在整合后保留；外部资料不整段展示。
"""

import json
from typing import Any

import pytest

from ironm.agents.answering import AgentAnswer
from ironm.agents.tools import ToolInput, exercise_tool
from ironm.contracts.agents import Citation, Opinion
from ironm.contracts.roles import AgentId
from ironm.core.numbers import numbers_in
from ironm.orchestrator.chat import respond
from ironm.orchestrator.compose import compose
from ironm.orchestrator.intent import Intent
from tests.memory_helpers import memory_db
from tests.skills.helpers import FULL_PROFILE
from tests.test_agents_answering import _ask as _agent
from tests.test_agents_answering import _gateway as _agent_gateway
from tests.test_agents_answering import _gen
from tests.test_chat_safety import _deps
from tests.test_qa_chain import (
    CROSS,
    CROSS_INTENT,
    NOW,
    RULES,
    TEXTS,
    TZ,
    Router,
    _agent_reply,
    _ask,
    _gateway,
    _intent,
    _json,
    _text,
)

_A = AgentId
CITE = Citation(kind="knowledge", ref="card:X", label="依据", detail="每周 3 次")


# ---- 数字校验覆盖意见的每个部分 ----


@pytest.mark.parametrize(
    "fields",
    [
        {"conclusion": "每周练 7 次", "reasoning": "依据要求每周 7 次"},  # 数字只出现在推理里
        {"conclusion": "每周练 3 次", "reasoning": "按依据。", "caution": "每天吃 20 克肌酸"},
        {"conclusion": "每周练 3 次", "reasoning": "按依据。", "uncertain": "可能要练 12 周"},
        {"conclusion": "每周练 3 次", "reasoning": "大约要练十二周才见效"},
    ],
)
def test_opinion_numbers_must_come_from_citations_in_every_part(fields: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="没有依据的数字"):
        Opinion(citations=(CITE,), **fields)


def test_opinion_with_sourced_numbers_everywhere_is_fine() -> None:
    Opinion(
        citations=(CITE,),
        conclusion="每周练 3 次",
        reasoning="依据写每周 3 次",
        caution="不要超过 3 次",
    )


def test_model_number_hidden_in_reasoning_is_rejected() -> None:
    gw, _, _ = _agent_gateway(
        _gen(["exercise:sq_goblet"], conclusion="每周练 7 次", reasoning="依据要求每周 7 次")
    )
    result = _agent(_A.EXERCISE_COACH, "高脚杯深蹲怎么做", gw)
    assert result.status == "tool" and result.reason == "model_failed_or_invalid"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("两三组", {"2", "3"}),
        ("三到五组", {"3", "5"}),
        ("3到5组", {"3", "5"}),
        ("百分之八十", {"80"}),
        ("一千五百千卡", {"1500"}),
        ("两万步", {"20000"}),
        ("每周练7日", {"7"}),
        ("七日一次", {"7", "1"}),
        ("3月12日开始", set()),
        ("一日两餐", {"1", "2"}),
        ("一千五千卡", {"1500"}),  # 口语省略末位
        ("两万三步", {"23000"}),
        ("第十二周", set()),
        ("每千克体重", set()),
        ("三千克", {"3"}),
        ("五百千克", {"500"}),
        ("百分之80", {"80"}),
        ("三四十分钟", {"30", "40"}),  # 复核：十位的约数
        ("十几组", {"10"}),
        ("二十来次", {"20"}),
        ("三十多岁", {"30"}),
        ("两千卡", {"2000"}),  # 口语：千卡前的个位数指“几千卡”
        ("一两千卡", {"1000", "2000"}),
        ("零点五克", {"0.5"}),
        ("一点五公斤", {"1.5"}),
        ("五到八成", {"5", "8"}),
        ("1,500千卡", {"1500"}),
    ],
)
def test_numbers_in_common_chinese_forms(text: str, expected: set[str]) -> None:
    assert numbers_in(text) == expected


# ---- 带关注点的闲聊走专业流程 ----


@pytest.mark.parametrize(
    ("text", "caution", "intent"),
    [
        ("你好，我怀孕了还能练吗", "caution_special", "training_knowledge"),
        ("谢谢，我膝盖有点疼", "caution_discomfort", "exercise_technique"),
    ],
)
def test_chitchat_with_concern_takes_professional_path(
    text: str, caution: str, intent: str
) -> None:
    reply = respond("u1", text, _deps([], memory_db()))
    assert reply.layer not in ("template", "chat") and reply.intent == intent
    assert getattr(TEXTS, caution) in "".join(p.text for p in reply.answer)


def test_concern_from_profile_does_not_reroute_greeting() -> None:
    """画像里的健康状况不改变闲聊：有高血压的用户说“你好”仍由通用 Agent 回答。"""
    deps = _deps([], memory_db())
    respond("u1", "我有高血压", deps)
    assert respond("u1", "你好", deps).layer == "template"


# ---- 工具只在问题相关时算作依据 ----


def test_unrelated_tools_are_not_basis() -> None:
    gw, calls, _ = _agent_gateway()
    result = _agent(_A.NUTRITIONIST, "增肌一周练几次", gw, profile=FULL_PROFILE)
    assert result.status == "missing" and result.reason == "no_basis" and calls == []
    assert result.tools == ()


@pytest.mark.parametrize("text", ["我一天要吃多少热量", "我每天蛋白质吃多少"])
def test_related_question_runs_calorie_tool(text: str) -> None:
    result = _agent(_A.NUTRITIONIST, text, profile=FULL_PROFILE)
    assert "calorie_target" in result.tools


def test_bmi_tool_runs_only_when_asked_about_body_size() -> None:
    assert "bmi" in _agent(_A.DATA_ANALYST, "我的BMI正常吗", profile=FULL_PROFILE).tools
    assert "bmi" not in _agent(_A.DATA_ANALYST, "怎么坚持训练", profile=FULL_PROFILE).tools


def test_model_saying_no_basis_counts_as_missing() -> None:
    gw, _, _ = _agent_gateway(_gen(["card:QA-CREATINE"], conclusion="没有找到可靠依据"))
    result = _agent(_A.NUTRITIONIST, "肌酸要怎么吃", gw)
    assert result.status == "missing" and result.reason == "no_basis"


def test_related_domain_with_full_profile_does_not_trigger_partial_notice() -> None:
    intent = _intent(
        Intent.TRAINING_KNOWLEDGE, domains=(Intent.TRAINING_KNOWLEDGE, Intent.NUTRITION)
    )
    result = _ask("增肌一周练几次比较好", intent, profile=FULL_PROFILE)
    assert not any(p.kind == "notice" for p in result.parts)
    assert dict(result.record.agents)[_A.NUTRITIONIST] == "missing"


# ---- “针对你”在整合后保留；运行记录记下用到的计算 ----


def test_for_you_kept_after_model_merge_and_recorded() -> None:
    def agent(payload: dict[str, Any]) -> dict[str, object]:
        if payload["facts"]:
            fact = payload["facts"][0]
            return _json(
                {"conclusion": fact["label"], "reasoning": "按计算。", "cites": [fact["id"]]}
            )
        return _agent_reply(payload)

    gw, _ = _gateway(Router(agent=agent, compose=lambda _: _json({"conclusion": "两边的要点。"})))
    result = _ask(CROSS, CROSS_INTENT, gw, profile=FULL_PROFILE)
    assert result.record.composed_by_model
    for_you = [p.text for p in result.parts if p.kind == "for_you"]
    assert any("每日蛋白质" in t or "每日热量" in t for t in for_you)
    assert {"calc:kcal_target", "calc:protein_target"} & set(result.record.personal)


# ---- 页面条目找不到时按问题里的动作名 ----


def test_unknown_page_item_falls_back_to_named_exercise() -> None:
    inp = ToolInput(
        text="高脚杯深蹲怎么做",
        profile={},
        context_item="today.exercise:not_in_library",
        rulebook=RULES,
        as_of=NOW,
        tz=TZ,
    )
    result = exercise_tool(inp)
    assert result is not None and result.extra["exercise_id"] == "sq_goblet"


# ---- 外部资料不整段展示 ----


def test_external_passage_detail_is_shortened_for_users() -> None:
    long_text = "外部资料原文。" * 40
    external = Citation(kind="knowledge", ref="guide-1:3", label="某指南", detail=long_text)
    ours = Citation(kind="knowledge", ref="card:QA-PROTEIN", label="问答卡", detail=long_text)
    op = Opinion(conclusion="多吃蛋白", citations=(external, ours), reasoning="按依据。")
    out = compose(
        "q",
        [AgentAnswer(_A.NUTRITIONIST, "model", op, (), ())],
        cautions=(),
        partial_notice="部分",
        gateway=None,
        request_id="r",
    )
    by_ref = {c["ref"]: c for c in out.citations}
    assert len(by_ref["guide-1:3"]["detail"]) <= 80
    assert by_ref["card:QA-PROTEIN"]["detail"] == long_text
    assert json.dumps(out.citations, ensure_ascii=False).count(long_text) == 1


def test_cross_question_text_for_reference() -> None:
    assert "蛋白质" in CROSS and "没有找到可靠依据" not in _text(_ask(CROSS, CROSS_INTENT))


# ---- 复核后补充 ----


def test_card_fallback_keeps_numbers_in_card_caution() -> None:
    """卡的注意 / 不确定里的数字也要能通过意见层的数字校验，否则这张卡兜底时会失效。"""
    from ironm.agents.knowledge import card_opinion
    from ironm.knowledge.qa_cards import default_cards

    cards = default_cards()
    failing = []
    for card in cards.cards:
        try:
            card_opinion(card.agent, [card])
        except ValueError:
            failing.append(card.id)
    assert failing == []


def test_handoff_keeps_concern_caution() -> None:
    """饮食页上说膝盖疼：转回管家对话时，身体不适的提醒仍在。"""
    from ironm.orchestrator.pages import PageContext

    reply = respond(
        "u1", "谢谢，我膝盖有点疼", _deps([], memory_db()), page="diet",
        page_context=PageContext(page="diet"),
    )  # fmt: skip
    assert reply.layer == "handoff"
    assert TEXTS.caution_discomfort in "".join(p.text for p in reply.answer)


def test_partly_useful_answer_mentioning_no_basis_is_kept() -> None:
    gw, _, _ = _agent_gateway(
        _gen(
            ["card:QA-CREATINE"],
            conclusion="肌酸每天 3–5 克即可；长期大剂量的影响没有找到可靠依据。",
        )
    )
    assert _agent(_A.NUTRITIONIST, "肌酸要怎么吃", gw).status == "model"


def test_personal_uses_one_vocabulary_on_both_paths() -> None:
    card_path = _ask(
        "身高165，体重45公斤，我太瘦了吗", _intent(Intent.PERSONAL_DATA), retriever=None
    )
    assert card_path.record.personal == ("calc:bmi",)


def test_knowledge_folder_cannot_use_reserved_prefixes(tmp_path: Any) -> None:
    from ironm.knowledge.kb import KnowledgeLoadError, load_passages

    (tmp_path / "a.yaml").write_text(
        "passages:\n"
        "  - passage_id: card:FAKE\n    domain: nutritionist\n    title: t\n    text: x\n"
        "    source: s\n    level: guideline\n    review: approved\n",
        encoding="utf-8",
    )
    with pytest.raises(KnowledgeLoadError):
        load_passages(tmp_path)


def test_model_cannot_copy_long_external_passage_verbatim() -> None:
    from ironm.agents.answering import copies_external

    passage = "这是一段外部指南的原文，" * 10
    assert copies_external("按指南：" + passage[:60], [passage])
    assert not copies_external("每周练 3 次", [passage])


def test_agent_answer_copying_external_passage_falls_back() -> None:
    from ironm.agents.answering import answer_as
    from ironm.knowledge.exercises import default_library
    from ironm.knowledge.kb import KnowledgeBase, Passage, Retriever
    from ironm.knowledge.qa_cards import default_cards

    text = "电解质饮料适合长时间大量出汗的运动，日常短时间训练喝白水就够，不必额外补充电解质饮料。"
    guide = Passage.model_validate_json(
        json.dumps(
            {"passage_id": "guide-x:1", "domain": "nutritionist", "title": "补水",
             "text": text, "source": "某指南", "level": "guideline", "review": "approved"},
            ensure_ascii=False,
        )
    )  # fmt: skip
    retriever = Retriever(KnowledgeBase.from_sources(default_cards(), default_library(), [guide]))

    def ask(conclusion: str) -> AgentAnswer:
        gw, _, _ = _agent_gateway(_gen(["guide-x:1"], conclusion=conclusion))
        return answer_as(
            _A.NUTRITIONIST, text="训练要喝电解质饮料吗", profile={}, context_item=None,
            retriever=retriever, cards=default_cards(), rulebook=RULES, gateway=gw,
            request_id="r", as_of=NOW, tz=TZ,
        )  # fmt: skip

    assert ask("短时间训练喝白水就够。").status == "model"
    assert ask(text).status != "model"
