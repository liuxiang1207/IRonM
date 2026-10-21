"""知识库与检索（spec [2026-10-21] §12）：按领域召回、融合、重排、门槛。"""

from collections.abc import Sequence
from pathlib import Path

import pytest

from ironm.contracts.roles import AgentId
from ironm.knowledge.exercises import default_library
from ironm.knowledge.kb import (
    KnowledgeBase,
    KnowledgeLoadError,
    Passage,
    Retriever,
    load_passages,
)
from ironm.knowledge.qa_cards import default_cards

_A = AgentId
KB = KnowledgeBase.from_sources(default_cards(), default_library())


def _ids(query: str, domain: AgentId, retriever: Retriever | None = None) -> list[str]:
    return [r.passage.passage_id for r in (retriever or Retriever(KB)).search(query, domain)]


@pytest.mark.parametrize(
    ("query", "domain", "expected"),
    [
        ("肌酸要怎么吃", _A.NUTRITIONIST, "card:QA-CREATINE"),
        ("高脚杯深蹲怎么做", _A.EXERCISE_COACH, "exercise:sq_goblet"),
        ("增肌一周练几次", _A.TRAINING_PLANNER, "card:QA-STRENGTH-FREQUENCY"),
        ("先有氧还是先力量", _A.TRAINING_PLANNER, "card:QA-CARDIO-ORDER"),
        ("点外卖怎么选", _A.NUTRITIONIST, "card:QA-TAKEOUT"),
    ],
)
def test_recall_and_rerank_find_the_right_passage(
    query: str, domain: AgentId, expected: str
) -> None:
    assert _ids(query, domain)[0] == expected


def test_only_the_agents_own_domain_is_searched() -> None:
    assert _ids("肌酸要怎么吃", _A.TRAINING_PLANNER) == []


def test_irrelevant_question_returns_nothing() -> None:
    assert _ids("你好", _A.NUTRITIONIST) == []
    assert _ids("动感单车和椭圆机哪个好", _A.TRAINING_PLANNER) == []


def test_keep_k_limits_results() -> None:
    assert len(_ids("一天要吃多少蛋白质", _A.NUTRITIONIST, Retriever(KB, keep_k=2))) == 2


def test_unreviewed_passages_never_returned(tmp_path: Path) -> None:
    extra = Passage(
        passage_id="note:sleep",
        domain=_A.RECOVERY,
        title="睡不好怎么办",
        text="最近睡不好时，固定起床时间。",
        source="自写",
        level="expert",
        review="pending_review",
    )
    kb = KnowledgeBase([*KB.passages, extra])
    assert "note:sleep" not in _ids("最近睡不好", _A.RECOVERY, Retriever(kb))
    approved = extra.model_copy(update={"review": "approved"})
    kb = KnowledgeBase([*KB.passages, approved])
    assert _ids("最近睡不好", _A.RECOVERY, Retriever(kb))[0] == "note:sleep"


class _Embedder:
    """测试用向量：说到睡眠（“睡”“醒”）的文本指向同一个方向。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if ("睡" in t or "醒" in t) else [0.0, 1.0] for t in texts]


class _AllPass:
    def score(self, query: str, passages: Sequence[Passage]) -> list[float]:
        return [1.0 for _ in passages]


def test_vector_recall_finds_what_keywords_miss() -> None:
    """“夜里老醒”与睡眠卡没有共同的两字片段，关键词召回不到；向量召回补上。"""
    assert _ids("夜里老醒", _A.RECOVERY, Retriever(KB, reranker=_AllPass())) == []
    found = _ids("夜里老醒", _A.RECOVERY, Retriever(KB, embedder=_Embedder(), reranker=_AllPass()))
    assert "card:QA-SLEEP" in found


def test_min_score_threshold_applies() -> None:
    assert _ids("一天要吃多少蛋白质", _A.NUTRITIONIST, Retriever(KB, min_score=0.99)) == []


def test_load_passages_from_folder(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(
        "passages:\n  - {passage_id: g1, domain: recovery, title: 睡眠, text: 成人每晚睡够, "
        "source: 指南, level: guideline, review: approved}\n",
        encoding="utf-8",
    )
    [p] = load_passages(tmp_path)
    assert p.passage_id == "g1" and p.domain is _A.RECOVERY
    assert load_passages(tmp_path / "missing") == [] and load_passages(None) == []


def _one(domain: str, level: str) -> str:
    return (
        f"passages:\n  - {{passage_id: g1, domain: {domain}, title: x, text: y, source: z, "
        f"level: {level}, review: approved}}\n"
    )


@pytest.mark.parametrize(
    "body",
    [
        "passages: {}\n",
        _one("dancer", "guideline"),  # 没有这个领域
        _one("recovery", "rumor"),  # 没有这个证据级别
    ],
)
def test_bad_passages_rejected(tmp_path: Path, body: str) -> None:
    (tmp_path / "bad.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(KnowledgeLoadError):
        load_passages(tmp_path)


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(KnowledgeLoadError, match="重复"):
        KnowledgeBase([KB.passages[0], KB.passages[0]])


def test_pinned_cards_come_first_once_and_bypass_threshold() -> None:
    """问答卡触发词命中的卡排在最前、不重复；只认本领域的段落。"""
    r = Retriever(KB)
    query = "增肌一周练几次，蛋白质一天吃多少"  # 两个领域混在一句里，片段覆盖被稀释
    assert "card:QA-STRENGTH-FREQUENCY" not in _ids(query, _A.TRAINING_PLANNER, r)
    pinned = ["card:QA-STRENGTH-FREQUENCY", "card:QA-PROTEIN"]
    got = [x.passage.passage_id for x in r.search(query, _A.TRAINING_PLANNER, pinned=pinned)]
    assert got[0] == "card:QA-STRENGTH-FREQUENCY" and "card:QA-PROTEIN" not in got
    both = ["card:QA-PROTEIN-POWDER", "card:QA-PROTEIN"]
    got = [x.passage.passage_id for x in r.search(query, _A.NUTRITIONIST, pinned=both)]
    assert got[:2] == both and len(got) == len(set(got))
