"""问答召回独立审查后的修正（spec [2026-10-21] §13）。"""

import threading
from collections.abc import Sequence
from typing import Any

import pytest

from ironm.contracts.roles import AgentId
from ironm.knowledge.exercises import default_library
from ironm.knowledge.kb import KnowledgeBase, Retriever
from ironm.knowledge.qa_cards import default_cards
from ironm.orchestrator.chat import respond
from ironm.orchestrator.intent import Intent, load_intent_book
from ironm.orchestrator.rewrite import rewrite_queries
from tests.memory_helpers import memory_db
from tests.test_chat_safety import _deps
from tests.test_qa_chain import (
    CROSS,
    CROSS_INTENT,
    RAPID,
    Router,
    _agent_reply,
    _ask,
    _gateway,
    _intent,
    _json,
)

_A = AgentId
BOOK = load_intent_book()
CARDS = default_cards()
KB = KnowledgeBase.from_sources(CARDS, default_library())


def _safety_says_no_basis(payload: dict[str, Any]) -> dict[str, object]:
    first = (payload["passages"] or payload["facts"])[0]
    if payload["role"].startswith("安全官"):
        no_basis = {"conclusion": "没有找到可靠依据", "reasoning": "无关。"}
        return _json({**no_basis, "cites": [first["id"]]})
    return _agent_reply(payload)


# ---- 审查 1：安全官的模型说“没有依据”时，命中触发词的安全类卡照样给出 ----


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("一个月减10斤，蛋白质一天吃多少", Intent.NUTRITION),
        (RAPID, Intent.NUTRITION),
    ],
)
def test_safety_card_survives_model_no_basis(text: str, intent: Intent) -> None:
    gw, _ = _gateway(Router(agent=_safety_says_no_basis))
    result = _ask(text, _intent(intent), gw)
    assert dict(result.record.agents)[_A.SAFETY_OFFICER] == "card"
    assert "QA-RAPID-LOSS" in result.record.card_ids
    assert "answer:safety_officer:card" in result.record.degraded_steps


# ---- 审查 2、5：改写不能改变问题的性质 ----


def test_rewrite_cannot_turn_safety_question_into_chitchat() -> None:
    router = Router(
        rewrite=lambda _: _json({"main": "你好呀", "variants": []}),
        agent=_agent_reply,
        # 意图模型只在看到“你好呀”时才判为闲聊：改写若被采用并拿去识别意图，就会降成闲聊
        intent=lambda p: _json(
            {
                "intent": "chitchat" if p["message"] == "你好呀" else "nutrition",
                "concerns": [],
                "confidence": 0.95,
            }
        ),
    )
    deps = _deps([], memory_db())
    deps = type(deps)(**{**deps.__dict__, "gateway": _gateway(router)[0]})
    reply = respond("u1", RAPID, deps)
    assert reply.layer not in ("template", "chat")
    assert "安全官的要点" in "".join(p.text for p in reply.answer)  # 走了专业流程，安全官作答


@pytest.mark.parametrize(
    ("text", "main", "reason"),
    [
        ("两周后婚礼，怎么快速掉秤", "婚礼前的饮食安排", "safety_cards"),  # 丢了安全类卡的触发词
        ("肌酸要怎么吃", "我怀孕了，肌酸要怎么吃", "concerns"),  # 加了用户没说的情况
        ("增肌一周练几次", "增肌一周练几次，一天只吃一顿可以吗", "safety_cards"),  # 加了触发词
    ],
)
def test_rewrite_rejected_when_it_changes_safety_meaning(text: str, main: str, reason: str) -> None:
    gw, _ = _gateway(Router(rewrite=lambda _: _json({"main": main, "variants": []})))
    out = rewrite_queries(
        text, history=(), context_item=None, book=BOOK, cards=CARDS, gateway=gw,
        request_id="r", max_queries=4,
    )  # fmt: skip
    assert out.status == "rejected" and out.reason == reason and out.queries == (text,)


def test_intent_never_downgraded_by_rewrite_but_can_be_clarified() -> None:
    """原话含糊（“那这个呢”）时，改写补全后的主查询可以把追问升为专业问题；反之不行。"""
    seen: list[str] = []

    def intent(payload: dict[str, Any]) -> dict[str, object]:
        seen.append(payload["message"])
        name = "exercise_technique" if "深蹲" in payload["message"] else "unknown"
        return _json({"intent": name, "concerns": [], "confidence": 0.9})

    router = Router(
        rewrite=lambda _: _json({"main": "高脚杯深蹲怎么做", "variants": []}),
        agent=_agent_reply,
        intent=intent,
    )
    deps = _deps([], memory_db())
    deps = type(deps)(**{**deps.__dict__, "gateway": _gateway(router)[0]})
    reply = respond("u1", "那这个呢", deps)
    assert seen == ["那这个呢", "高脚杯深蹲怎么做"]  # 先按原话，含糊时再看补全后的主查询
    assert reply.intent == "exercise_technique"


# ---- 审查 3：安全类卡只按用户原话匹配 ----


def test_variants_cannot_pull_in_unrelated_safety_card() -> None:
    gw, _ = _gateway(Router(agent=_safety_says_no_basis))
    queries = ("增肌一周练几次", "增肌和减脂一周练几次", CROSS)
    result = _ask(CROSS, CROSS_INTENT, gw, queries=queries)
    assert dict(result.record.agents)[_A.SAFETY_OFFICER] == "missing"
    assert "QA-RAPID-LOSS" not in result.record.card_ids


# ---- 审查 4：查询向量每次请求只算一次；失败有冷却并记入降级 ----


class _CountingEmbedder:
    def __init__(self, fail_queries: bool = False) -> None:
        self.calls: list[int] = []
        self.fail_queries = fail_queries
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            self.calls.append(len(texts))
        if self.fail_queries and len(texts) < 10:
            raise TimeoutError
        return [[1.0, 0.0] if "睡" in t else [0.0, 1.0] for t in texts]


def test_query_vectors_computed_once_per_request() -> None:
    emb = _CountingEmbedder()
    retriever = Retriever(KB, embedder=emb)
    result = _ask(CROSS, CROSS_INTENT, retriever=retriever, embedder=emb,
                  queries=("增肌一周练几次", "蛋白质一天吃多少", CROSS))  # fmt: skip
    query_calls = [n for n in emb.calls if n < 10]
    assert query_calls == [3]  # 几个 Agent 共用同一组查询向量
    assert "vector:unavailable" not in result.record.degraded_steps


def test_query_vector_failure_cooldown_and_recorded() -> None:
    emb = _CountingEmbedder(fail_queries=True)
    retriever = Retriever(KB, embedder=emb)
    first = _ask(CROSS, CROSS_INTENT, retriever=retriever, embedder=emb)
    assert "vector:unavailable" in first.record.degraded_steps
    before = len(emb.calls)
    _ask(CROSS, CROSS_INTENT, retriever=retriever, embedder=emb)
    assert len(emb.calls) == before  # 冷却期内不再调用向量服务


class _WrongDims:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] if len(texts) < 10 else [1.0, 0.0] for _ in texts]


def test_vector_dimension_mismatch_skips_route_not_answer() -> None:
    r = Retriever(KB, embedder=_WrongDims())
    got = r.retrieve(["肌酸要怎么吃"], _A.NUTRITIONIST)
    assert got.vector == () and got.vector_status == "unavailable"
    assert got.passages and got.passages[0].passage.passage_id == "card:QA-CREATINE"


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}],  # 重复
        [{"index": 0, "embedding": [1.0]}, {"index": 5, "embedding": [2.0]}],  # 越界
        [{"index": True, "embedding": [1.0]}, {"index": 1, "embedding": [2.0]}],  # 布尔
        [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [2.0, 3.0]}],  # 维度不一
    ],
)
def test_embedding_response_indices_and_dims_validated(data: list[dict[str, Any]]) -> None:
    from ironm.llm.gateway import _vectors
    from ironm.llm.validate import OutputRejected

    with pytest.raises(OutputRejected):
        _vectors({"data": data}, 2)


def test_passage_vectors_do_not_block_other_requests() -> None:
    """第一次计算段落向量时，其他请求不等待，先只用关键词一路。"""
    started, release = threading.Event(), threading.Event()

    class _Slow:
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            if len(texts) > 10:
                started.set()
                release.wait(5)
            return [[0.0, 1.0] for _ in texts]

    r = Retriever(KB, embedder=_Slow())
    t = threading.Thread(target=lambda: r.retrieve(["肌酸要怎么吃"], _A.NUTRITIONIST))
    t.start()
    assert started.wait(5)
    other = r.retrieve(["肌酸要怎么吃"], _A.NUTRITIONIST)  # 不被阻塞
    assert other.vector_status == "unavailable" and other.passages
    release.set()
    t.join(5)


# ---- 小问题 ----


def test_coverage_scorer_keeps_content_words_containing_particles() -> None:
    r = Retriever(KB)
    got = r.retrieve(["有氧和力量先做哪个"], _A.TRAINING_PLANNER)
    assert got.passages and got.passages[0].passage.passage_id == "card:QA-CARDIO-ORDER"


def test_partial_notice_when_primary_failed_and_only_safety_spoke() -> None:
    from ironm.agents.answering import AgentAnswer
    from ironm.contracts.agents import Citation, Opinion
    from ironm.orchestrator.compose import compose

    op = Opinion(
        conclusion="不要极端节食",
        citations=(Citation(kind="knowledge", ref="card:QA-RAPID-LOSS", label="卡"),),
        reasoning="按依据。",
    )
    failed = AgentAnswer(_A.NUTRITIONIST, "missing", None, (), (), "agent_error:RuntimeError")
    quiet = AgentAnswer(_A.NUTRITIONIST, "missing", None, (), (), "no_basis")
    safety = AgentAnswer(_A.SAFETY_OFFICER, "model", op, (), ())
    assert compose("q", [failed, safety], cautions=(), partial_notice="部分", gateway=None,
                   request_id="r").partial  # fmt: skip
    assert not compose("q", [quiet, safety], cautions=(), partial_notice="部分", gateway=None,
                       request_id="r").partial  # fmt: skip


def test_rewrite_cannot_turn_vague_question_into_chitchat() -> None:
    """原话含糊（该追问）时，改写后的主查询若是闲聊，不采用：照样追问。"""
    router = Router(
        rewrite=lambda _: _json({"main": "你好呀", "variants": []}),
        intent=lambda p: _json(
            {
                "intent": "chitchat" if p["message"] == "你好呀" else "unknown",
                "concerns": [],
                "confidence": 0.3,
            }
        ),
    )
    deps = _deps([], memory_db())
    deps = type(deps)(**{**deps.__dict__, "gateway": _gateway(router)[0]})
    assert respond("u1", "我想问问上次说的那个事情现在怎么样了", deps).layer == "clarify"
