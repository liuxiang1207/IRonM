"""问答召回与大模型优先（spec [2026-10-21] §13）：多查询改写、两路召回、大模型重排、
问答卡最低优先级、安全官以模型作答、降级记录。"""

import json
from typing import Any

import pytest

from ironm.agents.rerank import ModelReranker
from ironm.contracts.roles import AgentId
from ironm.knowledge.exercises import default_library
from ironm.knowledge.kb import KnowledgeBase, Passage, Retriever
from ironm.knowledge.qa_cards import default_cards
from ironm.llm.gateway import GatewayEmbedder
from ironm.orchestrator.chat import respond
from ironm.orchestrator.intent import Intent, load_intent_book
from ironm.orchestrator.rewrite import rewrite_queries
from tests.llm.test_gateway import _reply
from tests.memory_helpers import memory_db
from tests.test_chat_safety import _deps
from tests.test_qa_chain import (
    CROSS,
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


def _rewrite(text: str, reply: Any, history: tuple[str, ...] = (), max_queries: int = 4) -> Any:
    router = Router(rewrite=lambda _: reply) if reply is not None else None
    gw = _gateway(router)[0] if router is not None else None
    return rewrite_queries(
        text, history=history, context_item=None, book=BOOK, gateway=gw, request_id="r",
        max_queries=max_queries,
    )  # fmt: skip


# ---- 多查询改写 ----


def test_rewrite_splits_and_keeps_original_last() -> None:
    out = _rewrite(
        CROSS,
        _json(
            {"main": "增肌训练频率与蛋白质摄入", "variants": ["增肌一周练几次", "蛋白质一天吃多少"]}
        ),
    )
    assert out.status == "model"
    assert out.queries == ("增肌训练频率与蛋白质摄入", "增肌一周练几次", "蛋白质一天吃多少", CROSS)


def test_rewrite_respects_max_queries() -> None:
    reply = _json({"main": "主查询", "variants": ["说法一", "说法二", "说法三", "说法四"]})
    out = _rewrite("肌酸要怎么吃", reply, max_queries=3)
    assert len(out.queries) == 3 and out.queries[-1] == "肌酸要怎么吃"


def test_rewrite_fills_reference_from_history() -> None:
    reply = _json({"main": "高脚杯深蹲膝盖疼怎么办", "variants": []})
    out = _rewrite("那这个呢，膝盖疼", reply, history=("高脚杯深蹲怎么做",))
    assert out.queries[0] == "高脚杯深蹲膝盖疼怎么办"


@pytest.mark.parametrize(
    ("text", "reply", "reason"),
    [
        ("肌酸要怎么吃", _json({"main": "肌酸每天吃 5 克可以吗", "variants": []}), "numbers"),
        ("我怀孕了，还能练深蹲吗", _json({"main": "深蹲怎么练", "variants": []}), "concerns"),
        ("深蹲时膝盖疼怎么办", _json({"main": "深蹲要点", "variants": []}), "concerns"),
        ("肌酸要怎么吃", _reply("不是 JSON"), "model_failed"),
    ],
)
def test_bad_rewrite_falls_back_to_original(text: str, reply: Any, reason: str) -> None:
    out = _rewrite(text, reply)
    assert out.queries == (text,) and out.status == "rejected" and out.reason == reason


def test_without_model_only_original() -> None:
    out = _rewrite("肌酸要怎么吃", None)
    assert out.queries == ("肌酸要怎么吃",) and out.status == "original"


# ---- 对话入口：改写在意图识别之前，运行记录不含查询原文 ----


def test_chat_rewrites_before_intent_and_records_trace() -> None:
    seen: list[str] = []

    def intent(payload: dict[str, Any]) -> dict[str, object]:
        seen.append(payload["message"])
        return _json({"intent": "nutrition", "concerns": [], "confidence": 0.9})

    rewrite = _json({"main": "一水肌酸的服用方法", "variants": ["肌酸怎么吃"]})
    router = Router(rewrite=lambda _: rewrite, agent=_agent_reply, intent=intent)
    runs: list[Any] = []
    deps = _deps([], memory_db())
    deps = type(deps)(
        **{
            **deps.__dict__,
            "gateway": _gateway(router)[0],
            "record_run": lambda u, r: runs.append(r),
        }
    )
    reply = respond("u1", "肌酸要怎么吃", deps)
    assert seen == ["一水肌酸的服用方法"]  # 主查询交给意图识别
    assert reply.layer == "l1"
    [record] = runs
    assert record.query_count == 3 and record.rewrite == "model"
    trace = {t.agent: t for t in record.retrieval}
    assert "card:QA-CREATINE" in trace[_A.NUTRITIONIST].keyword
    dumped = record.model_dump_json()
    assert "肌酸" not in dumped.replace("CREATINE", "")  # 不含原话与查询


def test_general_chat_is_not_rewritten() -> None:
    router = Router()
    deps = _deps([], memory_db())
    deps = type(deps)(**{**deps.__dict__, "gateway": _gateway(router)[0]})
    respond("u1", "你好", deps)
    assert "rewrite" not in router.stages


# ---- 大模型重排 ----


def _passages() -> list[Passage]:
    kb = KnowledgeBase.from_sources(CARDS, default_library())
    return [p for p in kb.passages if p.passage_id in ("card:QA-CREATINE", "card:QA-PROTEIN")]


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ({"card:QA-CREATINE": 0.9, "card:QA-PROTEIN": 0.1}, [0.9, 0.1]),
        ({"card:QA-CREATINE": 0.9}, None),  # 漏打
        ({"card:QA-CREATINE": 0.9, "card:QA-PROTEIN": 0.1, "card:X": 1}, None),  # 多打
        ({"card:QA-CREATINE": 1.5, "card:QA-PROTEIN": 0.1}, None),  # 越界
    ],
)
def test_model_reranker_validates_scores(scores: dict[str, float], expected: Any) -> None:
    passages = _passages()
    order = {p.passage_id: n for n, p in enumerate(passages)}
    gw, _ = _gateway(Router(rerank=lambda _: _json({"scores": scores})))
    got = ModelReranker(gw, request_id="r", question="肌酸要怎么吃", role="营养师").score(
        "肌酸要怎么吃", passages
    )
    if expected is None:
        assert got is None
    else:
        assert got == [
            scores[p.passage_id] for p in sorted(passages, key=lambda p: order[p.passage_id])
        ]


def test_model_rerank_drops_irrelevant_candidates_end_to_end() -> None:
    """重排判为无关的候选不算依据：营养师没有可用依据，不调用模型作答。"""
    router = Router(rerank=lambda p: _json({"scores": {c["id"]: 0.0 for c in p["passages"]}}))
    gw, _ = _gateway(router)
    result = _ask("肌酸要怎么吃", _intent(Intent.SUPPLEMENTS_DRUGS), gw)
    trace = {t.agent: t for t in result.record.retrieval}
    assert trace[_A.NUTRITIONIST].reranked_by == "model" and trace[_A.NUTRITIONIST].kept == ()
    assert "agent" not in router.stages


# ---- 问答卡优先级最低（端到端） ----


def test_answer_cites_guide_not_card_when_both_qualify() -> None:
    guide = Passage(
        passage_id="guide:creatine", domain=_A.NUTRITIONIST, title="肌酸补充",
        text="一水肌酸每天服用，随餐即可，长期规律服用比时机更重要。", source="某指南",
        level="guideline", review="approved",
    )  # fmt: skip
    retriever = Retriever(KnowledgeBase.from_sources(CARDS, default_library(), [guide]))
    gw, _ = _gateway(Router(agent=_agent_reply))
    result = _ask("肌酸要怎么吃", _intent(Intent.SUPPLEMENTS_DRUGS), gw, retriever=retriever)
    refs = [c["ref"] for c in result.citations]
    assert refs == ["guide:creatine"]


# ---- 向量服务 ----


class _EmbedTransport:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.bodies = fail, []  # type: ignore[var-annotated]

    def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
    ) -> dict[str, object]:
        assert url.endswith("/embeddings")
        self.bodies.append(body)
        if self.fail:
            raise TimeoutError
        data = [
            {"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])
        ]
        return {"data": list(reversed(data))}


def test_gateway_embedder_batches_orders_and_records() -> None:
    from ironm.contracts.model_call import ModelCall
    from ironm.core.clock import FixedClock
    from ironm.core.ids import SeededIdGenerator
    from ironm.llm.config import MODELS_PATH, load_models
    from ironm.llm.gateway import EMBED_BATCH, Gateway
    from tests.llm.test_gateway import ENV, NOW

    calls: list[ModelCall] = []
    transport = _EmbedTransport()
    gw = Gateway(
        load_models(MODELS_PATH), environ={**ENV, "SILICONFLOW_API_KEY": "k"},
        clock=FixedClock(NOW), ids=SeededIdGenerator(1), transport=transport, record=calls.append,
    )  # fmt: skip
    texts = ["x" * (n + 1) for n in range(EMBED_BATCH + 3)]
    vectors = GatewayEmbedder(gw, "req-9").embed(texts)
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]  # 按 index 排回原顺序
    assert len(transport.bodies) == 2 and len(transport.bodies[0]["input"]) == EMBED_BATCH
    assert {c.stage.value for c in calls} == {"embedding"} and calls[0].request_id == "req-9"
    assert all("x" not in json.dumps(c.model_dump(mode="json")) for c in calls)  # 不记文本

    no_key = Gateway(
        load_models(MODELS_PATH), environ=ENV, clock=FixedClock(NOW), ids=SeededIdGenerator(1)
    )
    with pytest.raises(Exception):  # noqa: B017 - 没有密钥：抛错，由检索跳过这一路
        GatewayEmbedder(no_key, "r").embed(["a"])


# ---- 降级记录 ----


def test_degraded_steps_recorded_when_models_unavailable() -> None:
    result = _ask("肌酸要怎么吃", _intent(Intent.SUPPLEMENTS_DRUGS))
    assert "model_unavailable" in result.record.degraded_steps
    assert dict(result.record.agents)[_A.NUTRITIONIST] == "card"


def test_rewrite_rejection_recorded() -> None:
    result = _ask("肌酸要怎么吃", _intent(Intent.SUPPLEMENTS_DRUGS), rewrite="rejected:numbers")
    assert "rewrite:numbers" in result.record.degraded_steps


# ---- 变异检验补充 ----


class _Fixed:
    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, query: str, passages: Any) -> list[float]:
        return [self.value for _ in passages]


def test_model_scores_use_model_threshold_not_coverage_threshold() -> None:
    r = Retriever(KnowledgeBase.from_sources(CARDS, default_library()), min_score=0.2,
                  model_min_score=0.5)  # fmt: skip
    assert r.retrieve(["肌酸要怎么吃"], _A.NUTRITIONIST, reranker=_Fixed(0.4)).passages == ()
    assert r.retrieve(["肌酸要怎么吃"], _A.NUTRITIONIST, reranker=_Fixed(0.6)).passages


def test_coverage_fallback_takes_best_query() -> None:
    """主查询说得含糊，另一条查询说清楚了：按各查询中最高的覆盖分判断。"""
    r = Retriever(KnowledgeBase.from_sources(CARDS, default_library()))
    got = r.retrieve(["你好呀今天", "肌酸要怎么吃"], _A.NUTRITIONIST)
    assert got.reranked_by == "coverage"
    assert got.passages and got.passages[0].passage.passage_id == "card:QA-CREATINE"


def test_safety_card_kept_even_when_reranker_scores_it_low() -> None:
    """安全官的保证：命中触发词的安全类卡不会被重排挡掉。"""
    from tests.test_qa_chain import RAPID

    router = Router(
        rerank=lambda p: _json({"scores": {c["id"]: 0.0 for c in p["passages"]}}),
        agent=_agent_reply,
    )
    result = _ask(RAPID, _intent(Intent.NUTRITION), _gateway(router)[0])
    assert dict(result.record.agents)[_A.SAFETY_OFFICER] == "model"
    assert "card:QA-RAPID-LOSS" in [c["ref"] for c in result.citations]


def test_safety_card_fallback_keeps_only_for_you_from_others() -> None:
    """模型不可用：直接用安全类卡；其他领域只保留“针对你”的计算（BMI），不拼接它们的意见。"""
    result = _ask("身高 165，体重 45 公斤，想减到 40", _intent(Intent.PERSONAL_DATA))
    assert dict(result.record.agents)[_A.SAFETY_OFFICER] == "card"
    assert not any(p.kind == "answer" for p in result.parts)
    assert any(p.kind == "for_you" and "16.5" in p.text for p in result.parts)


def test_safety_opinion_goes_first_into_merge() -> None:
    from tests.test_qa_chain import RAPID

    seen: list[dict[str, Any]] = []

    def merge(payload: dict[str, Any]) -> dict[str, object]:
        seen.append(payload)
        return _json({"conclusion": "安全官的要点"})

    router = Router(agent=_agent_reply, compose=merge)
    intent = _intent(Intent.NUTRITION, domains=(Intent.NUTRITION, Intent.TRAINING_KNOWLEDGE))
    _ask(RAPID, intent, _gateway(router)[0], queries=(RAPID, "一天只吃一顿", "快速减重"))
    assert seen and seen[0]["opinions"][0]["role"].startswith("安全官")
