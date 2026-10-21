"""问答链路（spec [2026-10-21] §12）：通用 Agent 直接答；专业问题由各领域 Agent 并行
检索 → 生成，管家整合；安全官的提醒必在；问答不加载 Skill。"""

import ast
import json
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from ironm.agents import answering
from ironm.agents import rerank as rerank_mod
from ironm.agents.answering import AgentAnswer
from ironm.agents.tools import ToolInput, calorie_tool
from ironm.contracts.agents import Citation, Opinion
from ironm.contracts.model_call import ModelCall
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import Screening, Tier
from ironm.core.clock import FixedClock
from ironm.core.ids import SeededIdGenerator
from ironm.knowledge.kb import Retriever
from ironm.knowledge.qa_cards import default_cards
from ironm.llm.config import MODELS_PATH, load_models
from ironm.llm.gateway import Gateway
from ironm.orchestrator import compose as compose_mod
from ironm.orchestrator import general as general_mod
from ironm.orchestrator import intent as intent_mod
from ironm.orchestrator import opinions
from ironm.orchestrator import rewrite as rewrite_mod
from ironm.orchestrator.chat import load_answer_texts, respond
from ironm.orchestrator.compose import compose
from ironm.orchestrator.general import general_answer
from ironm.orchestrator.intent import Intent, IntentResult, load_intent_book, recognize
from ironm.orchestrator.opinions import collect_opinions
from ironm.orchestrator.qa import answer_question
from ironm.rules.loader import RULES_DIR, load_rules
from ironm.safety.tiers import classify
from tests.llm.test_gateway import ENV, _reply
from tests.memory_helpers import memory_db
from tests.skills.helpers import FULL_PROFILE, proposed, run_agent
from tests.test_chat_safety import RETRIEVER, _deps

_A = AgentId
RULES = load_rules(RULES_DIR)
CARDS = default_cards()
TEXTS = load_answer_texts()
NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)
TZ = ZoneInfo("Asia/Shanghai")

type Reply = Callable[[dict[str, Any]], dict[str, object]]


class Router:
    """按系统提示（哪个环节）与请求内容（哪个 Agent）给出回复：并行作答时顺序不固定。"""

    def __init__(self, **by_stage: Reply) -> None:
        self.by_stage = by_stage
        self.stages: list[str] = []
        self._lock = threading.Lock()

    def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
    ) -> dict[str, object]:
        system = body["messages"][0]["content"]
        stage = next(
            (
                name
                for name, path in (
                    ("agent", answering.PROMPT_PATH),
                    ("compose", compose_mod.PROMPT_PATH),
                    ("general", general_mod.PROMPT_PATH),
                    ("rerank", rerank_mod.PROMPT_PATH),
                    ("rewrite", rewrite_mod.PROMPT_PATH),
                    ("intent", intent_mod.PROMPT_PATH),
                )
                if system.startswith(path.read_text(encoding="utf-8").split("{")[0])
            ),
            "other",
        )
        if stage == "rerank" and stage not in self.by_stage:
            return _reply("跳过重排")  # 没配置重排时：检索改用覆盖比例打分，不记入 stages
        with self._lock:
            self.stages.append(stage)
        if stage not in self.by_stage:
            raise TimeoutError(stage)  # 没有配置的环节当作模型不可用
        return self.by_stage[stage](json.loads(body["messages"][1]["content"]))


def _gateway(router: Router) -> tuple[Gateway, list[ModelCall]]:
    calls: list[ModelCall] = []
    gw = Gateway(
        load_models(MODELS_PATH),
        environ=ENV,
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(9),
        transport=router,
        record=calls.append,
    )
    return gw, calls


def _json(body: dict[str, object]) -> dict[str, object]:
    return _reply(json.dumps(body, ensure_ascii=False))


SAFETY_TOPICS = ("掉秤", "一天只吃一顿")  # 测试里安全官“认为相关”的问题


def _agent_reply(payload: dict[str, Any]) -> dict[str, object]:
    """各 Agent 按给它的第一条依据作答，结论带上自己的职责，便于检查整合结果。"""
    first = (payload["passages"] or payload["facts"])[0]
    role = payload["role"].split("：")[0]
    if role == "安全官" and not any(t in payload["question"] for t in SAFETY_TOPICS):
        # 模型判断：与安全无关的问题，安全官没有可说的
        return _json(
            {"conclusion": "没有找到可靠依据", "reasoning": "无关。", "cites": [first["id"]]}
        )
    return _json({"conclusion": f"{role}的要点", "reasoning": "按依据。", "cites": [first["id"]]})


def _intent(intent: Intent, *concerns: str, domains: tuple[Intent, ...] = ()) -> IntentResult:
    return IntentResult.model_validate(
        {
            "intent": intent,
            "concerns": concerns,
            "confidence": 0.9,
            "source": "rule",
            "clarify": False,
            "domains": domains,
        }
    )


def _ask(text: str, intent: IntentResult, gateway: Gateway | None = None, **kw: Any) -> Any:
    return answer_question(
        text,
        intent,
        cards=CARDS,
        texts=TEXTS,
        gateway=gateway,
        request_id="req-1",
        rulebook=RULES,
        retriever=kw.pop("retriever", RETRIEVER),
        profile=kw.pop("profile", {}),
        as_of=NOW,
        tz=TZ,
        **kw,
    )


def _text(result: Any) -> str:
    return "".join(p.text for p in result.parts)


class SpyRetriever(Retriever):
    def __init__(self, base: Retriever) -> None:
        self.__dict__.update(base.__dict__)
        self.queries: list[tuple[str, AgentId]] = []

    def retrieve(self, queries: Any, domain: AgentId, **kw: Any) -> Any:
        self.queries += [(q, domain) for q in queries]
        return super().retrieve(queries, domain, **kw)


# ---- 通用 Agent ----


@pytest.mark.parametrize(
    ("text", "kind"), [("你好", "chitchat"), ("这个app怎么记录体重", "app_help")]
)
def test_general_questions_answered_directly_without_retrieval(text: str, kind: str) -> None:
    spy = SpyRetriever(RETRIEVER)
    deps = _deps([], memory_db())
    reply = respond("u1", text, type(deps)(**{**deps.__dict__, "retriever": spy}))
    assert reply.layer == "template" and spy.queries == []
    assert getattr(TEXTS, kind) in "".join(p.text for p in reply.answer)


def test_general_agent_uses_model_but_never_numbers() -> None:
    ok = Router(general=lambda _: _json({"text": "在记录页点体重就能记。"}))
    said, by_model = general_answer(
        "怎么记录体重", "app_help", gateway=_gateway(ok)[0], request_id="r", fallback="固定"
    )
    assert (said, by_model) == ("在记录页点体重就能记。", True)
    numbers = Router(general=lambda _: _json({"text": "每天走 8000 步就好。"}))
    said, by_model = general_answer(
        "你好", "chitchat", gateway=_gateway(numbers)[0], request_id="r", fallback="固定"
    )
    assert (said, by_model) == ("固定", False)


# ---- 专业问题：检索 → 生成 ----


def test_single_domain_answer_cites_knowledge_passage() -> None:
    router = Router(agent=_agent_reply)
    gw, _ = _gateway(router)
    result = _ask("肌酸要怎么吃", _intent(Intent.SUPPLEMENTS_DRUGS), gw)
    assert _spoke(result.record) == ((_A.NUTRITIONIST, "model"),)
    assert [c["ref"] for c in result.citations] == ["card:QA-CREATINE"]
    assert result.record.card_ids == ("QA-CREATINE",) and result.record.layer == "l1"
    assert router.stages == ["agent"]  # 只有一个领域：不经管家整合
    assert result.parts[0].text == "营养师的要点"


CROSS = "增肌一周练几次，蛋白质一天吃多少"
CROSS_INTENT = _intent(
    Intent.TRAINING_KNOWLEDGE, domains=(Intent.TRAINING_KNOWLEDGE, Intent.NUTRITION)
)
# 改写把一句里的两件事拆开（spec §13）；原话放在最后
CROSS_QUERIES = ("增肌一周练几次", "蛋白质一天吃多少", CROSS)


def _spoke(record: Any) -> tuple[tuple[AgentId, str], ...]:
    """作答的 Agent（安全官没话说时不列出：它每次都参与）。"""
    return tuple(
        (a, st) for a, st in record.agents if not (a is _A.SAFETY_OFFICER and st == "missing")
    )


def test_cross_domain_agents_answer_in_parallel_and_butler_merges() -> None:
    seen: list[dict[str, Any]] = []

    def merge(payload: dict[str, Any]) -> dict[str, object]:
        seen.append(payload)
        return _json({"conclusion": "训练规划师的要点；营养师的要点"})

    gw, _ = _gateway(Router(agent=_agent_reply, compose=merge))
    result = _ask(CROSS, CROSS_INTENT, gw, queries=CROSS_QUERIES)
    statuses = dict(result.record.agents)
    assert statuses[_A.TRAINING_PLANNER] == statuses[_A.NUTRITIONIST] == "model"
    assert result.record.query_count == 3
    assert result.record.composed_by_model
    # 相关领域（动作教练）没检索到依据：不说话，也不算“部分建议”
    assert statuses.get(_A.EXERCISE_COACH, "missing") == "missing"
    assert not any(p.kind == "notice" for p in result.parts)
    # 管家拿到两个领域的意见，主领域在前
    assert [o["conclusion"] for o in seen[0]["opinions"]] == ["训练规划师的要点", "营养师的要点"]
    assert {"card:QA-STRENGTH-FREQUENCY", "card:QA-PROTEIN"} <= set(result.record.agent_basis)


def test_cross_domain_without_models_keeps_both_domains() -> None:
    result = _ask(CROSS, CROSS_INTENT, queries=CROSS_QUERIES)
    statuses = dict(result.record.agents)
    assert statuses[_A.TRAINING_PLANNER] == statuses[_A.NUTRITIONIST] == "card"
    text = _text(result)
    for card in ("QA-STRENGTH-FREQUENCY", "QA-PROTEIN"):
        assert CARDS.get(card).conclusion in text
    assert result.record.card_ids == ("QA-STRENGTH-FREQUENCY", "QA-PROTEIN")


def test_no_basis_says_so_without_inventing() -> None:
    router = Router(agent=_agent_reply)
    gw, _ = _gateway(router)
    result = _ask("穿什么跑鞋好", _intent(Intent.TRAINING_KNOWLEDGE), gw)
    assert result.record.agents and all(st == "missing" for _, st in result.record.agents)
    assert result.record.card_ids == ()
    assert "没有找到可靠依据" in _text(result)
    assert router.stages == []  # 没有依据就不让模型作答


@pytest.mark.parametrize("extra", [{}, {"high_protein": True}])
def test_same_number_in_qa_and_plan(extra: dict[str, Any]) -> None:
    """问答里的每日热量目标与计划链路的能量目标一致：二者调用同一份计算。"""
    profile = {**FULL_PROFILE, **extra, "goal": "fat_loss"}
    screening = Screening.model_validate_json(
        json.dumps({k: v for k, v in FULL_PROFILE.items() if k != "activity"})
    )
    tier = classify(screening, as_of=NOW, tz=TZ, rulebook=RULES)
    assert isinstance(tier.tier, Tier)
    qa = calorie_tool(
        ToolInput(
            text="我一天要吃多少热量",
            profile=profile,
            context_item=None,
            rulebook=RULES,
            as_of=NOW,
            tz=TZ,
        )
    )
    plan = proposed(
        run_agent(
            _A.NUTRITIONIST, "goal.create", tier.tier, matched=tier.matched, **FULL_PROFILE, **extra
        )["energy_target"]
    )
    assert qa is not None
    assert qa.extra["kcal"] == plan["daily_kcal_target"]
    assert qa.extra["protein_g"] == plan["daily_protein_g_target"]


# ---- 安全官 ----


def test_safety_caution_always_in_merged_answer() -> None:
    """模型整合时漏写了身体不适的提醒：补进回答。"""
    gw, _ = _gateway(
        Router(agent=_agent_reply, compose=lambda _: _json({"conclusion": "两边的要点。"}))
    )
    intent = _intent(
        Intent.TRAINING_KNOWLEDGE,
        "discomfort",
        domains=(Intent.TRAINING_KNOWLEDGE, Intent.NUTRITION),
    )
    result = _ask(CROSS, intent, gw, queries=CROSS_QUERIES)
    assert result.record.composed_by_model
    assert any(p.kind == "caution" and p.text == TEXTS.caution_discomfort for p in result.parts)


RAPID = "两周后婚礼，怎么一天只吃一顿快速掉秤"


def test_safety_card_is_answered_by_safety_officer_model() -> None:
    """命中安全类卡：不再直接用卡，由安全官基于这张卡用模型作答，放在回答最前（spec §13）。"""
    router = Router(agent=_agent_reply)
    gw, _ = _gateway(router)
    result = _ask(RAPID, _intent(Intent.NUTRITION), gw)
    assert dict(result.record.agents)[_A.SAFETY_OFFICER] == "model"
    assert result.parts[0].text == "安全官的要点"
    assert "card:QA-RAPID-LOSS" in [c["ref"] for c in result.citations]


def test_safety_card_used_directly_only_when_model_cannot_answer() -> None:
    result = _ask(RAPID, _intent(Intent.NUTRITION))
    assert dict(result.record.agents)[_A.SAFETY_OFFICER] == "card"
    assert result.record.card_ids == ("QA-RAPID-LOSS",)
    assert result.parts[0].text == CARDS.get("QA-RAPID-LOSS").conclusion
    assert "model_unavailable" in result.record.degraded_steps


# ---- 并行收集与整合 ----


def _opinion(conclusion: str, ref: str = "card:QA-PROTEIN") -> Opinion:
    cite = Citation(kind="knowledge", ref=ref, label="依据", detail="每公斤体重 1.6 克")
    return Opinion(conclusion=conclusion, citations=(cite,), reasoning="按依据。")


def _collect(agents: list[AgentId]) -> Any:
    return collect_opinions(
        agents,
        text="问题",
        profile={},
        context_item=None,
        retriever=RETRIEVER,
        cards=CARDS,
        rulebook=RULES,
        gateway=None,
        request_id="r",
        as_of=NOW,
        tz=TZ,
    )


def test_agents_run_in_parallel_and_one_failure_is_only_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2, timeout=5)  # 依次执行时会超时

    def fake(agent: AgentId, **_: Any) -> AgentAnswer:
        if agent is _A.RECOVERY:
            raise RuntimeError("boom")
        barrier.wait()
        return AgentAnswer(agent, "model", _opinion(agent.value), (), ())

    monkeypatch.setattr(opinions, "answer_as", fake)
    got = _collect([_A.TRAINING_PLANNER, _A.RECOVERY, _A.NUTRITIONIST])
    assert [a.agent for a in got.answers] == [_A.TRAINING_PLANNER, _A.RECOVERY, _A.NUTRITIONIST]
    assert [a.agent for a in got.with_opinion] == [_A.TRAINING_PLANNER, _A.NUTRITIONIST]
    [missing] = got.missing
    assert missing.reason == "agent_error:RuntimeError"


def _answers(*conclusions: str | None) -> list[AgentAnswer]:
    agents = [_A.TRAINING_PLANNER, _A.NUTRITIONIST, _A.RECOVERY]
    return [
        AgentAnswer(a, "model", _opinion(c), (), ())
        if c
        else AgentAnswer(a, "missing", None, (), ())
        for a, c in zip(agents, conclusions, strict=False)
    ]


def test_compose_single_opinion_needs_no_model() -> None:
    router = Router()
    out = compose(
        "q", _answers("多练腿"), cautions=(), partial_notice="部分", gateway=_gateway(router)[0],
        request_id="r",
    )  # fmt: skip
    assert out.parts[0].text == "多练腿" and not out.by_model and router.stages == []


def test_compose_partial_notice_comes_first() -> None:
    out = compose(
        "q", _answers("多练腿", None), cautions=(), partial_notice="部分", gateway=None,
        request_id="r",
    )  # fmt: skip
    assert out.partial and out.parts[0].kind == "notice" and out.parts[0].text == "部分"


def test_related_domain_without_basis_is_not_partial_but_primary_is() -> None:
    quiet = AgentAnswer(_A.RECOVERY, "missing", None, (), (), "no_basis")
    [first] = _answers("多练腿")
    out = compose("q", [first, quiet], cautions=(), partial_notice="部分", gateway=None,
                  request_id="r")  # fmt: skip
    assert not out.partial and out.parts[0].text == "多练腿"
    out = compose("q", [quiet, first], cautions=(), partial_notice="部分", gateway=None,
                  request_id="r")  # fmt: skip
    assert out.partial and out.parts[0].text == "部分"


def test_compose_rewrites_once_when_number_has_no_source() -> None:
    replies: Iterator[dict[str, object]] = iter(
        [{"conclusion": "每公斤 3 克。"}, {"conclusion": "每公斤 1.6 克。"}]
    )
    gw, _ = _gateway(Router(compose=lambda _: _json(next(replies))))
    out = compose(
        "q", _answers("多练腿", "吃够蛋白"), cautions=(), partial_notice="部分", gateway=gw,
        request_id="r",
    )  # fmt: skip
    assert out.by_model and out.rewrites == 1 and out.parts[0].text == "每公斤 1.6 克。"


@pytest.mark.parametrize(
    "reply",
    [
        lambda _: _json({"conclusion": "每公斤 3 克。"}),  # 两次都没有来源
        lambda _: _reply("不是 JSON"),
    ],
)
def test_compose_falls_back_to_ordered_opinions(reply: Reply) -> None:
    gw, _ = _gateway(Router(compose=reply))
    out = compose(
        "q", _answers("多练腿", "吃够蛋白"), cautions=("注意",), partial_notice="部分",
        gateway=gw, request_id="r",
    )  # fmt: skip
    assert not out.by_model
    assert [p.text for p in out.parts if p.kind in ("conclusion", "answer")] == [
        "多练腿",
        "吃够蛋白",
    ]
    assert any(p.kind == "caution" and p.text == "注意" for p in out.parts)


# ---- 不加载 Skill ----

QA_MODULES = [
    "ironm/orchestrator/chat.py",
    "ironm/orchestrator/qa.py",
    "ironm/orchestrator/opinions.py",
    "ironm/orchestrator/compose.py",
    "ironm/orchestrator/general.py",
    "ironm/agents/answering.py",
    "ironm/agents/tools.py",
]


@pytest.mark.parametrize("path", QA_MODULES)
def test_qa_modules_do_not_import_skills(path: str) -> None:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    imported = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ] + [a.name for node in ast.walk(tree) if isinstance(node, ast.Import) for a in node.names]
    assert not [m for m in imported if m.startswith("ironm.skills")]


def test_conversation_never_loads_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    from ironm.skills import loader

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("问答链路加载了 Skill")

    monkeypatch.setattr(loader, "load_skills", forbidden)
    monkeypatch.setattr(loader, "select", forbidden)
    for text in ("你好", "高脚杯深蹲怎么做", CROSS, "肌酸要怎么吃", "深蹲时膝盖疼怎么办"):
        reply = respond("u1", text, _deps([], memory_db()))
        assert reply.sections or reply.answer, text


def test_lexicon_routes_cross_domain_question_to_both_agents() -> None:
    result = recognize(CROSS, book=load_intent_book(), rulebook=RULES, gateway=None, request_id="r")
    record = _ask(CROSS, result).record
    # 主领域在前，安全官每次都参与、排在最后
    assert [a for a, _ in record.agents] == [
        _A.TRAINING_PLANNER,
        _A.NUTRITIONIST,
        _A.SAFETY_OFFICER,
    ]


def test_page_question_only_asks_that_pages_agent() -> None:
    """训练页上问跨领域的问题：只由训练规划师作答，不叫营养师。"""
    result = _ask(
        CROSS,
        CROSS_INTENT,
        page="training",
        page_domain=_A.TRAINING_PLANNER,
        context_item="today.exercise:sq_goblet",
        queries=CROSS_QUERIES,
    )
    assert _spoke(result.record) == ((_A.TRAINING_PLANNER, "card"),)
    assert result.record.card_ids == ("QA-STRENGTH-FREQUENCY",)


def test_page_item_reaches_the_exercise_tool() -> None:
    """动作页问“这个怎么做”：动作教练按页面条目找到动作。"""
    result = _ask(
        "这个怎么做",
        _intent(Intent.EXERCISE_TECHNIQUE),
        page="training",
        page_domain=_A.EXERCISE_COACH,
        context_item="today.exercise:sq_box.set:2",
    )
    assert _spoke(result.record) == ((_A.EXERCISE_COACH, "tool"),)
    assert "exercise:sq_box" in [c["ref"] for c in result.citations]


def test_parallel_agents_record_model_calls_one_at_a_time() -> None:
    """各 Agent 并行调用模型；调用记录写入同一个数据库连接，不能交叠（交叠的事务会报错）。"""
    barrier = threading.Barrier(2, timeout=5)
    active, overlaps, recorded = [0], [0], []
    lock = threading.Lock()

    def record(call: ModelCall) -> None:
        with lock:
            active[0] += 1
            overlaps[0] += active[0] > 1
        time.sleep(0.02)
        recorded.append(call.call_id)
        with lock:
            active[0] -= 1

    arrived = [0]

    def reply(payload: dict[str, Any]) -> dict[str, object]:
        with lock:
            arrived[0] += 1
            first_two = arrived[0] <= 2
        if first_two:
            barrier.wait()  # 前两个 Agent 同时拿到回复、同时写记录
        return _agent_reply(payload)

    gw = Gateway(
        load_models(MODELS_PATH),
        environ=ENV,
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(3),
        transport=Router(agent=reply),
        record=record,
    )
    result = _ask(CROSS, CROSS_INTENT, gw, queries=CROSS_QUERIES)
    assert dict(result.record.agents)[_A.NUTRITIONIST] == "model"
    assert len(recorded) >= 2 and overlaps[0] == 0
