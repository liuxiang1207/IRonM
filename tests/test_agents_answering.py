"""专业 Agent 的问答（spec [2026-10-21] §12）：工具、检索本领域知识、带依据的意见与兜底。"""

import json
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from ironm.agents import rerank
from ironm.agents.answering import SPECS, answer_as
from ironm.agents.tools import ToolInput, bmi_tool, calorie_tool, exercise_tool
from ironm.contracts.model_call import ModelCall
from ironm.contracts.roles import AgentId
from ironm.core.clock import FixedClock
from ironm.core.ids import SeededIdGenerator
from ironm.knowledge.exercises import default_library
from ironm.knowledge.kb import KnowledgeBase, Retriever
from ironm.knowledge.qa_cards import default_cards
from ironm.llm.config import MODELS_PATH, load_models
from ironm.llm.gateway import Gateway
from ironm.rules.loader import RULES_DIR, load_rules
from sim.first_plan_demo import PERSONAS
from tests.llm.test_gateway import ENV, FakeTransport, _reply

_A = AgentId
RULES = load_rules(RULES_DIR)
NOW = datetime(2026, 10, 21, 9, tzinfo=UTC)
TZ = ZoneInfo("Asia/Shanghai")
CARDS = default_cards()
RETRIEVER = Retriever(KnowledgeBase.from_sources(CARDS, default_library()))
GYM = dict(next(iter(PERSONAS.values())))


def _inp(text: str, profile: dict[str, Any] | None = None, item: str | None = None) -> ToolInput:
    return ToolInput(
        text=text, profile=profile or {}, context_item=item, rulebook=RULES, as_of=NOW, tz=TZ
    )


# ---- 工具 ----


def test_bmi_from_text_profile_or_stated() -> None:
    said = bmi_tool(_inp("身高165，体重45公斤"))
    assert said is not None and said.extra["bmi"] == "16.5"
    assert "身高 165 厘米" in said.citations[0].label
    profile = bmi_tool(_inp("我胖吗", {"height_cm": 170.0, "weight_kg": 70.0}))
    assert profile is not None and "资料里的" in profile.citations[0].label
    stated = bmi_tool(_inp("我BMI 17"))
    assert stated is not None and stated.extra["bmi"] == "17"
    assert bmi_tool(_inp("今天练什么")) is None


def test_calorie_tool_matches_plan_calculation() -> None:
    result = calorie_tool(_inp("我一天吃多少", GYM))
    assert result is not None
    kcal = result.extra["kcal"]
    assert any(str(kcal) in c.label for c in result.citations)
    assert calorie_tool(_inp("我一天吃多少", {"sex": "female"})) is None  # 信息不全算不出


def test_exercise_tool_from_question_or_page_item() -> None:
    by_text = exercise_tool(_inp("高脚杯深蹲怎么做"))
    assert by_text is not None and by_text.citations[0].ref == "exercise:sq_goblet"
    by_item = exercise_tool(_inp("这个怎么做", item="today.exercise:sq_box.set:2"))
    assert by_item is not None and by_item.extra["exercise_id"] == "sq_box"
    assert exercise_tool(_inp("今天天气怎么样")) is None


def test_exercise_tool_adds_restriction_caution_and_easier_step() -> None:
    result = exercise_tool(_inp("高脚杯深蹲做不了", {"restrictions": ["knee_pain"]}))
    assert result is not None
    assert any(c.kind == "rule" for c in result.citations)
    assert any("做不了时先降阶：徒手深蹲" in h for h in result.how)
    harder = exercise_tool(_inp("高脚杯深蹲太简单了"))
    assert harder is not None and any("进阶" in h for h in harder.how)


# ---- 作答 ----


class AnswerTransport(FakeTransport):
    """重排的请求一律回不合规的内容（检索改用覆盖比例打分），其余按顺序回预设结果。
    requests 只记作答的请求。"""

    def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
    ) -> dict[str, object]:
        if body["messages"][0]["content"] == RERANK_PROMPT:
            self.reranks += 1
            return _reply("跳过重排")
        return super().__call__(url, headers, body, timeout_s)

    reranks = 0


RERANK_PROMPT = rerank.PROMPT_PATH.read_text(encoding="utf-8")


def _gateway(*results: object) -> tuple[Gateway, list[ModelCall], FakeTransport]:
    calls: list[ModelCall] = []
    transport = AnswerTransport(*results)
    gw = Gateway(
        load_models(MODELS_PATH),
        environ=ENV,
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(6),
        transport=transport,
        record=calls.append,
    )
    return gw, calls, transport


def _ask(agent: AgentId, text: str, gateway: Gateway | None = None, **kw: Any) -> Any:
    return answer_as(
        agent,
        text=text,
        profile=kw.pop("profile", GYM),
        context_item=kw.pop("item", None),
        retriever=RETRIEVER,
        cards=CARDS,
        rulebook=RULES,
        gateway=gateway,
        request_id="req-1",
        as_of=NOW,
        tz=TZ,
    )


def _gen(
    cites: list[str], conclusion: str = "肌酸每天 3–5 克即可。", **kw: Any
) -> dict[str, object]:
    body = {"conclusion": conclusion, "reasoning": "按知识库段落。", "cites": cites, **kw}
    return _reply(json.dumps(body, ensure_ascii=False))


def test_model_answer_grounded_in_retrieved_passage() -> None:
    gw, calls, _ = _gateway(_gen(["card:QA-CREATINE"]))
    result = _ask(_A.NUTRITIONIST, "肌酸要怎么吃", gw)
    assert result.status == "model" and result.passages[0] == "card:QA-CREATINE"
    assert result.opinion.citations[0].ref == "card:QA-CREATINE"
    assert [c.stage.value for c in calls] == ["rerank", "answer"]


def test_model_can_cite_tool_facts() -> None:
    kcal = calorie_tool(_inp("我一天吃多少", GYM)).extra["kcal"]  # type: ignore[union-attr]
    gw, _, _ = _gateway(_gen(["calc:kcal_target"], conclusion=f"你每天大约吃 {kcal} 千卡。"))
    result = _ask(_A.NUTRITIONIST, "我一天要吃多少热量", gw)
    assert result.status == "model" and "calorie_target" in result.tools


@pytest.mark.parametrize(
    "bad",
    [
        _gen(["card:QA-CREATINE"], conclusion="肌酸每天吃 20 克。"),  # 数字不在依据里
        _gen(["card:NOT-THERE"]),  # 引用了没给它的依据
        _reply("不是 JSON"),
    ],
)
def test_invalid_model_answer_falls_back_to_card(bad: dict[str, object]) -> None:
    gw, _, _ = _gateway(bad)
    result = _ask(_A.NUTRITIONIST, "肌酸要怎么吃", gw)
    assert result.status == "card" and result.reason == "model_failed_or_invalid"
    assert result.opinion.citations[0].ref == "QA-CREATINE"


def test_without_model_exercise_answer_comes_from_tool() -> None:
    result = _ask(_A.EXERCISE_COACH, "高脚杯深蹲怎么做")
    assert result.status == "tool" and "exercise_lookup" in result.tools
    assert "高脚杯深蹲" in result.opinion.conclusion


def test_no_basis_means_missing_without_model_call() -> None:
    gw, calls, _ = _gateway()
    result = _ask(_A.TRAINING_PLANNER, "动感单车和椭圆机哪个好", gw)
    assert result.status == "missing" and result.reason == "no_basis" and calls == []


def test_model_sees_only_declared_profile_fields() -> None:
    gw, _, transport = _gateway(_gen(["card:QA-CREATINE"]))
    _ask(
        _A.NUTRITIONIST,
        "肌酸要怎么吃",
        gw,
        profile={**GYM, "secret_note": "x", "restrictions": ["knee_pain"]},
    )
    payload = json.loads(transport.requests[0][2]["messages"][1]["content"])
    assert set(payload["profile"]) <= set(SPECS[_A.NUTRITIONIST].data)
    assert "secret_note" not in payload["profile"] and "restrictions" not in payload["profile"]
