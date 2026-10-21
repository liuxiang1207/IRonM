"""问答：查知识 → 回答（spec [2026-10-21] §6、§7、§12）。

- 查知识：按领域打分查问答卡；安全类的卡每次都查，命中时优先。功能页上只查本功能的领域与安全类。
- 回答：
  - L0：命中安全类的卡，或只命中一个领域、又没有安全提示要加时，直接用卡（不调用模型）；
  - L1：要把安全提示融进回答，或要合并两个以上领域的卡时，由模型按统一结构改写；
    数字必须有登记来源，不合规时重写 1 次，仍不合规退回 L0 并告诉用户；
  - 一般：没有审定知识时，不给具体建议。
- 要改计划的问题只说明“需进入计划调整”，不改计划；命中安全类的卡时先按卡回答。
- 针对你：用户在这句话里说的身高、体重或 BMI，算出 BMI 写进回答（视为登记过的数字）。
运行记录（RunRecord）不含用户原话与这些数字，可按请求编号回放。
"""

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from ironm.agents.answering import AgentAnswer
from ironm.contracts.common import ContractModel
from ironm.contracts.model_call import ModelStage
from ironm.contracts.roles import AgentId
from ironm.core.body import bmi
from ironm.core.log import get_logger
from ironm.core.numbers import numbers_in, unregistered
from ironm.knowledge.kb import Embedder, Retriever
from ironm.knowledge.qa_cards import CardBook, PartKind, QACard
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.orchestrator.compose import compose
from ironm.orchestrator.domains import DOMAIN_WEIGHTS, select_domains
from ironm.orchestrator.intent import Concern, Intent, IntentResult
from ironm.orchestrator.opinions import collect_opinions
from ironm.orchestrator.qa_parts import AnswerPart
from ironm.rules.loader import RULES_DIR, RuleBook, load_rules
from ironm.safety.matching import clauses

__all__ = ["MAX_CARDS", "AnswerPart", "QAResult", "RunRecord", "answer_question"]

_SAFETY = AgentId.SAFETY_OFFICER
ANSWER_TEMPLATE = "answer_l1@1.0.0"
PROMPT_PATH = Path(__file__).parent / "prompts" / "answer_l1.md"
MAX_CARDS = 3  # 交给模型改写的问答卡最多几张

_log = get_logger("ironm.qa")
_DEFAULT_TZ = ZoneInfo("Asia/Shanghai")
_NO_RULES = load_rules(RULES_DIR)  # 测试等未传规则表时用内置规则


class RunRecord(ContractModel):
    """一次问答的运行记录。不含用户原话、回答原文与用户说的数字。"""

    request_id: Annotated[str, Field(min_length=1)]
    intent: Intent
    concerns: tuple[Concern, ...]
    domain_scores: tuple[tuple[AgentId, float], ...]
    primary_domain: AgentId | None
    related_domains: tuple[AgentId, ...]
    page: str | None = None  # 功能页提问时的页面
    context_item: str | None = None  # 带入的条目编号（标识，不含原话）
    matched_cards: tuple[str, ...]  # 查到的问答卡（安全类在前）
    card_ids: tuple[str, ...]  # 回答用到的问答卡
    personal: tuple[str, ...]  # 用到的“针对你”数字的依据编号（如 calc:bmi、said:bmi）
    layer: Literal["l0", "l1", "general", "plan_note"]
    degraded: bool
    number_rewrites: int  # 数字校验后重写的次数
    # 各专业 Agent 的状态（model / tool / card / missing）与依据编号；不含原话
    agents: tuple[tuple[AgentId, str], ...] = ()
    agent_basis: tuple[str, ...] = ()
    composed_by_model: bool = False
    # 检索（spec [2026-10-21] §13）：查询条数、改写结果、各 Agent 的召回与重排；不含查询原文
    query_count: int = 1
    rewrite: str | None = None  # model / original / rejected:<原因>
    retrieval: tuple["RetrievalTrace", ...] = ()
    degraded_steps: tuple[str, ...] = ()  # 降级发生在哪些步骤，如 rerank:nutritionist


class RetrievalTrace(ContractModel):
    agent: AgentId
    reranked_by: str  # model / coverage / none
    keyword: tuple[str, ...]  # 关键词一路召回的段落编号
    vector: tuple[str, ...]  # 向量一路召回的段落编号
    kept: tuple[tuple[str, float], ...]  # 重排后留下的段落与分数


class QAResult(ContractModel):
    parts: tuple[AnswerPart, ...]
    record: RunRecord
    # 依据（可逐层展开）；来自问答类 Skill 的意见时为各意见的 Citation
    citations: tuple[dict[str, Any], ...] = ()
    # 没有给出意见的 Agent 与原因（如 unknown_exercise），回答里如实标注
    silent: tuple[str, ...] = ()


class _Texts(BaseModel):
    """回答用到的固定说法（与 answer_texts.yaml 对应的子集）。"""

    model_config = ConfigDict(frozen=True, extra="ignore")

    general: str
    degraded: str
    plan_change: str
    caution_discomfort: str
    caution_special: str
    bmi_measured: str
    bmi_stated: str
    partial: str


_HEIGHT_RE = re.compile(r"身高(1\d{2}(?:\.\d)?)|(1\d{2}(?:\.\d)?)(?:cm|厘米|公分)")
_WEIGHT_RE = re.compile(r"(\d{2,3}(?:\.\d)?)(公斤|kg|千克|斤)")
_BMI_RE = re.compile(r"bmi(?:是|为|只有)?(\d{1,2}(?:\.\d)?)")


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


# 两条路径记同一套编号：问答卡路径的说法名称 → Agent 路径里工具给出的依据编号
_PERSONAL_REF = {"bmi_measured": "calc:bmi", "bmi_stated": "said:bmi"}


def _is_personal(ref: object) -> bool:
    return str(ref).startswith(("calc:", "said:"))


def _personal(text: str) -> tuple[str, dict[str, str]] | None:
    """用户在这句话里说的身高体重（算出 BMI）或 BMI：(说法名称, 数字)。"""
    joined = "".join(clauses(text))
    height, weight = _HEIGHT_RE.search(joined), _WEIGHT_RE.search(joined)
    if height and weight:
        cm = float(height.group(1) or height.group(2))
        kg = float(weight.group(1)) / (2 if weight.group(2) == "斤" else 1)
        if 120 <= cm <= 230 and 25 <= kg <= 300:
            values = {"height": _fmt(cm), "weight": _fmt(kg), "bmi": _fmt(bmi(cm, kg))}
            return "bmi_measured", values
    stated = _BMI_RE.search(joined)
    if stated and 10 <= float(stated.group(1)) <= 60:
        return "bmi_stated", {"bmi": stated.group(1)}
    return None


class _L1Answer(BaseModel):
    """L1 模型输出：统一结构的各部分，没有的给 null。"""

    model_config = ConfigDict(extra="forbid")

    conclusion: Annotated[str, Field(min_length=1)]
    why: str | None = None
    how: str | None = None
    caution: str | None = None
    uncertain: str | None = None


def _card_parts(card: QACard) -> list[AnswerPart]:
    return [AnswerPart(kind=k, text=v) for k, v in card.parts()]


def _l1(
    text: str,
    cards: list[QACard],
    cautions: list[str],
    gateway: Gateway,
    request_id: str,
    extra_numbers: set[str],
) -> tuple[list[AnswerPart] | None, int]:
    """用模型改写；返回 (各部分, 数字重写次数)。失败或数字两次都不合规时返回 None。"""
    knowledge = [{"id": c.id, **dict(c.parts()), "sources": list(c.sources)} for c in cards]
    registered = set().union(*(numbers_in(" ".join(v for _, v in c.parts())) for c in cards))
    registered |= extra_numbers
    payload = {"question": text, "knowledge": knowledge, "cautions": cautions}
    system = PROMPT_PATH.read_text(encoding="utf-8")
    rewrites = 0
    for attempt in range(2):
        hint = (
            ""
            if attempt == 0
            else f"\n上一次的回答用了没有来源的数字。只能使用这些数字：{sorted(registered)}。"
        )
        result = gateway.complete(
            ModelStage.ANSWER,
            request_id=request_id,
            template=ANSWER_TEMPLATE,
            messages=[
                Message("system", system + hint),
                Message("user", json.dumps(payload, ensure_ascii=False)),
            ],
            validate=json_model(_L1Answer),
        )
        if result.value is None:
            return None, rewrites
        a = result.value
        values: list[tuple[PartKind, str | None]] = [
            ("conclusion", a.conclusion),
            ("why", a.why),
            ("how", a.how),
            ("caution", a.caution),
            ("uncertain", a.uncertain),
        ]
        if not unregistered(" ".join(v for _, v in values if v), registered):
            return [AnswerPart(kind=k, text=v) for k, v in values if v], rewrites
        rewrites += 1
    return None, rewrites


def _cards_cited(citations: Sequence[Mapping[str, Any]], cards: CardBook) -> tuple[str, ...]:
    """回答里用到的问答卡（知识库段落 card:<编号>，或兜底时直接用的卡）。"""
    known = {c.id for c in cards.cards}
    refs = (str(c["ref"]).removeprefix("card:") for c in citations)
    return tuple(dict.fromkeys(r for r in refs if r in known))


def _trace(a: AgentAnswer) -> RetrievalTrace:
    r = a.retrieval
    assert r is not None
    return RetrievalTrace(
        agent=a.agent,
        reranked_by=r.reranked_by,
        keyword=r.keyword,
        vector=r.vector,
        kept=tuple((x.passage.passage_id, round(x.score, 3)) for x in r.passages),
    )


def _degraded(answers: Sequence[AgentAnswer], has_model: bool) -> list[str]:
    """模型可用却没用上的步骤：重排退回覆盖比例、Agent 退回工具或问答卡。"""
    if not has_model:
        return ["model_unavailable"]
    out = [
        f"rerank:{a.agent.value}"
        for a in answers
        if a.retrieval is not None and a.retrieval.reranked_by == "coverage"
    ]
    out += [f"answer:{a.agent.value}:{a.status}" for a in answers if a.status in ("tool", "card")]
    return out  # fmt: skip


def answer_question(
    text: str,
    intent: IntentResult,
    *,
    cards: CardBook,
    texts: BaseModel,
    gateway: Gateway | None,
    request_id: str,
    rulebook: RuleBook | None = None,
    page: str | None = None,
    page_domain: AgentId | None = None,
    context_item: str | None = None,
    retriever: Retriever | None = None,
    profile: Mapping[str, Any] | None = None,
    as_of: datetime | None = None,
    tz: ZoneInfo | None = None,
    queries: Sequence[str] = (),
    embedder: Embedder | None = None,
    rewrite: str | None = None,
) -> QAResult:
    """page 不为空时是功能页提问：只调用本功能的 Agent（page_domain），只查它与安全类的卡。

    有 retriever 时，专业问题由选中的 Agent（含安全官）并行作答、管家整合（spec §12、§13）；
    Agent 都没有意见（或没有 retriever）时才降级到问答卡：命中安全类的卡时直接用卡。
    """
    fixed = _Texts.model_validate(texts.model_dump())
    if intent.intent not in DOMAIN_WEIGHTS:
        raise ValueError(f"意图 {intent.intent} 不经过问答（由对话入口处理）")
    domains = select_domains(intent, rulebook)
    if page is not None:
        allowed = {page_domain} if page_domain is not None else set()
        primary = page_domain
    else:
        allowed = {d for d in (domains.primary, *domains.related) if d is not None}
        primary = domains.primary
    # 查知识：安全类每次都查且排在前面，其次主领域，再按命中的具体程度
    found = [(c, s) for c, s in cards.matches(text) if c.agent == _SAFETY or c.agent in allowed]
    ranked = sorted(found, key=lambda cs: (cs[0].agent != _SAFETY, cs[0].agent != primary, -cs[1]))
    matched = [c for c, _ in ranked]

    cautions = []
    if "discomfort" in intent.concerns:
        cautions.append(fixed.caution_discomfort)
    if "special_population" in intent.concerns:
        cautions.append(fixed.caution_special)
    tried: tuple[tuple[AgentId, str], ...] = ()  # 作答过的 Agent（都没有意见时也记下）
    basis: tuple[str, ...] = ()
    traces: tuple[RetrievalTrace, ...] = ()
    steps: list[str] = [f"rewrite:{rewrite.split(':', 1)[1]}"] if rewrite and ":" in rewrite else []
    query_count = max(len(queries), 1)
    if retriever is not None and intent.intent is not Intent.PLAN:
        agents = [page_domain] if page is not None else [domains.primary, *domains.related]
        # 安全官每次都参与，排在最后：它的提醒写进回答，但它没话说时不算“部分建议”
        answering = [*(a for a in agents if a is not None and a != _SAFETY), _SAFETY]
        collected = collect_opinions(
            answering,
            text=text,
            profile=profile or {},
            context_item=context_item,
            retriever=retriever,
            cards=cards,
            rulebook=rulebook or _NO_RULES,
            gateway=gateway,
            request_id=request_id,
            as_of=as_of or datetime.now(UTC),
            tz=tz or _DEFAULT_TZ,
            queries=queries,
            embedder=embedder,
        )
        tried = tuple((a.agent, a.status) for a in collected.answers)
        traces = tuple(_trace(a) for a in collected.answers if a.retrieval is not None)
        steps += _degraded(collected.answers, gateway is not None)
        basis = tuple(dict.fromkeys(b for a in collected.answers for b in (*a.passages, *a.tools)))
        if collected.with_opinion:
            composed = compose(
                text,
                collected.answers,
                cautions=cautions,
                partial_notice=fixed.partial,
                gateway=gateway,
                request_id=request_id,
            )
            by_model = composed.by_model or any(a.status == "model" for a in collected.answers)
            final = list(composed.parts)
            said = _personal(text)
            if said is not None and not any(p.kind == "for_you" for p in final):
                # 这句话里给了身高体重（或 BMI），但作答的 Agent 里没有会算的：照样算给用户
                mine = AnswerPart(kind="for_you", text=getattr(fixed, said[0]).format(**said[1]))
                final.insert(1 if final and final[0].kind != "notice" else 2, mine)
            record = RunRecord(
                request_id=request_id,
                intent=intent.intent,
                concerns=intent.concerns,
                domain_scores=domains.scores,
                primary_domain=primary,
                related_domains=() if page is not None else domains.related,
                page=page,
                context_item=context_item,
                matched_cards=tuple(c.id for c in matched),
                card_ids=_cards_cited(composed.citations, cards),
                personal=tuple(
                    dict.fromkeys(
                        [
                            *(str(c["ref"]) for c in composed.citations if _is_personal(c["ref"])),
                            *([_PERSONAL_REF[said[0]]] if said is not None else []),
                        ]
                    )
                ),
                layer="l1" if by_model else "l0",
                degraded=composed.partial,
                number_rewrites=composed.rewrites,
                agents=tried,
                agent_basis=basis,
                composed_by_model=composed.by_model,
                retrieval=traces,
                degraded_steps=tuple([*steps, *(["compose"] if composed.fallback else [])]),
                query_count=query_count,
                rewrite=rewrite,
            )
            _log.info(
                "qa.answered", request_id=request_id, status="agents", count=len(composed.parts)
            )
            return QAResult(parts=tuple(final), record=record, citations=composed.citations)
    caution_parts = [AnswerPart(kind="caution", text=c) for c in cautions]
    personal = _personal(text)
    for_you = (
        []
        if personal is None
        else [AnswerPart(kind="for_you", text=getattr(fixed, personal[0]).format(**personal[1]))]
    )

    parts: list[AnswerPart]
    degraded, rewrites, used = False, 0, []
    safety_first = bool(matched) and matched[0].agent == _SAFETY
    if intent.intent is Intent.PLAN and not safety_first:
        layer = "plan_note"
        parts = [AnswerPart(kind="answer", text=fixed.plan_change), *caution_parts]
    elif not matched:
        layer, degraded = "general", True
        parts = [AnswerPart(kind="answer", text=fixed.general), *caution_parts]
    else:
        best = matched[0]
        # 安全类的卡就是为这类情况审定的：直接用，安全提示附在后面；
        # 要融入安全提示，或要合并两个以上领域的卡时，才需要模型改写
        fields = {c.agent for c in matched} - {_SAFETY}
        direct = (not cautions and len(fields) <= 1) or safety_first
        l1_parts: list[AnswerPart] | None = None
        if not direct and gateway is not None:
            chosen = matched[:MAX_CARDS]
            extra = {n for p in for_you for n in numbers_in(p.text)}
            l1_parts, rewrites = _l1(text, chosen, cautions, gateway, request_id, extra)
            used = [c.id for c in chosen]
        if l1_parts is not None:
            layer, parts = "l1", [*l1_parts[:1], *for_you, *l1_parts[1:]]
        else:
            # L0：直接用最合适的卡；本应改写却失败时要告诉用户
            degraded = not direct
            layer, used = "l0", [best.id]
            notice = [AnswerPart(kind="notice", text=fixed.degraded)] if degraded else []
            card = _card_parts(best)
            parts = [*notice, *card[:1], *for_you, *card[1:], *caution_parts]

    record = RunRecord(
        request_id=request_id,
        intent=intent.intent,
        concerns=intent.concerns,
        domain_scores=domains.scores,
        primary_domain=primary,
        related_domains=() if page is not None else domains.related,
        page=page,
        context_item=context_item,
        matched_cards=tuple(c.id for c in matched),
        card_ids=tuple(used),
        personal=() if personal is None else (_PERSONAL_REF[personal[0]],),
        layer=layer,  # type: ignore[arg-type]
        degraded=degraded,
        number_rewrites=rewrites,
        agents=tried,
        agent_basis=basis,
        retrieval=traces,
        degraded_steps=tuple([*steps, *([f"fallback:{layer}"] if tried else [])]),
        query_count=query_count,
        rewrite=rewrite,
    )
    _log.info("qa.answered", request_id=request_id, status=layer, count=len(parts))
    return QAResult(parts=tuple(parts), record=record)
