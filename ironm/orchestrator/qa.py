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
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.common import ContractModel
from ironm.contracts.model_call import ModelStage
from ironm.contracts.roles import AgentId
from ironm.core.body import bmi
from ironm.core.log import get_logger
from ironm.knowledge.qa_cards import CardBook, PartKind, QACard
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.orchestrator.domains import DOMAIN_WEIGHTS, select_domains
from ironm.orchestrator.intent import Concern, Intent, IntentResult
from ironm.orchestrator.numbers import numbers_in, unregistered
from ironm.rules.loader import RuleBook
from ironm.safety.matching import clauses

__all__ = ["MAX_CARDS", "AnswerPart", "QAResult", "RunRecord", "answer_question"]

_SAFETY = AgentId.SAFETY_OFFICER
ANSWER_TEMPLATE = "answer_l1@1.0.0"
PROMPT_PATH = Path(__file__).parent / "prompts" / "answer_l1.md"
MAX_CARDS = 3  # 交给模型改写的问答卡最多几张

_log = get_logger("ironm.qa")


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
    personal: tuple[str, ...]  # 用到的“针对你”计算（只记名称，如 bmi_measured）
    layer: Literal["l0", "l1", "general", "plan_note"]
    degraded: bool
    number_rewrites: int  # 数字校验后重写的次数


class AnswerPart(ContractModel):
    kind: Literal[
        "conclusion",
        "for_you",
        "why",
        "how",
        "caution",
        "uncertain",
        "clarify",
        "notice",
        "answer",
    ]
    text: Annotated[str, Field(min_length=1)]


class QAResult(ContractModel):
    parts: tuple[AnswerPart, ...]
    record: RunRecord


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


_HEIGHT_RE = re.compile(r"身高(1\d{2}(?:\.\d)?)|(1\d{2}(?:\.\d)?)(?:cm|厘米|公分)")
_WEIGHT_RE = re.compile(r"(\d{2,3}(?:\.\d)?)(公斤|kg|千克|斤)")
_BMI_RE = re.compile(r"bmi(?:是|为|只有)?(\d{1,2}(?:\.\d)?)")


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


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
) -> QAResult:
    """page 不为空时是功能页提问：只查 page_domain（本功能的领域）与安全类的卡。"""
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
        personal=() if personal is None else (personal[0],),
        layer=layer,  # type: ignore[arg-type]
        degraded=degraded,
        number_rewrites=rewrites,
    )
    _log.info("qa.answered", request_id=request_id, status=layer, count=len(parts))
    return QAResult(parts=tuple(parts), record=record)
