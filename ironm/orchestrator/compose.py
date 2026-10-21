"""管家整合（spec [2026-10-21] §12）：把各专业 Agent 的意见合成一个回答。

- 只有一条意见：直接按它组织（结论 → 针对你 → 为什么 → 怎么做 → 注意 → 不确定），不调用模型。
- 多条意见：由模型整合；安全提醒必须写进回答；不同领域有分歧时如实写出。
  回答中的数字只能来自各意见（含依据），不合规时重写 1 次，仍不合规改为按顺序拼接各意见。
- 主领域没有意见、或有 Agent 出错 / 模型作答失败时，写明“这次只包含部分建议”；
  相关领域只是没检索到依据时不提示（它本来就不一定有话说）。
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.agents.answering import SPECS, AgentAnswer
from ironm.contracts.agents import Citation, Opinion
from ironm.contracts.model_call import ModelStage
from ironm.contracts.roles import AgentId
from ironm.core.numbers import numbers_in, unregistered
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.orchestrator.qa_parts import AnswerPart

__all__ = ["Composed", "compose"]

PROMPT_PATH = Path(__file__).parent / "prompts" / "compose.md"
PROMPT_TEMPLATE = "compose@1.0.0"


class Composed(BaseModel):
    model_config = ConfigDict(frozen=True)

    parts: tuple[AnswerPart, ...]
    by_model: bool  # 是否由模型整合
    partial: bool  # 是否有 Agent 缺失
    fallback: bool  # 本应由模型整合却退回了按顺序拼接
    rewrites: int  # 数字校验后重写的次数
    citations: tuple[dict[str, Any], ...]


class _Merged(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conclusion: Annotated[str, Field(min_length=1)]
    why: str | None = None
    how: str | None = None
    caution: str | None = None
    uncertain: str | None = None


def _for_you(opinion: Opinion) -> list[AnswerPart]:
    """计算类依据（如 BMI、热量目标）就是“针对你”的部分。"""
    return [
        AnswerPart(kind="for_you", text=c.label)
        for c in opinion.citations
        if c.kind == "calculation"
    ]


def _render(answers: Sequence[AgentAnswer], cautions: Sequence[str]) -> list[AnswerPart]:
    """不经模型：主领域的意见在前，其余作为补充，按统一结构排列。"""
    ops = [a.opinion for a in answers if a.opinion is not None]
    first, rest = ops[0], ops[1:]
    parts = [AnswerPart(kind="conclusion", text=first.conclusion)]
    parts += [AnswerPart(kind="answer", text=o.conclusion) for o in rest]
    for o in ops:
        parts += _for_you(o)
    parts.append(AnswerPart(kind="why", text=first.reasoning))
    parts += [AnswerPart(kind="how", text=o.how) for o in ops if o.how]
    notes = [*(o.caution for o in ops if o.caution), *cautions]
    parts += [AnswerPart(kind="caution", text=n) for n in dict.fromkeys(notes)]
    parts += [AnswerPart(kind="uncertain", text=o.uncertain) for o in ops if o.uncertain]
    return parts


# 自己写的内容（问答卡、动作库）可以完整展示；外部资料只给出处与一小段摘录（spec §12 约束）
_OWN_REFS = ("card:", "exercise:")
EXCERPT_CHARS = 80


def _shown(c: Citation) -> dict[str, Any]:
    data = c.model_dump()
    detail = c.detail or ""
    external = c.kind == "knowledge" and not c.ref.startswith(_OWN_REFS)
    if external and len(detail) > EXCERPT_CHARS:
        data["detail"] = detail[: EXCERPT_CHARS - 1] + "…"
    return data


def _opinion_text(o: Opinion) -> str:
    cites = " ".join(" ".join(v for v in (c.label, c.detail) if v) for c in o.citations)
    body = " ".join(v for v in (o.conclusion, o.reasoning, o.how, o.caution, o.uncertain) if v)
    return f"{body} {cites}"


def _merge(
    text: str,
    answers: Sequence[AgentAnswer],
    cautions: Sequence[str],
    gateway: Gateway,
    request_id: str,
) -> tuple[list[AnswerPart] | None, int]:
    ops = [(a.agent, a.opinion) for a in answers if a.opinion is not None]
    registered = set().union(*(numbers_in(_opinion_text(o)) for _, o in ops))
    registered |= {n for c in cautions for n in numbers_in(c)}
    payload = {
        "question": text,
        "opinions": [
            {"role": SPECS[agent].role, **o.model_dump(exclude={"citations"})} for agent, o in ops
        ],
        "required_cautions": list(cautions),
    }
    system = PROMPT_PATH.read_text(encoding="utf-8")
    rewrites = 0
    for attempt in range(2):
        hint = (
            ""
            if attempt == 0
            else f"\n上一次用了没有来源的数字。只能使用这些数字：{sorted(registered)}。"
        )
        result = gateway.complete(
            ModelStage.ANSWER,
            request_id=request_id,
            template=PROMPT_TEMPLATE,
            messages=[
                Message("system", system + hint),
                Message("user", json.dumps(payload, ensure_ascii=False)),
            ],
            validate=json_model(_Merged),
        )
        m = result.value
        if m is None:
            return None, rewrites
        values: list[
            tuple[Literal["conclusion", "why", "how", "caution", "uncertain"], str | None]
        ] = [
            ("conclusion", m.conclusion),
            ("why", m.why),
            ("how", m.how),
            ("caution", m.caution),
            ("uncertain", m.uncertain),
        ]
        if not unregistered(" ".join(v for _, v in values if v), registered):
            parts = [AnswerPart(kind=k, text=v) for k, v in values if v]
            # 安全提醒必须原样出现：模型漏写时补在注意里
            said = " ".join(p.text for p in parts)
            parts += [AnswerPart(kind="caution", text=c) for c in cautions if c not in said]
            return parts, rewrites
        rewrites += 1
    return None, rewrites


def compose(
    text: str,
    answers: Sequence[AgentAnswer],
    *,
    cautions: Sequence[str],
    partial_notice: str,
    gateway: Gateway | None,
    request_id: str,
) -> Composed:
    """answers 按主领域在前排列，安全官在最后；至少要有一条意见。

    安全官有意见、也有其他领域的意见时，它的结论与注意作为必须写进回答的提醒（冲突时以它为准）；
    只有安全官有意见时，它的意见就是回答。
    """
    with_opinion = [a for a in answers if a.opinion is not None]
    if not with_opinion:
        raise ValueError("没有可整合的意见")
    # 安全官每次都参与：它没话说不算“部分建议”
    core = [a for a in answers if a.agent is not AgentId.SAFETY_OFFICER]
    partial = bool(core) and (
        core[0].opinion is None or any(a.opinion is None and a.reason != "no_basis" for a in core)
    )
    extra_for_you: list[AnswerPart] = []
    safety = next((a for a in with_opinion if a.agent is AgentId.SAFETY_OFFICER), None)
    others = [a for a in with_opinion if a.agent is not AgentId.SAFETY_OFFICER]
    if safety is None:
        body = with_opinion
    elif safety.status == "card":
        # 安全官的模型回答不了，降级为直接用命中的安全类卡（spec §13）；其他领域不再拼进来
        body, partial = [safety], False
        # 其他领域算出的“针对你”的数字（如 BMI）仍然给出
        extra_for_you = [p for a in others for p in _for_you(a.opinion)]  # type: ignore[arg-type]
    else:
        # 安全官的意见放在最前，冲突时以它为准；它的结论作为必须写进回答的提醒交给整合
        body = [safety, *others]
        cautions = [*dict.fromkeys([safety.opinion.conclusion, *cautions])]  # type: ignore[union-attr]
        if not others:
            # 只有安全官作答：其他领域只是没有依据时不提示；出错或作答失败时照样提示
            partial = any(a.opinion is None and a.reason != "no_basis" for a in core)
    by_model, rewrites = False, 0
    parts: list[AnswerPart] | None = None
    wanted_model = len(body) > 1 and gateway is not None
    if wanted_model:
        parts, rewrites = _merge(text, body, cautions, gateway, request_id)  # type: ignore[arg-type]
        by_model = parts is not None
    if parts is None:
        parts = _render(body, cautions)
        parts = [parts[0], *extra_for_you, *parts[1:]]
    else:
        # “针对你”的计算（BMI、热量目标）不交给模型改写，原样放在结论之后
        mine = [p for a in body for p in _for_you(a.opinion)]  # type: ignore[arg-type]
        parts = [parts[0], *mine, *parts[1:]]
    if partial:
        parts = [AnswerPart(kind="notice", text=partial_notice), *parts]
    citations = tuple(
        {"agent": a.agent.value, **_shown(c)}
        for a in with_opinion
        for c in a.opinion.citations  # type: ignore[union-attr]
    )
    return Composed(
        parts=tuple(parts),
        by_model=by_model,
        partial=partial,
        fallback=wanted_model and not by_model,
        rewrites=rewrites,
        citations=citations,
    )
