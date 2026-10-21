"""意图识别（spec [2026-10-21] §4）：词表级 + 模型级，合并后得出意图、关注点与置信度。

- 词表级：每个意图一组说法，按命中的不同说法数排序；置信度按领先程度取规则表参数。
- 模型级：经模型网关识别；失败时只用词表级的结果。
- 合并：取置信度高者，相同时取词表；身体不适、特殊人群两类关注点取两者之并（只升不降）。
- 归不进任何领域且置信度低时追问；消息太短（如“好”）不追问，按闲聊处理。
意图识别不改计划、不写画像。
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ironm.contracts.common import ContractModel
from ironm.contracts.model_call import ModelStage
from ironm.core.yamlfile import load_yaml
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.rules.loader import RuleBook
from ironm.safety.matching import check_term, clauses

__all__ = [
    "INTENT_RULE",
    "INTENT_TERMS_PATH",
    "Concern",
    "Intent",
    "IntentBook",
    "IntentParams",
    "IntentResult",
    "load_intent_book",
    "recognize",
    "rule_intent",
]

INTENT_RULE = "R-QA-INTENT"
INTENT_TERMS_PATH = Path(__file__).with_name("intent_terms.yaml")
PROMPT_PATH = Path(__file__).parent / "prompts" / "intent_l3.md"
PROMPT_TEMPLATE = "intent_l3@1.0.0"


class Intent(StrEnum):
    EXERCISE_TECHNIQUE = "exercise_technique"
    TRAINING_KNOWLEDGE = "training_knowledge"
    PLAN = "plan"
    NUTRITION = "nutrition"
    SUPPLEMENTS_DRUGS = "supplements_drugs"
    PERSONAL_DATA = "personal_data"
    LOGGING = "logging"
    RECOVERY_SLEEP = "recovery_sleep"
    HABIT_MOTIVATION = "habit_motivation"
    WELLNESS = "wellness"
    CHITCHAT = "chitchat"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"  # 归不进任何领域


type Concern = Literal["discomfort", "special_population"]
type Source = Literal["rule", "model", "both"]


class IntentParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_conf_clear: Annotated[float, Field(gt=0, le=1)]
    rule_conf_single: Annotated[float, Field(gt=0, le=1)]
    rule_conf_tie: Annotated[float, Field(gt=0, le=1)]
    clarify_below: Annotated[float, Field(gt=0, le=1)]
    clarify_min_chars: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.rule_conf_tie < self.rule_conf_single <= self.rule_conf_clear:
            raise ValueError("应满足 并列 < 单个命中 ≤ 明确领先")
        return self


class IntentResult(ContractModel):
    intent: Intent
    concerns: tuple[Concern, ...]
    confidence: Annotated[float, Field(ge=0, le=1)]
    source: Source
    clarify: bool  # 是否应先追问


def _words(groups: list[str]) -> tuple[str, ...]:
    out = tuple(w for g in groups for w in g.split())
    for w in out:
        check_term(w)
    return out


class IntentBook(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    intents: dict[Intent, tuple[str, ...]]
    concerns: dict[Concern, tuple[str, ...]]

    @classmethod
    def from_yaml(cls, data: object) -> "IntentBook":
        if not isinstance(data, dict):
            raise ValueError("意图词表应为映射")
        return cls(
            version=str(data["version"]),
            intents={Intent(k): _words(v) for k, v in data["intents"].items()},
            concerns={k: _words(v) for k, v in data["concerns"].items()},
        )

    @model_validator(mode="after")
    def _check(self) -> Self:
        missing = set(Intent) - {Intent.UNKNOWN} - set(self.intents)
        if missing:
            raise ValueError(f"意图没有任何说法：{sorted(missing)}")
        return self


def load_intent_book(path: Path = INTENT_TERMS_PATH) -> IntentBook:
    try:
        return IntentBook.from_yaml(load_yaml(path))
    except (KeyError, ValueError, ValidationError) as e:
        raise ValueError(f"{path}：意图词表有误：{e}") from e


def _concerns(text: str, book: IntentBook) -> tuple[Concern, ...]:
    return tuple(c for c, words in book.concerns.items() if any(w in text for w in words))


def rule_intent(text: str, book: IntentBook, p: IntentParams) -> IntentResult:
    joined = "".join(clauses(text))
    hits = {i: sum(w in joined for w in set(words)) for i, words in book.intents.items()}
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])  # 稳定排序：同分时按词表顺序
    (top, n1), (_, n2) = ranked[0], ranked[1]
    if n1 == 0:
        intent, conf = Intent.UNKNOWN, 0.0
    elif n1 == n2:
        intent, conf = top, p.rule_conf_tie
    elif n1 >= 2:
        intent, conf = top, p.rule_conf_clear
    else:
        intent, conf = top, p.rule_conf_single
    return IntentResult(
        intent=intent,
        concerns=_concerns(joined, book),
        confidence=conf,
        source="rule",
        clarify=False,
    )


class _ModelIntent(BaseModel):
    """模型的输出格式。"""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    concerns: list[Concern] = []
    confidence: Annotated[float, Field(ge=0, le=1)]


def _merge(rule: IntentResult, model: _ModelIntent | None) -> IntentResult:
    if model is None:
        return rule
    concerns = tuple(dict.fromkeys([*rule.concerns, *model.concerns]))
    source: Source
    if model.intent == rule.intent:
        intent, conf, source = rule.intent, max(rule.confidence, model.confidence), "both"
    elif model.confidence > rule.confidence:
        intent, conf, source = model.intent, model.confidence, "model"
    else:
        intent, conf, source = rule.intent, rule.confidence, "rule"
    return IntentResult(
        intent=intent, concerns=concerns, confidence=conf, source=source, clarify=False
    )


def _prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def recognize(
    text: str,
    *,
    book: IntentBook,
    rulebook: RuleBook,
    gateway: Gateway | None,
    request_id: str,
) -> IntentResult:
    p = rulebook.get(INTENT_RULE).params_as(IntentParams)
    rule = rule_intent(text, book, p)
    model: _ModelIntent | None = None
    if gateway is not None:
        user = json.dumps({"message": text}, ensure_ascii=False)
        model = gateway.complete(
            ModelStage.INTENT,
            request_id=request_id,
            template=PROMPT_TEMPLATE,
            messages=[Message("system", _prompt()), Message("user", user)],
            validate=json_model(_ModelIntent),
        ).value
    result = _merge(rule, model)
    chars = len("".join(clauses(text)))
    if result.intent is Intent.UNKNOWN and chars < p.clarify_min_chars:
        # 太短的话（“好”“嗯”）不追问，按闲聊处理
        return result.model_copy(update={"intent": Intent.CHITCHAT})
    clarify = result.intent is Intent.UNKNOWN and result.confidence < p.clarify_below
    return result.model_copy(update={"clarify": clarify})
