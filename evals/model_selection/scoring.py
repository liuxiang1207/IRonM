"""模型选型评测的自动判分与成本估算。自动判分只是初筛，结论需人工抽查。"""

import json
import math
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

# 拒答的常见说法；正常问答出现这些即计为误拒
_REFUSAL_RE = re.compile(
    r"无法回答|不能回答|无法提供|不能提供|无法为你|不便回答|抱歉，?我不能|我不能帮|超出了我的"
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExtractExpect(_Model):
    type: str
    fields: dict[str, float | int | str | bool]
    absent: tuple[str, ...] = ()  # 用户没说、不得编造的字段


class Price(_Model):
    """每百万 token 的美元价格。"""

    input_hit: float
    input_miss: float
    output: float


@dataclass(frozen=True)
class Usage:
    input_hit: int = 0
    input_miss: int = 0
    output: int = 0


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reasons: tuple[str, ...] = field(default=())
    refused: bool = False


def _same(actual: object, expected: object) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual == expected
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return math.isclose(actual, expected, rel_tol=0.01, abs_tol=1e-9)
    return actual == expected


def score_extraction(answer: str, expect: ExtractExpect) -> Verdict:
    try:
        obj = json.loads(_FENCE_RE.sub("", answer.strip()))
    except json.JSONDecodeError:
        return Verdict(False, ("不是合法 JSON",))
    if not isinstance(obj, dict):
        return Verdict(False, ("顶层不是对象",))
    reasons: list[str] = []
    if obj.get("type") != expect.type:
        reasons.append(f"类型应为 {expect.type}，实际 {obj.get('type')!r}")
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    for key, value in expect.fields.items():
        if not _same(payload.get(key), value):
            reasons.append(f"{key} 应为 {value!r}，实际 {payload.get(key)!r}")
    for key in expect.absent:
        if payload.get(key) is not None:
            reasons.append(f"编造了用户没说的 {key}={payload[key]!r}")
    return Verdict(not reasons, tuple(reasons))


def score_text(
    answer: str, must_any: tuple[tuple[str, ...], ...], forbid: tuple[str, ...]
) -> Verdict:
    """must_any：每组至少命中一个词；forbid：正则，命中即一票否决。"""
    reasons = [f"缺少要点：{'/'.join(g)}" for g in must_any if not any(w in answer for w in g)]
    reasons += [f"命中禁止内容：{p}" for p in forbid if re.search(p, answer)]
    return Verdict(not reasons, tuple(reasons), refused=bool(_REFUSAL_RE.search(answer)))


def cost_usd(usage: Usage, price: Price) -> float:
    return (
        usage.input_hit * price.input_hit
        + usage.input_miss * price.input_miss
        + usage.output * price.output
    ) / 1_000_000
