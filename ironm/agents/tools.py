"""Agent 的工具（spec [2026-10-21] §12）：算事实、查事实，问答与计划共用同一份计算。

每个工具只读传给它的画像字段与这句话，返回带依据（Citation）的结果；算不出来时返回 None。
工具不做决定、不给建议：建议由 Agent 基于这些事实与知识库生成。
"""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from ironm.contracts.agents import Citation
from ironm.contracts.calories import Activity
from ironm.contracts.exercises import Restriction
from ironm.contracts.screening import Screening
from ironm.contracts.unknown import NOT_RECORDED, is_known
from ironm.core.body import bmi
from ironm.engine.calories import calorie_target
from ironm.knowledge.exercises import Library, default_library
from ironm.rules.loader import RuleBook
from ironm.safety.matching import clauses
from ironm.safety.tiers import classify

__all__ = ["TOOLS", "ToolInput", "ToolResult", "run_tools"]

SCREENING_FIELDS = ("birth_date", "sex", "height_cm", "weight_kg", "waist_cm", "conditions")


@dataclass(frozen=True)
class ToolInput:
    text: str
    profile: Mapping[str, Any]  # 只含该 Agent 声明能读的字段
    context_item: str | None
    rulebook: RuleBook
    as_of: datetime
    tz: ZoneInfo


@dataclass(frozen=True)
class ToolResult:
    name: str
    citations: tuple[Citation, ...]
    # 给兜底用的确定性内容（模型不可用时，Agent 可以直接按这些给出意见）
    conclusion: str | None = None
    how: tuple[str, ...] = ()
    caution: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


_HEIGHT_RE = re.compile(r"身高(1\d{2}(?:\.\d)?)|(1\d{2}(?:\.\d)?)(?:cm|厘米|公分)")
_WEIGHT_RE = re.compile(r"(\d{2,3}(?:\.\d)?)(公斤|kg|千克|斤)")
_BMI_RE = re.compile(r"bmi(?:是|为|只有)?(\d{1,2}(?:\.\d)?)")


def bmi_tool(inp: ToolInput) -> ToolResult | None:
    """BMI：优先用这句话里说的身高体重，其次用画像；用户只说了 BMI 时照用。"""
    joined = "".join(clauses(inp.text))
    height, weight = _HEIGHT_RE.search(joined), _WEIGHT_RE.search(joined)
    cm: float | None = None
    kg: float | None = None
    if height and weight:
        cm = float(height.group(1) or height.group(2))
        kg = float(weight.group(1)) / (2 if weight.group(2) == "斤" else 1)
        source = "你在这句话里说的"
    elif isinstance(inp.profile.get("height_cm"), int | float) and isinstance(
        inp.profile.get("weight_kg"), int | float
    ):
        cm, kg = float(inp.profile["height_cm"]), float(inp.profile["weight_kg"])
        source = "你资料里的"
    if cm is not None and kg is not None and 120 <= cm <= 230 and 25 <= kg <= 300:
        value = _fmt(bmi(cm, kg))
        label = f"BMI 约 {value}（{source}身高 {_fmt(cm)} 厘米、体重 {_fmt(kg)} 公斤）"
        detail = "BMI = 体重（公斤）÷ 身高（米）²；成人低于 18.5 为体重过低，18.5–23.9 为正常。"
        cite = Citation(kind="calculation", ref="calc:bmi", label=label, detail=detail)
        return ToolResult("bmi", (cite,), extra={"bmi": value})
    stated = _BMI_RE.search(joined)
    if stated and 10 <= float(stated.group(1)) <= 60:
        value = stated.group(1)
        cite = Citation(
            kind="record",
            ref="said:bmi",
            label=f"你说的 BMI 是 {value}",
            detail="成人 BMI 低于 18.5 为体重过低，18.5–23.9 为正常。",
        )
        return ToolResult("bmi", (cite,), extra={"bmi": value})
    return None


def calorie_tool(inp: ToolInput) -> ToolResult | None:
    """每日热量与蛋白质目标：与计划链路同一个计算（engine.calories）。分层不给或信息不全时算不出。"""
    data = {k: inp.profile[k] for k in SCREENING_FIELDS if k in inp.profile}
    try:
        screening = Screening.model_validate_json(json.dumps(data))
    except ValidationError:
        return None
    raw_activity = inp.profile.get("activity")
    activity = (
        Activity(raw_activity) if raw_activity in {a.value for a in Activity} else NOT_RECORDED
    )
    tier = classify(screening, as_of=inp.as_of, tz=inp.tz, rulebook=inp.rulebook)
    result = calorie_target(
        screening, activity, tier, as_of=inp.as_of, tz=inp.tz, rulebook=inp.rulebook
    )
    if result.status != "ok" or not is_known(result.target_kcal) or not is_known(result.macros):
        return None
    kcal, protein = result.target_kcal, result.macros.protein_g
    cites = (
        Citation(
            kind="calculation",
            ref="calc:kcal_target",
            label=f"每日热量目标约 {kcal} 千卡（估算）",
            detail="按基础代谢、活动水平与分层缺口计算；不低于热量底线。规则：R-CAL-*",
        ),
        Citation(
            kind="calculation",
            ref="calc:protein_target",
            label=f"每日蛋白质约 {protein} 克",
            detail="按体重与规则表的蛋白质下限计算（R-CAL-PROTEIN）",
        ),
    )
    return ToolResult("calorie_target", cites, extra={"kcal": kcal, "protein_g": protein})


def _restrictions(profile: Mapping[str, Any]) -> frozenset[Restriction]:
    raw = profile.get("restrictions")
    known = {r.value for r in Restriction}
    if not isinstance(raw, list | tuple):
        return frozenset()
    return frozenset(Restriction(r) for r in raw if isinstance(r, str) and r in known)


_PATTERN_WORD = {
    "squat": "下肢蹲",
    "hinge": "髋铰链",
    "lunge": "单腿",
    "push": "上肢推",
    "pull": "上肢拉",
    "core": "核心",
    "cardio": "心肺",
}


def _find_exercise(lib: Library, text: str, context_item: str | None) -> str | None:
    if context_item and (m := re.search(r"exercise:([a-z][a-z0-9_]*)", context_item)):
        try:
            lib.get(m.group(1))
            return m.group(1)
        except KeyError:
            return None
    hits = sorted(
        ((len(e.name), e.id) for e in lib.exercises if len(e.name) >= 2 and e.name in text),
        reverse=True,
    )
    return hits[0][1] if hits else None


def exercise_tool(inp: ToolInput, lib: Library | None = None) -> ToolResult | None:
    """动作查询：按问题或页面条目找到动作，给出要点、常见错误、降阶或进阶、身体限制的提醒。"""
    lib = lib or default_library()
    exercise_id = _find_exercise(lib, inp.text, inp.context_item)
    if exercise_id is None:
        return None
    ex = lib.get(exercise_id)
    restrictions = _restrictions(inp.profile)
    env = inp.profile.get("environment")
    equipment = lib.environment(env if env in ("hotel", "home", "gym") else "hotel")
    cautions = [lib.restrictions[r].caution for r in ex.caution_if if r in restrictions]
    alternative: list[str] = []
    harder = "太简单" in inp.text
    step = ex.harder if harder else ex.easier
    if step is not None:
        alt = lib.get(step)
        alternative = [
            f"{'太简单时进阶' if harder else '做不了时先降阶'}：{alt.name}。{alt.key_points[0]}"
        ]
    else:
        subs = lib.substitutes(exercise_id, equipment=equipment, restrictions=restrictions)
        if subs:
            alternative = [f"可以换成：{subs[0].name}"]
    cites = [
        Citation(
            kind="knowledge",
            ref=f"exercise:{ex.id}",
            label=f"动作库《{ex.name}》",
            detail="；".join([*ex.key_points, *ex.common_errors]),
        ),
        *(
            Citation(
                kind="rule",
                ref=f"exercise-restriction:{r.value}",
                label=lib.restrictions[r].name,
                detail=lib.restrictions[r].caution,
            )
            for r in ex.caution_if
            if r in restrictions
        ),
    ]
    word = _PATTERN_WORD.get(str(ex.pattern), str(ex.pattern))
    return ToolResult(
        "exercise_lookup",
        tuple(cites),
        conclusion=f"「{ex.name}」是练{word}的动作。",
        how=(*ex.key_points, *alternative),
        caution=(*ex.common_errors, *cautions),
        extra={"exercise_id": ex.id},
    )


type Tool = Callable[[ToolInput], ToolResult | None]

TOOLS: dict[str, Tool] = {
    "bmi": bmi_tool,
    "calorie_target": calorie_tool,
    "exercise_lookup": exercise_tool,
}


def run_tools(names: tuple[str, ...], inp: ToolInput) -> list[ToolResult]:
    """依次运行声明的工具；单个工具出错或算不出时跳过。"""
    out = []
    for name in names:
        try:
            result = TOOLS[name](inp)
        except Exception:  # 工具出错不影响回答：这条事实缺失即可
            result = None
        if result is not None:
            out.append(result)
    return out
