"""发布前的三道关（spec [2026-10-21] §17）：整合 → 盲审 → 终审。

- 整合：越权丢弃；违反安全官约束（或无法比较）退回；跨字段一致；一次最多改 2 项，训练、饮食各
  最多 1 项，超出时按“安全 > 恢复 > 用户现实约束 > 目标进度”取舍。同一来源对同一领域的几处
  修改（如复盘同时调有氧和组数）算一项。
- 盲审：三路互不可见；审查时看不到修改出自哪个 Agent、哪个 Skill，也看不到用户身份。
- 终审 7 道闸门：格式、权限、底线、三件套、数据、审查、版本。只有阻断级问题卡住发布。
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from ironm.contracts.agents import Constraint, Proposal
from ironm.contracts.common import deep_thaw
from ironm.contracts.evidence import EvidenceCard
from ironm.contracts.plan import FIELD_OWNERS, PlanField
from ironm.contracts.roles import AgentId
from ironm.contracts.rules import SourceKind
from ironm.contracts.screening import Tier
from ironm.orchestrator.review import review_proposal
from ironm.rules.loader import RuleBook
from ironm.safety.decision import plan_allowed

__all__ = [
    "Blind",
    "Candidate",
    "GateResult",
    "Integrated",
    "LaneResults",
    "blind_review",
    "final_gates",
    "integrate",
]

_F = PlanField
TRAINING = frozenset(
    {
        _F.TRAINING_DAYS_PER_WEEK,
        _F.TRAINING_SCHEDULE,
        _F.SESSION_SLOTS,
        _F.INTENSITY_TARGET,
        _F.PROGRESSION_RULE,
        _F.DELOAD_SCHEDULE,
        _F.WEEKLY_CARDIO_MINUTES,
        _F.DAILY_STEPS_TARGET,
        _F.EXERCISE_SELECTION,
        _F.MINIMUM_TASK,
        _F.ALTERNATIVE_VERSION,
    }
)
DIET = frozenset(
    {
        _F.DIET_MODE,
        _F.DAILY_KCAL_TARGET,
        _F.DAILY_PROTEIN_G_TARGET,
        _F.PORTION_GUIDE,
        _F.EATING_OUT_STRATEGY,
    }
)
MAX_CHANGES = 2
# 优先级（数字越小越先保留）：安全 > 恢复 > 用户现实约束 > 目标进度
PRIORITY = {"safety": 0, "recovery": 1, "reality": 2, "progress": 3}


@dataclass(frozen=True)
class Candidate:
    proposal: Proposal
    agent: AgentId
    skill: str  # 编号@版本
    source: str  # weekly_review、diet_review、rollback 等
    category: Literal["safety", "recovery", "reality", "progress"]


@dataclass(frozen=True)
class Integrated:
    accepted: tuple[Candidate, ...]
    rejected: tuple[tuple[Candidate, str], ...]  # (修改, 原因代码)


def domain(field: PlanField) -> str:
    return "training" if field in TRAINING else "diet" if field in DIET else "other"


def _shape_ok(field: PlanField, value: Any) -> bool:
    """跨字段与取值范围的基本一致性（整合阶段，宁可退回）。"""
    if field is _F.SESSION_SLOTS:
        return isinstance(value, list) and all(
            isinstance(s, Mapping)
            and 1 <= s.get("sets", 0) <= 6
            and isinstance(s.get("reps"), list | tuple)
            and len(s["reps"]) == 2
            and 1 <= s["reps"][0] <= s["reps"][1] <= 30
            for s in value
        )
    if field is _F.WEEKLY_CARDIO_MINUTES:
        return isinstance(value, int) and 0 <= value <= 600
    if field is _F.DAILY_KCAL_TARGET:
        return isinstance(value, int) and 800 <= value <= 6000
    return True


def integrate(
    candidates: Iterable[Candidate],
    *,
    constraints: Sequence[Constraint],
    profile: Mapping[str, Any],
) -> Integrated:
    accepted: list[Candidate] = []
    rejected: list[tuple[Candidate, str]] = []
    for c in candidates:
        p = c.proposal
        if FIELD_OWNERS[p.field] != c.agent:
            rejected.append((c, "overreach"))
            continue
        verdict = review_proposal(p, constraints)
        if verdict.result != "pass":
            rejected.append((c, verdict.reason_code or "constraint"))
            continue
        value = deep_thaw(p.to_value)
        available = profile.get("available_days")
        too_many = (
            p.field is _F.TRAINING_DAYS_PER_WEEK
            and isinstance(available, int)
            and isinstance(value, int)
            and value > available
        )
        if too_many or not _shape_ok(p.field, value):
            rejected.append((c, "inconsistent"))
            continue
        accepted.append(c)
    # 同一来源、同一领域的几处修改算一项；每个领域最多一项，总共最多 MAX_CHANGES 项
    groups: dict[tuple[str, str], list[Candidate]] = {}
    for c in accepted:
        groups.setdefault((domain(c.proposal.field), c.source), []).append(c)
    ranked = sorted(
        groups.items(),
        key=lambda kv: (min(PRIORITY[c.category] for c in kv[1]), kv[0][1], kv[0][0]),
    )
    kept: list[Candidate] = []
    used_domains: set[str] = set()
    for (dom, _), members in ranked:
        if dom in used_domains or len(used_domains) >= MAX_CHANGES:
            rejected.extend((m, "over_limit") for m in members)
            continue
        used_domains.add(dom)
        kept.extend(members)
    return Integrated(tuple(kept), tuple(rejected))


# ---- 盲审 ----


@dataclass(frozen=True)
class Blind:
    """审查官看到的修改：没有 Agent、Skill 与用户身份。"""

    key: str  # 只用于把结果对回修改，不含来源信息
    field: PlanField
    from_value: Any
    to_value: Any
    rule_id: str
    facts: tuple[str, ...]
    expected_effect: str
    rollback_condition: str
    review_date: date


@dataclass(frozen=True)
class LaneResults:
    safety: dict[str, str]  # key → pass / reject:<原因>
    evidence: dict[str, str]  # key → supported / engineering / unsupported
    quality: dict[str, str]  # key → pass / needs_change / fail

    def blocked(self, key: str) -> str | None:
        """这项修改被拦下的原因；没有被拦下时为 None。"""
        if self.safety[key] != "pass":
            return f"safety:{self.safety[key]}"
        if self.evidence[key] == "unsupported":
            return "evidence:unsupported"
        if self.quality[key] == "fail":
            return "quality:fail"
        return None


def blind(candidates: Iterable[Candidate]) -> list[Blind]:
    out = []
    for n, c in enumerate(candidates):
        p = c.proposal
        out.append(
            Blind(
                key=f"m{n}",
                field=p.field,
                from_value=deep_thaw(p.from_value),
                to_value=deep_thaw(p.to_value),
                rule_id=p.rule_id,
                facts=p.reason_facts,
                expected_effect=p.expected_effect,
                rollback_condition=p.rollback_condition,
                review_date=p.review_date,
            )
        )
    return out


def _safety_lane(b: Blind, *, tier: Tier | None, constraints: Sequence[Constraint]) -> str:
    if tier is None or not plan_allowed(tier):
        return "reject:tier_no_plan"
    if not (b.expected_effect.strip() and b.rollback_condition.strip()):
        return "reject:missing_trio"
    for c in constraints:
        if c.field != b.field:
            continue
        limit = deep_thaw(c.value)
        lowered = (
            isinstance(b.from_value, int)
            and isinstance(b.to_value, int)
            and b.to_value < b.from_value
        )
        no_deficit = c.op == "exclude" and limit == "deficit"
        if no_deficit and b.field is _F.DAILY_KCAL_TARGET and lowered:
            return "reject:no_deficit_allowed"
        if isinstance(limit, int | float) and isinstance(b.to_value, int | float):
            if c.op == "min" and b.to_value < limit:
                return f"reject:below_{c.rule_id}"
            if c.op == "max" and b.to_value > limit:
                return f"reject:above_{c.rule_id}"
    return "pass"


def _evidence_lane(b: Blind, rulebook: RuleBook) -> str:
    try:
        rule = rulebook.get(b.rule_id)
    except KeyError:
        return "unsupported"
    if rule.status != "approved":
        return "unsupported"
    return "engineering" if rule.source.kind is SourceKind.DESIGN else "supported"


def _quality_lane(b: Blind) -> str:
    if not _shape_ok(b.field, b.to_value):
        return "fail"
    before, after = b.from_value, b.to_value
    numbers = isinstance(before, int) and isinstance(after, int) and before > 0
    big = numbers and abs(after - before) > before / 2  # 一次变化超过一半
    if b.field is _F.WEEKLY_CARDIO_MINUTES and big:
        return "needs_change"
    return "pass"


def blind_review(
    blinds: Sequence[Blind],
    *,
    tier: Tier | None,
    constraints: Sequence[Constraint],
    rulebook: RuleBook,
) -> LaneResults:
    """三路分别审，互不可见（各自只拿到同一份不含来源的修改）。"""
    return LaneResults(
        safety={b.key: _safety_lane(b, tier=tier, constraints=constraints) for b in blinds},
        evidence={b.key: _evidence_lane(b, rulebook) for b in blinds},
        quality={b.key: _quality_lane(b) for b in blinds},
    )


# ---- 终审 ----


@dataclass(frozen=True)
class GateResult:
    blocking: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def publish(self) -> bool:
        return not self.blocking


def final_gates(
    *,
    plan: Mapping[str, Any] | None,
    accepted: Sequence[Candidate],
    cards: Sequence[EvidenceCard],
    constraints: Sequence[Constraint],
    lanes: LaneResults,
    keys: Mapping[str, Candidate],
    data_fresh: bool,
) -> GateResult:
    """plan：待发布的计划版本（已按契约构造，构造失败时为 None）。"""
    blocking: list[str] = []
    warnings: list[str] = []
    if plan is None:
        blocking.append("format")  # 1 格式
    for c in accepted:  # 2 权限
        if FIELD_OWNERS[c.proposal.field] != c.agent:
            blocking.append(f"permission:{c.proposal.field.value}")
    if len({domain(c.proposal.field) for c in accepted}) > MAX_CHANGES:  # 3 底线（调整幅度）
        blocking.append("too_many_changes")
    for c in accepted:
        if review_proposal(c.proposal, constraints).result != "pass":
            blocking.append(f"floor:{c.proposal.field.value}")
    carded = {card.field for card in cards}
    for c in accepted:  # 4 三件套
        if c.proposal.field not in carded:
            blocking.append(f"trio:{c.proposal.field.value}")
    for card in cards:
        if not (card.expected_effect and card.rollback_condition and card.review_date):
            blocking.append(f"trio:{card.field.value}")
    if not data_fresh:  # 5 数据
        blocking.append("data_stale")
    for key, c in keys.items():  # 6 审查
        reason = lanes.blocked(key)
        if reason is not None and c in accepted:
            blocking.append(f"review:{c.proposal.field.value}:{reason}")
        if lanes.evidence[key] == "engineering":
            warnings.append(f"evidence_engineering:{c.proposal.field.value}")
        if lanes.quality[key] == "needs_change":
            warnings.append(f"quality_needs_change:{c.proposal.field.value}")
    versions = plan is not None and plan.get("rule_set_versions") and plan.get("skill_versions")
    if plan is not None and not versions:  # 7 版本
        blocking.append("version")
    return GateResult(tuple(blocking), tuple(warnings))
