"""每周复盘、两周饮食复盘与到期复核（spec [2026-10-21] §17）。

复盘：算事实 → 安全官给约束 → 复盘 Skill 按决策表提出修改 → 整合 → 盲审 → 终审 → 发布新的计划版本，
并写决策日志。两周饮食复盘（只对减脂）与当周训练复盘合并成一个版本。

到期复核：调整到了复核日，进阶类的修改若两周内完成率低于一半或平均 RPE ≥9，撤回到原来的值（同样
经过三道关，生成新版本）；降热量的修改若减得太快，撤回；数据不足延后一周；其余保留。降低负荷的
调整不自动撤回，由每周复盘按决策表再进阶。
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from ironm.agents.base import Agent
from ironm.contracts.agents import (
    AgentOutput,
    Constraint,
    ConstraintOutput,
    NoOpinionOutput,
    Proposal,
    ProposalOutput,
)
from ironm.contracts.common import deep_thaw
from ironm.contracts.events import Event, WeightEvent
from ironm.contracts.plan import FIELD_OWNERS, PlanField, PlanVersion
from ironm.contracts.roles import AgentId
from ironm.contracts.screening import SafetyFlag, Tier, TierResult
from ironm.contracts.skills import ScenePack
from ironm.contracts.unknown import is_known
from ironm.engine.review_facts import WeekFacts, week_facts, week_start
from ironm.engine.trend import weight_trend
from ironm.orchestrator.evidence import build_cards
from ironm.orchestrator.first_plan import build_snapshot, current_tier
from ironm.orchestrator.gates import Candidate, blind, blind_review, final_gates, integrate
from ironm.orchestrator.onboarding import DEFAULT_SCHEDULES, GOAL_SCENES
from ironm.rules.loader import RuleBook
from ironm.skills.library.nutritionist.diet_adjust.skill import RULE as DIET_RULE
from ironm.skills.library.nutritionist.diet_adjust.skill import DietParams, slow_loss
from ironm.skills.library.training_planner.weekly_adjust.skill import RULE as WEEKLY_RULE
from ironm.skills.library.training_planner.weekly_adjust.skill import WeeklyParams, needs_recovery
from ironm.skills.loader import LoadedSkill, select
from ironm.store.plans import PlanStore, StoredPlan
from ironm.store.reviews import Adjustment, DecisionStore

__all__ = ["ReviewDeps", "run_recheck", "run_review"]

_F = PlanField
STALE_WEIGHT_DAYS = 10  # 超过 10 天没称重，体重数据视为过期（spec [2026-10-20] §15）


@dataclass(frozen=True)
class ReviewDeps:
    plans: PlanStore
    decisions: DecisionStore
    rulebook: RuleBook
    skills: tuple[LoadedSkill, ...]


def _run(
    agent: AgentId,
    intent: str,
    snap: Any,
    deps: ReviewDeps,
    tier: TierResult,
    scene: ScenePack,
) -> tuple[AgentOutput, ...]:
    sel = select(deps.skills, agent=agent, intent=intent, scene=scene, tier=tier)
    return tuple(Agent(agent).execute(snap, sel, deps.rulebook))


def _size(value: Any) -> float | None:
    """比较大小用的数：数值本身；槽位按总组数与次数上限。"""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, list | tuple) and value and all(isinstance(s, Mapping) for s in value):
        return float(sum(s.get("sets", 0) * 100 + s.get("reps", [0, 0])[1] for s in value))
    return None


def _direction(field: str, before: Any, after: Any) -> Literal["up", "down", "same"]:
    a, b = _size(before), _size(after)
    if a is None or b is None or a == b:
        return "same"
    return "up" if b > a else "down"


def _category(c_field: PlanField, before: Any, after: Any, source: str) -> str:
    if source == "rollback":
        return "recovery"
    direction = _direction(c_field.value, before, after)
    if c_field is _F.DAILY_KCAL_TARGET:  # 多吃一点（减得太快）偏安全；少吃一点是目标进度
        return "safety" if direction == "up" else "progress"
    return "recovery" if direction == "down" else "progress"


def _candidates(
    outputs: Iterable[AgentOutput], source_of: Mapping[AgentId, str]
) -> list[Candidate]:
    out = []
    for o in outputs:
        if not isinstance(o, ProposalOutput):
            continue
        for p in o.items:
            before, after = deep_thaw(p.from_value), deep_thaw(p.to_value)
            source = source_of.get(o.agent, "review")
            out.append(
                Candidate(
                    proposal=p,
                    agent=o.agent,
                    skill=o.skill,
                    source=source,
                    category=_category(p.field, before, after, source),  # type: ignore[arg-type]
                )
            )
    return out


def _last_weigh_in(events: Iterable[Event], as_of: datetime) -> datetime | None:
    times = [e.occurred_at for e in events if isinstance(e, WeightEvent) and e.occurred_at <= as_of]
    return max(times) if times else None


def _publish(
    user_id: str,
    *,
    stored: StoredPlan,
    candidates: list[Candidate],
    outputs: Sequence[AgentOutput],
    constraints: Sequence[Constraint],
    tier: TierResult,
    profile: Mapping[str, Any],
    deps: ReviewDeps,
    now: datetime,
    kind: str,
    data_fresh: bool,
) -> dict[str, Any]:
    """三道关之后发布；返回决策日志中“修改与审查”的部分。"""
    integrated = integrate(candidates, constraints=constraints, profile=profile)
    blinds = blind(integrated.accepted)
    keys = {b.key: c for b, c in zip(blinds, integrated.accepted, strict=True)}
    known_tier = tier.tier if is_known(tier.tier) and isinstance(tier.tier, Tier) else None
    lanes = blind_review(blinds, tier=known_tier, constraints=constraints, rulebook=deps.rulebook)
    accepted = [c for k, c in keys.items() if lanes.blocked(k) is None]
    blocked = [(c, lanes.blocked(k)) for k, c in keys.items() if lanes.blocked(k) is not None]
    plan_data: dict[str, Any] | None = None
    cards: list[Any] = []
    old_fields: dict[str, Any] = dict(stored.record["plan"]["fields"])
    if accepted:
        fields = dict(old_fields)
        for c in accepted:
            fields[c.proposal.field.value] = deep_thaw(c.proposal.to_value)
        previous = stored.record["plan"]
        used = {c.skill.split("@")[0]: c.skill.split("@")[1] for c in accepted}
        data = {
            "version": stored.version + 1,
            "parent": stored.version,
            "fields": fields,
            "diff": [
                {"field": f, "from_value": old_fields.get(f), "to_value": fields[f]}
                for f in fields
                if old_fields.get(f) != fields[f]
            ],
            "rule_set_versions": deps.rulebook.versions(),
            "skill_versions": {**previous.get("skill_versions", {}), **used},
            "created_at": now.isoformat(),
        }
        try:
            plan_data = PlanVersion.model_validate_json(
                json.dumps(data, ensure_ascii=False)
            ).model_dump(mode="json")
        except ValidationError:
            plan_data = None
        cards = list(
            build_cards(
                outputs, deps.skills, deps.rulebook, only=[c.proposal.proposal_id for c in accepted]
            )
        )
    gates = final_gates(
        plan=plan_data if accepted else {"rule_set_versions": 1, "skill_versions": 1},
        accepted=accepted,
        cards=cards,
        constraints=constraints,
        lanes=lanes,
        keys=keys,
        data_fresh=data_fresh,
    )
    published = None
    if accepted and gates.publish and plan_data is not None:
        changed = {c.proposal.field.value for c in accepted}
        kept_cards = [c for c in stored.record.get("cards", []) if c["field"] not in changed]
        record = {
            **stored.record,
            "plan": plan_data,
            "cards": [*kept_cards, *(c.model_dump(mode="json") for c in cards)],
            "tier": tier.model_dump(mode="json"),
        }
        published = deps.plans.add(user_id, stored.version, record, kind, now)
        for c in accepted:
            if c.source == "rollback":
                continue
            before = old_fields.get(c.proposal.field.value)
            deps.decisions.add_adjustment(
                Adjustment(
                    adjustment_id=f"{user_id}:{published}:{c.proposal.field.value}",
                    user_id=user_id,
                    field=c.proposal.field.value,
                    from_value=before,
                    to_value=deep_thaw(c.proposal.to_value),
                    direction=_direction(c.proposal.field.value, before, c.proposal.to_value),
                    plan_version=published,
                    review_date=c.proposal.review_date,
                    status="pending",
                )
            )
    return {
        "published_version": published,
        "changes": [
            {
                "field": c.proposal.field.value,
                "from": old_fields.get(c.proposal.field.value),
                "to": deep_thaw(c.proposal.to_value),
                "why": list(c.proposal.reason_facts),
                "review_date": c.proposal.review_date.isoformat(),
            }
            for c in accepted
        ]
        if published
        else [],
        "not_published": [
            {"field": c.proposal.field.value, "reason": reason}
            for c, reason in [*integrated.rejected, *blocked]
        ]
        + (
            [{"field": c.proposal.field.value, "reason": "gates"} for c in accepted]
            if accepted and not published
            else []
        ),
        "lanes": {
            keys[k].proposal.field.value: {
                "safety": lanes.safety[k],
                "evidence": lanes.evidence[k],
                "quality": lanes.quality[k],
            }
            for k in keys
        },
        "gates": {"blocking": list(gates.blocking), "warnings": list(gates.warnings)},
    }


def _recovery_streak(user_id: str, week: date, deps: ReviewDeps) -> int:
    """这一周之前，连续几周命中了“疲劳高或睡眠差”（按各周决策日志里记下的事实判断）。"""
    p = deps.rulebook.get(WEEKLY_RULE).params_as(WeeklyParams)
    earlier = {
        d.period: d.record
        for d in deps.decisions.recent(user_id, kind="weekly", limit=p.recovery_max_weeks + 4)
    }
    streak = 0
    while (
        record := earlier.get((week - timedelta(days=7 * (streak + 1))).isoformat())
    ) is not None and needs_recovery(record.get("facts", {}).get("this_week", {}), p):
        streak += 1
    return streak


def _summary(changes: Sequence[Mapping[str, Any]], reasons: Sequence[str]) -> str:
    names = {
        "weekly_cardio_minutes": "每周有氧分钟数",
        "session_slots": "每个动作的组数与次数",
        "daily_kcal_target": "每日热量目标",
    }
    if changes:
        parts = [f"{names.get(c['field'], c['field'])}：{c['from']} → {c['to']}" for c in changes]
        when = min(c["review_date"] for c in changes)
        return f"这次调整了{'；'.join(parts)}。{changes[0]['why'][0]}。{when} 再看效果。"
    text = {
        "weekly_insufficient": "这周记录太少，先不调整。",
        "weekly_partial_week": "这周只记录了一部分，先不调整。",
        "weekly_keep": "这周按计划进行，不需要调整。",
        "weekly_recovery_streak": "已经连续 3 周疲劳高或睡得差：先不再减量，这几天以休息为主；"
        "持续不缓解请就医。",
        "diet_insufficient": "饮食或称重记录还不够判断，饮食先不变，下次再看。",
        "diet_simplify_first": "先把训练完成好，饮食暂时不变。",
        "diet_keep": "体重变化在目标范围内，饮食不变。",
        "diet_at_floor": "热量已经到底线，不再往下调；可以多走路或增加一点有氧。",
        "diet_red_flag": "这段时间有需要注意的身体情况，饮食先不变，必要时请咨询医生。",
    }
    lines = [text[r] for r in reasons if r in text]
    return "".join(dict.fromkeys(lines)) or "这次没有调整。"


def run_review(
    user_id: str,
    week: date,
    *,
    deps: ReviewDeps,
    events: Sequence[Event],
    profile: Mapping[str, Any],
    flags: Sequence[SafetyFlag],
    as_of: datetime,
    now: datetime,
    tz: ZoneInfo,
) -> dict[str, Any]:
    """week：这一周的周一（用户时区）；as_of：数据截止时刻（周期结束或签到时刻）。"""
    stored = deps.plans.latest(user_id)
    if stored is None:
        empty: dict[str, Any] = {
            "week": week.isoformat(),
            "decision": "no_plan",
            "summary": "还没有计划。",
        }
        deps.decisions.add(user_id, "weekly", week.isoformat(), empty, now)
        return empty
    fields: Mapping[str, Any] = stored.record["plan"]["fields"]
    plan = PlanVersion.model_validate_json(json.dumps(stored.record["plan"], ensure_ascii=False))
    goal = stored.record.get("goal", "fitness")
    scene = GOAL_SCENES.get(goal, ScenePack.FITNESS)
    tier = current_tier(profile, as_of=as_of, tz=tz, rulebook=deps.rulebook, flags=flags)
    days = fields.get("training_days_per_week")
    schedule = fields.get("training_schedule") or DEFAULT_SCHEDULES.get(
        days if isinstance(days, int) else 2, []
    )
    this = week_facts(
        events, schedule=schedule, start=week, tz=tz, as_of=as_of, rulebook=deps.rulebook
    )
    prev = week_facts(
        events,
        schedule=schedule,
        start=week - timedelta(days=7),
        tz=tz,
        as_of=as_of,
        rulebook=deps.rulebook,
    )
    review: dict[str, Any] = {
        "completion": this.completion,
        "prev_completion": prev.completion,
        "avg_rpe": this.avg_rpe,
        "fatigue": this.fatigue,
        "fatigue_baseline": this.fatigue_baseline,
        "sleep_quality": this.sleep_quality,
        "partial_week": this.partial_week,
        "menstruation": this.menstruation,
        "recovery_weeks_before": _recovery_streak(user_id, week, deps),
    }
    first = deps.plans.history(user_id)[0]
    started = first.at.astimezone(tz).date()
    diet_due = goal == "fat_loss" and ((week - week_start(started)).days // 7) % 2 == 1
    diet = (
        _diet_facts(events, flags, this, prev, week, started, as_of, tz, deps, tier)
        if diet_due
        else {}
    )
    review.update(diet)
    intent = "review.biweekly" if diet_due else "review.weekly"
    who = {**profile, "user_id": user_id}
    base = build_snapshot(who, tier, as_of, tz, plan=plan, review=review)
    constraints = tuple(
        c
        for o in _run(AgentId.SAFETY_OFFICER, intent, base, deps, tier, scene)
        if isinstance(o, ConstraintOutput)
        for c in o.items
    )
    snap = build_snapshot(who, tier, as_of, tz, constraints, plan, review)
    outputs = list(_run(AgentId.TRAINING_PLANNER, "review.weekly", snap, deps, tier, scene))
    if diet_due:
        outputs += _run(AgentId.NUTRITIONIST, "review.biweekly", snap, deps, tier, scene)
    reasons = [o.reason_code for o in outputs if isinstance(o, NoOpinionOutput) and o.reason_code]
    candidates = _candidates(
        outputs,
        {AgentId.TRAINING_PLANNER: "weekly_review", AgentId.NUTRITIONIST: "diet_review"},
    )
    last_weight = _last_weigh_in(events, as_of)
    stale = last_weight is None or (as_of - last_weight).days > STALE_WEIGHT_DAYS
    diet_change = any(c.proposal.field is _F.DAILY_KCAL_TARGET for c in candidates)
    result = _publish(
        user_id,
        stored=stored,
        candidates=candidates,
        outputs=outputs,
        constraints=constraints,
        tier=tier,
        profile=profile,
        deps=deps,
        now=now,
        kind="weekly_review",
        data_fresh=not (diet_change and stale),
    )
    insufficient = [r for r in reasons if r.endswith("insufficient")]
    earlier = deps.decisions.recent(user_id, kind="weekly", limit=1)
    repeat = bool(earlier) and "insufficient" in earlier[0].record.get("decision", "")
    record: dict[str, Any] = {
        "week": week.isoformat(),
        "facts": {
            "this_week": this.as_review(),
            "last_week": prev.as_review(),
            **{k: v for k, v in review.items() if k.startswith(("loss", "ci_", "meal", "weigh"))},
        },
        "inference": reasons,
        "uncertain": [
            *(["本周没有记 RPE，按完成率判断"] if this.avg_rpe is None else []),
            *(["本周没有签到"] if this.fatigue is None and this.sleep_quality is None else []),
        ],
        "decision": "adjusted"
        if result["published_version"]
        else ("insufficient" if insufficient else "kept"),
        "diet_review": diet_due,
        # 饮食复盘用到的全部事实（spec §17 决策日志：事实写全，复盘可回放、可核对）
        "diet_facts": diet,
        **result,
        "summary": _summary(result["changes"], reasons),
        "suggestion": "连续两次记录不够：可以换一个更方便的称重日，或改用更省事的记录方式。"
        if insufficient and repeat
        else None,
    }
    deps.decisions.add(user_id, "weekly", week.isoformat(), record, now)
    return record


def _diet_facts(
    events: Sequence[Event],
    flags: Sequence[SafetyFlag],
    this: WeekFacts,
    prev: WeekFacts,
    week: date,
    started: date,
    as_of: datetime,
    tz: ZoneInfo,
    deps: ReviewDeps,
    tier: TierResult,
) -> dict[str, Any]:
    p = deps.rulebook.get(DIET_RULE).params_as(DietParams)
    weights = [e for e in events if isinstance(e, WeightEvent)]

    def loss(until: datetime) -> tuple[float | None, float | None]:
        fit = weight_trend(
            weights, as_of=until, tz=tz, window_days=p.trend_window_days, rulebook=deps.rulebook
        )
        if fit.status != "ok" or not is_known(fit.slope_kg_per_week):
            return None, None
        half = (
            (float(fit.ci_high) - float(fit.ci_low)) / 2
            if is_known(fit.ci_high) and is_known(fit.ci_low)
            else None
        )
        return -float(fit.slope_kg_per_week), half

    now_loss, half = loss(as_of)
    before_loss, _ = loss(as_of - timedelta(days=p.trend_window_days))
    period_start = datetime.combine(week - timedelta(days=7), datetime.min.time(), tzinfo=tz)
    red = any(f.at >= period_start and f.level in ("emergency", "redflag") for f in flags)
    completions = [c for c in (this.completion, prev.completion) if c is not None]
    adherence = [d for d in (this.diet_adherence_days, prev.diet_adherence_days) if d is not None]
    return {
        "red_flag": red,
        "meal_days_min": min(this.meal_log_days, prev.meal_log_days),
        "weigh_ins_min": min(this.weigh_ins, prev.weigh_ins),
        "loss_kg_per_week": now_loss,
        "ci_half_width": half,
        "completion": round(sum(completions) / len(completions), 3) if completions else None,
        "adherence_days": round(sum(adherence) / len(adherence), 1) if adherence else None,
        "prev_slow": before_loss is not None and slow_loss(before_loss, tier.tier, p),
        "days_since_start": (as_of.astimezone(tz).date() - started).days,
        "prev_loss_kg_per_week": before_loss,  # 上一周期的减重速度（判断“连续两个周期”用）
    }


def run_recheck(
    user_id: str,
    today: date,
    *,
    deps: ReviewDeps,
    events: Sequence[Event],
    profile: Mapping[str, Any],
    flags: Sequence[SafetyFlag],
    now: datetime,
    tz: ZoneInfo,
) -> dict[str, Any] | None:
    """到复核日的调整：保留、撤回（新版本）或延后一周。没有到期的调整时返回 None。"""
    due = deps.decisions.due_adjustments(user_id, today)
    stored = deps.plans.latest(user_id)
    if not due or stored is None:
        return None
    fields: Mapping[str, Any] = stored.record["plan"]["fields"]
    days = fields.get("training_days_per_week")
    schedule = fields.get("training_schedule") or DEFAULT_SCHEDULES.get(
        days if isinstance(days, int) else 2, []
    )
    monday = week_start(today)
    weeks = [
        week_facts(
            events,
            schedule=schedule,
            start=monday - timedelta(days=7 * k),
            tz=tz,
            as_of=now,
            rulebook=deps.rulebook,
        )
        for k in (1, 2)
    ]
    p = deps.rulebook.get(DIET_RULE).params_as(DietParams)
    outcomes: list[dict[str, Any]] = []
    rollbacks: list[tuple[Adjustment, Proposal]] = []
    for adj in due:
        current = fields.get(adj.field)
        if current != adj.to_value:
            deps.decisions.settle(adj.adjustment_id, "superseded", now)
            outcomes.append({"adjustment": adj.adjustment_id, "result": "superseded"})
            continue
        bad, insufficient = False, False
        if adj.field == _F.DAILY_KCAL_TARGET.value and adj.direction == "down":
            fit = weight_trend(
                [e for e in events if isinstance(e, WeightEvent)],
                as_of=now,
                tz=tz,
                window_days=p.trend_window_days,
                rulebook=deps.rulebook,
            )
            if fit.status != "ok" or not is_known(fit.slope_kg_per_week):
                insufficient = True
            else:
                bad = -float(fit.slope_kg_per_week) > p.fast_loss_kg_per_week
        elif adj.direction == "up":
            recorded = [w for w in weeks if w.completion is not None]
            insufficient = not recorded
            bad = any(
                (w.completion is not None and w.completion < 0.5)
                or (w.avg_rpe is not None and w.avg_rpe >= 9)
                for w in weeks
            )
        if bad:
            rollbacks.append((adj, _rollback(adj, today)))
        elif insufficient:
            deps.decisions.settle(
                adj.adjustment_id, "pending", now, review_date=today + timedelta(days=7)
            )
            outcomes.append({"adjustment": adj.adjustment_id, "result": "postponed"})
        else:
            deps.decisions.settle(adj.adjustment_id, "kept", now)
            outcomes.append({"adjustment": adj.adjustment_id, "result": "kept"})
    result: dict[str, Any] = {"published_version": None}
    if rollbacks:
        tier = current_tier(profile, as_of=now, tz=tz, rulebook=deps.rulebook, flags=flags)
        plan = PlanVersion.model_validate_json(
            json.dumps(stored.record["plan"], ensure_ascii=False)
        )
        goal = stored.record.get("goal", "fitness")
        scene = GOAL_SCENES.get(goal, ScenePack.FITNESS)
        base = build_snapshot({**profile, "user_id": user_id}, tier, now, tz, plan=plan)
        constraints = tuple(
            c
            for o in _run(AgentId.SAFETY_OFFICER, "plan.adjust", base, deps, tier, scene)
            if isinstance(o, ConstraintOutput)
            for c in o.items
        )
        outputs, candidates = [], []
        skills_by_field = {c["field"]: c["skill"] for c in stored.record.get("cards", [])}
        for adj, proposal in rollbacks:
            agent = FIELD_OWNERS[PlanField(adj.field)]
            skill = skills_by_field.get(adj.field) or _review_skill(deps.skills, agent)
            output = ProposalOutput.model_validate_json(
                json.dumps(
                    {
                        "agent": agent.value,
                        "skill": skill,
                        "kind": "proposal",
                        "items": [proposal.model_dump(mode="json")],
                    },
                    ensure_ascii=False,
                )
            )
            outputs.append(output)
            candidates.append(
                Candidate(proposal, agent, skill, source="rollback", category="recovery")
            )
        result = _publish(
            user_id,
            stored=stored,
            candidates=candidates,
            outputs=outputs,
            constraints=constraints,
            tier=tier,
            profile=profile,
            deps=deps,
            now=now,
            kind="rollback",
            data_fresh=True,
        )
        rolled = {c["field"] for c in result["changes"]}
        for adj, _ in rollbacks:
            status = "rolled_back" if adj.field in rolled else "pending"
            if status == "rolled_back":
                deps.decisions.settle(adj.adjustment_id, "rolled_back", now)
            else:  # 撤回没通过三道关：一周后再看
                deps.decisions.settle(
                    adj.adjustment_id, "pending", now, review_date=today + timedelta(days=7)
                )
            outcomes.append({"adjustment": adj.adjustment_id, "result": status})
    record = {
        "date": today.isoformat(),
        "outcomes": outcomes,
        **result,
        "summary": "撤回了效果不好的调整，回到原来的安排。"
        if result.get("published_version")
        else "到期的调整已检查。",
    }
    deps.decisions.add(user_id, "recheck", today.isoformat(), record, now)
    return record


def _review_skill(skills: Sequence[LoadedSkill], agent: AgentId) -> str:
    for s in skills:
        if s.manifest.owner_agent == agent and s.manifest.id in ("weekly_adjust", "diet_adjust"):
            return s.manifest.versioned
    raise KeyError(agent)


def _rollback(adj: Adjustment, today: date) -> Proposal:
    data = {
        "proposal_id": f"rollback:{adj.field}",
        "field": adj.field,
        "from_value": adj.to_value,
        "to_value": adj.from_value,
        "reason_facts": [
            "到期复核：调整后两周内完成率低于一半或平均 RPE ≥9（或减得太快），回到调整前的安排"
        ],
        "rule_id": "R-RV-DIET" if adj.field == _F.DAILY_KCAL_TARGET.value else "R-RV-WEEKLY",
        "expected_effect": "回到能稳定完成的安排",
        "rollback_condition": "两周内能稳定完成后，按每周复盘再进阶",
        "review_date": (today + timedelta(days=14)).isoformat(),
        "evidence_ids": [],
    }
    return Proposal.model_validate_json(json.dumps(data, ensure_ascii=False))
