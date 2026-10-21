"""能量目标（spec §30）：用热量计算给出每日热量与蛋白质目标。"""

import json
from datetime import timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ironm.contracts.agents import AgentOutput
from ironm.contracts.calories import Activity, CalorieTarget
from ironm.contracts.screening import Screening, Tier, TierResult
from ironm.contracts.unknown import Unknown, UnknownReason, not_recorded
from ironm.engine.calories import DEFICIT, calorie_target
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import Change, no_opinion, proposals
from ironm.skills.view import SnapshotView

KCAL_PER_KG = 7700  # 每千克体重约 7700 千卡（估算，用于说明预期速度）
SCREENING_FIELDS = ("birth_date", "sex", "height_cm", "weight_kg")


def _screening(view: SnapshotView) -> Screening:
    data: dict[str, Any] = {}
    for name in SCREENING_FIELDS:
        value = view.get(f"profile.{name}")
        if not isinstance(value, Unknown):
            data[name] = value
    return Screening.model_validate_json(json.dumps(data))


def _zone(view: SnapshotView) -> ZoneInfo:
    name = view.get("profile.timezone")
    try:
        return ZoneInfo(name) if isinstance(name, str) else ZoneInfo("UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")  # 只影响生日当天的周岁，误差不超过一天


def _tier(view: SnapshotView) -> TierResult:
    tier = view.get("tier")
    matched = view.get("tier_matched")
    return TierResult(
        tier=tier if isinstance(tier, Tier) else Unknown(unknown=UnknownReason.INSUFFICIENT_DATA),
        status="ok" if isinstance(tier, Tier) else "insufficient",
        matched=matched if isinstance(matched, tuple) else (),
        reasons=(),
        missing=(),
        bmi=not_recorded(),
        rule_id="R-SF-TIER",
    )


def _facts(result: CalorieTarget) -> tuple[str, ...]:
    facts = [
        f"静息能量消耗约 {result.ree_kcal} 千卡",
        f"每日总消耗约 {result.tdee_kcal} 千卡（活动水平 {result.pal}）",
        f"缺口 {result.deficit_kcal} 千卡",
    ]
    notes = {
        "collaborative_minimum_intake": "用药协作层：这是每天至少要吃到的量，不是上限",
        "senior_maintenance": "65 岁以上且体重在适宜范围：只给维持热量",
        "floor_applied": "已按热量底线抬高",
        "no_deficit_possible": "底线高于每日总消耗：靠增加活动形成缺口",
        "protein_raised": "蛋白质按每千克参考体重提高",
    }
    facts += [notes[r] for r in result.reasons if r in notes]
    return tuple(facts)


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    activity = view.get("profile.activity")
    high_protein = view.get("profile.high_protein")
    result = calorie_target(
        _screening(view),
        Activity(activity)
        if isinstance(activity, str) and activity in set(Activity)
        else not_recorded(),
        _tier(view),
        as_of=view.as_of,
        tz=_zone(view),
        rulebook=context.rulebook,
        high_protein=high_protein is True,
    )
    if result.status != "ok":
        detail = ",".join(result.missing or result.reasons)
        return no_opinion(context, f"calories_{result.status}:{detail}")
    deficit = result.deficit_kcal if isinstance(result.deficit_kcal, int) else 0
    weekly_kg = round(deficit * 7 / KCAL_PER_KG, 1)
    expected = f"体重每周下降约 {weekly_kg} 千克（估算）" if deficit else "维持当前体重"
    rollback = "两周内体重下降快于每周 1 千克，或持续饥饿、疲劳、头晕时，回到上一目标并复核"
    review = (view.as_of + timedelta(days=14)).date()
    facts = _facts(result)
    protein = result.macros.protein_g if not isinstance(result.macros, Unknown) else None
    changes = [
        Change(
            field="daily_kcal_target",
            to_value=result.target_kcal,
            reason_facts=facts,
            rule_id=DEFICIT,
            expected_effect=expected,
            rollback_condition=rollback,
        )
    ]
    if protein is not None:
        changes.append(
            Change(
                field="daily_protein_g_target",
                to_value=protein,
                reason_facts=facts,
                rule_id="R-CAL-PROTEIN",
                expected_effect="减脂期间尽量保住肌肉量",
                rollback_condition=rollback,
            )
        )
    return proposals(context, view, changes, review)
