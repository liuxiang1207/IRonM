"""建档、计划版本与今日任务（spec [2026-10-21] §15）。

- 建档：把资料并入画像（新版本），按用户选的目标匹配 Skill 生成第一版计划；结果三选一：
  计划、需要补充的信息（追问）、不能给计划。
- 计划版本只增不改：再次建档生成新版本，带上一版本号与字段差异。
- 今日任务按当前计划与用户时区的当天日期给出。
"""

import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.calories import Activity
from ironm.contracts.exercises import Restriction
from ironm.contracts.plan import PlanVersion
from ironm.contracts.screening import Condition, SafetyFlag, Sex
from ironm.contracts.skills import ScenePack
from ironm.engine.review_facts import ReturnState
from ironm.knowledge.exercises import Library
from ironm.memory.repo import MemoryRepo
from ironm.orchestrator.first_plan import BLOCKING, FirstPlan, first_plan
from ironm.rules.loader import RuleBook
from ironm.skills.loader import LoadedSkill
from ironm.store.plans import PlanStore, StoredPlan

__all__ = ["Intake", "active_mode", "history_view", "onboard", "plan_view", "today_tasks"]

GOAL_SCENES: dict[str, ScenePack] = {
    "fitness": ScenePack.FITNESS,
    "strength": ScenePack.STRENGTH,
    "fat_loss": ScenePack.FAT_LOSS,
    "habits": ScenePack.HABITS,
    "maintenance": ScenePack.MAINTENANCE,
}
LEVELS = {"beginner": 1, "intermediate": 2, "advanced": 3}

CONDITION_LABELS: dict[str, str] = {
    "hypertension": "高血压（在治疗）",
    "hypertension_uncontrolled": "高血压没控制住",
    "diabetes_no_medication": "糖尿病（没用药）",
    "diabetes_on_medication": "糖尿病（在用药）",
    "heart_disease": "心脏病",
    "kidney_disease": "肾病",
    "cerebrovascular_disease": "中风或短暂性脑缺血史",
    "recent_injury": "近期受伤",
    "injury_rehab": "在做伤后康复",
    "weight_loss_drug_supervised": "医生指导下用减重药",
    "weight_loss_drug_unsupervised": "自己在用减重药",
    "pregnant": "怀孕",
    "breastfeeding": "哺乳期",
    "postpartum_within_year": "产后一年内",
    "eating_disorder_history": "有进食障碍史",
}

# 追问的说法：问题编号就是画像项；没列出的项不接受回答
QUESTIONS: dict[str, tuple[str, list[dict[str, str]] | None]] = {
    "profile.conditions": (
        "有没有下面这些健康状况？没有就选“都没有”。",
        [{"value": k, "label": v} for k, v in CONDITION_LABELS.items()],
    ),
    "profile.birth_date": ("你的出生日期？", None),
    "profile.height_cm": ("身高多少厘米？", None),
    "profile.weight_kg": ("体重多少公斤？", None),
    "profile.sex": (
        "性别？",
        [{"value": "male", "label": "男"}, {"value": "female", "label": "女"}],
    ),
    "profile.activity": (
        "平时的活动量？",
        [
            {"value": "sedentary", "label": "久坐或轻度"},
            {"value": "moderate", "label": "中度"},
            {"value": "vigorous", "label": "重度"},
        ],
    ),
    "profile.waist_cm": ("腰围多少厘米？可以不填。", None),
}


class Intake(BaseModel):
    """建档资料。没填的项为空（不覆盖画像里已有的值）；健康状况填空列表表示“都没有”。"""

    model_config = ConfigDict(extra="forbid")

    birth_date: date | None = None
    sex: Sex | None = None
    height_cm: Annotated[float, Field(ge=100, le=250)] | None = None
    weight_kg: Annotated[float, Field(ge=25, le=300)] | None = None
    waist_cm: Annotated[float, Field(ge=40, le=200)] | None = None
    conditions: list[Condition] | None = None
    goal: Literal["fitness", "strength", "fat_loss", "habits", "maintenance"] = "fitness"
    environment: Literal["gym", "home", "hotel"] = "gym"
    available_days: Annotated[int, Field(ge=1, le=7)] | None = None
    training_level: Literal["beginner", "intermediate", "advanced"] | None = None
    activity: Activity | None = None
    restrictions: list[Restriction] = []
    current_cardio_minutes: Annotated[int, Field(ge=0, le=10080)] | None = None

    @classmethod
    def from_request(cls, body: Mapping[str, Any]) -> "Intake":
        """请求体：资料 + answers（{"profile.x": 值}）。回答并入对应的资料项再校验。"""
        data = {k: v for k, v in body.items() if k != "answers"}
        answers = body.get("answers") or {}
        if not isinstance(answers, Mapping):
            raise ValueError("answers 应是对象")
        unknown = [k for k in answers if k not in QUESTIONS]
        if unknown:
            raise ValueError(f"不认识的回答：{sorted(unknown)}")
        data.update({k.removeprefix("profile."): v for k, v in answers.items()})
        return cls.model_validate(data)


def _merge(profile: Mapping[str, Any], intake: Intake, tz: ZoneInfo) -> dict[str, Any]:
    merged = dict(profile)
    given = intake.model_dump(mode="json", exclude_none=True)
    if "training_level" in given:
        given["training_level"] = LEVELS[given["training_level"]]
    merged.update(given)
    merged["timezone"] = tz.key
    return merged


def _questions(result: FirstPlan) -> list[dict[str, Any]]:
    out = []
    for q in result.questions:
        text, options = QUESTIONS.get(q.item, (q.item, None))
        item: dict[str, Any] = {
            "question_id": q.item,
            "text": text,
            "blocking": q.fallback == BLOCKING,
        }
        if options is not None:
            item["options"] = options
        out.append(item)
    return out


def _next_version(plan: PlanVersion, previous: StoredPlan | None) -> PlanVersion:
    """第一版计划按上一版本重新编号：版本号加 1，差异只列新增或改变的项。"""
    if previous is None:
        return plan
    before: Mapping[str, Any] = previous.record["plan"]["fields"]
    fields = plan.model_dump(mode="json")["fields"]
    data = plan.model_dump(mode="json")
    data["version"] = previous.version + 1
    data["parent"] = previous.version
    data["diff"] = [
        {"field": f, "from_value": before.get(f), "to_value": v}
        for f, v in fields.items()
        if before.get(f) != v
    ]
    return PlanVersion.model_validate_json(json.dumps(data, ensure_ascii=False))


def plan_view(stored: StoredPlan) -> dict[str, Any]:
    record = stored.record
    plan = record["plan"]
    return {
        "version": stored.version,
        "created_at": plan["created_at"],
        "fields": plan["fields"],
        "cards": record.get("cards", []),
        "notices": record.get("notices", []),
        "baseline": record.get("baseline"),
        "goal": record.get("goal"),
    }


def onboard(
    user_id: str,
    intake: Intake,
    *,
    memory: MemoryRepo,
    plans: PlanStore,
    flags: Iterable[SafetyFlag],
    rulebook: RuleBook,
    skills: tuple[LoadedSkill, ...],
    as_of: datetime,
    tz: ZoneInfo,
) -> dict[str, Any]:
    """建档并生成计划。计划已被别的请求更新时抛 store.plans.VersionConflict。"""
    version, profile = memory.profile(user_id)
    merged = _merge(profile, intake, tz)
    if merged != profile:
        memory.save_profile(user_id, version, merged, "onboarding", as_of)
    result = first_plan(
        {**merged, "user_id": user_id},
        as_of=as_of,
        tz=tz,
        rulebook=rulebook,
        skills=skills,
        flags=flags,
        scene=GOAL_SCENES[intake.goal],
    )
    tier = result.tier.model_dump(mode="json")["tier"]
    base: dict[str, Any] = {"tier": tier, "notices": list(result.notices)}
    if result.plan is None:
        if any(q.fallback == BLOCKING for q in result.questions):
            return {**base, "status": "questions", "questions": _questions(result)}
        return {**base, "status": "not_allowed", "questions": []}
    previous = plans.latest(user_id)
    plan = _next_version(result.plan, previous)
    record = {
        "plan": plan.model_dump(mode="json"),
        "cards": [c.model_dump(mode="json") for c in result.cards],
        "tier": result.tier.model_dump(mode="json"),
        "notices": list(result.notices),
        "baseline": result.baseline.model_dump(mode="json"),
        "goal": intake.goal,
    }
    expected = 0 if previous is None else previous.version
    plans.add(user_id, expected, record, "onboarding", as_of)
    stored = plans.latest(user_id)
    assert stored is not None
    return {**base, "status": "plan", "questions": _questions(result), "plan": plan_view(stored)}


def history_view(stored: list[StoredPlan]) -> list[dict[str, Any]]:
    """每一版改了哪些项、为什么改（理由取自该项的依据卡）。"""
    out = []
    before: Mapping[str, Any] = {}
    for s in stored:
        fields: Mapping[str, Any] = s.record["plan"]["fields"]
        cards = {c["field"]: c for c in s.record.get("cards", [])}
        changes = [
            {
                "field": f,
                "from": before.get(f),
                "to": fields.get(f),
                "reason": "；".join(cards[f]["facts"]) if f in cards else None,
            }
            for f in sorted(set(before) | set(fields))
            if before.get(f) != fields.get(f)
        ]
        out.append(
            {
                "version": s.version,
                "created_at": s.record["plan"]["created_at"],
                "reason": s.reason,
                "changes": changes,
            }
        )
        before = fields
    return out


DEFAULT_SCHEDULES = {1: [2], 2: [0, 3], 3: [0, 2, 4], 4: [0, 1, 3, 4], 5: [0, 1, 2, 3, 4]}


def _named(lib: Library | None, exercise_id: str) -> dict[str, Any]:
    if lib is None:
        return {"exercise_id": exercise_id, "exercise": exercise_id, "key_points": []}
    try:
        e = lib.get(exercise_id)
    except KeyError:
        return {"exercise_id": exercise_id, "exercise": exercise_id, "key_points": []}
    return {"exercise_id": exercise_id, "exercise": e.name, "key_points": list(e.key_points)}


def active_mode(mode: Mapping[str, Any] | None, today: date) -> dict[str, Any] | None:
    """出差 / 节假日：到期（结束日之后）自动恢复。"""
    if not mode or mode.get("mode") not in ("travel", "holiday"):
        return None
    try:
        until = date.fromisoformat(str(mode.get("until")))
    except ValueError:
        return None
    return {"mode": mode["mode"], "until": until.isoformat()} if today <= until else None


def today_tasks(
    stored: StoredPlan | None,
    today: date,
    *,
    library: Library | None = None,
    back: ReturnState | None = None,
    mode: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """今日任务：今天要不要练、练什么、最低任务、是不是称重日或签到日（spec §15）。

    叠加（spec §17）：出差用酒店 / 徒手版本、组数减半；节假日只留最低任务、饮食按维持；
    中断回归第一句说“欢迎回来”、训练量减半，中断超过 4 周先重做基线测试与安全筛查。
    """
    if stored is None:
        return {"status": "none"}
    fields: Mapping[str, Any] = stored.record["plan"]["fields"]
    weekday = today.weekday()
    days = fields.get("training_days_per_week")
    schedule = fields.get("training_schedule") or DEFAULT_SCHEDULES.get(
        days if isinstance(days, int) else 2, []
    )
    scheduled = weekday in schedule
    slots = {s["pattern"]: s for s in fields.get("session_slots") or []}
    current = active_mode(mode, today)
    picks = fields.get("exercise_selection") or []
    if current and current["mode"] == "travel":
        hotel = (fields.get("alternative_version") or {}).get("hotel") or []
        picks = [{**h, **_named(library, h["exercise_id"])} for h in hotel] or picks
    halve = (back is not None and back.status in ("welcome", "halved")) or bool(
        current and current["mode"] == "travel"
    )
    items = [
        {
            "slot": e["slot"],
            "exercise_id": e["exercise_id"],
            "exercise": e.get("name") or e.get("exercise") or e["exercise_id"],
            "sets": max(1, -(-slots[e["slot"]]["sets"] // 2))
            if halve
            else slots[e["slot"]]["sets"],
            "reps": slots[e["slot"]]["reps"],
            "rpe": slots[e["slot"]]["rpe"],
            "key_points": e.get("key_points", []),
            "cautions": e.get("cautions", []),
            "substitutes": e.get("substitutes", []),
        }
        for e in picks
        if e.get("slot") in slots
    ]
    if current and current["mode"] == "holiday":
        scheduled = False  # 节假日只留最低任务，不补练
    out: dict[str, Any] = {
        "status": "ok",
        "date": today.isoformat(),
        "weekday": weekday,
        "plan_version": stored.version,
        "training": {"scheduled": scheduled, "items": items if scheduled else []},
        "minimum_task": [
            {**m, **_named(library, m["exercise_id"])} for m in fields.get("minimum_task") or []
        ],
        "weekly_cardio_minutes": fields.get("weekly_cardio_minutes"),
        "steps_target": fields.get("daily_steps_target"),
        "weigh_in_today": weekday in (fields.get("weigh_in_days") or []),
        "checkin_today": weekday == fields.get("checkin_day"),
        "mode": current,
        "greeting": None,
        "next_step": None,
    }
    if current and current["mode"] == "holiday":
        out["diet_note"] = "节假日按维持吃，不设缺口；结束后自动恢复，不需要补练或节食补偿。"
    if back is not None and back.status in ("welcome", "rebaseline"):
        out["greeting"] = "欢迎回来！"
    if back is not None and back.status == "rebaseline":
        out["training"] = {"scheduled": False, "items": []}
        out["next_step"] = "先做一次基线测试和安全筛查，再按结果重新开始训练。"
    elif back is not None and back.status == "welcome":
        out["next_step"] = "今天先轻松练，组数减半，接下来一两周再慢慢回到原来的量。"
    return out
