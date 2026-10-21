"""冲刺 2 演示（spec §31）：几类模拟用户从建档到第一版计划。用法：make demo。"""

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ironm.contracts.common import deep_thaw
from ironm.knowledge.baseline import default_baseline
from ironm.orchestrator.first_plan import FirstPlan, first_plan
from ironm.rules.loader import RULES_DIR, load_rules
from ironm.skills.loader import load_skills

AS_OF = datetime(2026, 10, 21, 1, tzinfo=UTC)
TZ = ZoneInfo("Asia/Shanghai")

PLAN_FIELD_NAMES = {
    "training_days_per_week": "力量训练次数 / 周",
    "session_slots": "训练槽位",
    "intensity_target": "用力程度",
    "weekly_cardio_minutes": "有氧分钟 / 周",
    "exercise_selection": "具体动作",
    "minimum_task": "最低任务",
    "alternative_version": "酒店版本",
    "daily_kcal_target": "每日热量（千卡）",
    "daily_protein_g_target": "每日蛋白质（克）",
    "diet_mode": "饮食模式",
}

PERSONAS: dict[str, dict[str, Any]] = {
    "健身房 A 层（34 岁男，BMI 29）": {
        "birth_date": "1992-03-01",
        "sex": "male",
        "height_cm": 178.0,
        "weight_kg": 92.0,
        "waist_cm": 96.0,
        "conditions": [],
        "activity": "moderate",
        "available_days": 5,
        "current_cardio_minutes": 60,
        "environment": "gym",
        "training_level": 2,
        "timezone": "Asia/Shanghai",
    },
    "在家练的 B 层（66 岁女，高血压，膝盖疼）": {
        "birth_date": "1960-05-10",
        "sex": "female",
        "height_cm": 158.0,
        "weight_kg": 66.0,
        "waist_cm": 88.0,
        "conditions": ["hypertension"],
        "activity": "sedentary",
        "available_days": 3,
        "current_cardio_minutes": 120,
        "environment": "home",
        "restrictions": ["knee_pain"],
        "training_level": 1,
        "timezone": "Asia/Shanghai",
    },
    "孕期 C 层（30 岁女）": {
        "birth_date": "1996-01-01",
        "sex": "female",
        "height_cm": 162.0,
        "weight_kg": 60.0,
        "waist_cm": 78.0,
        "conditions": ["pregnant"],
        "activity": "sedentary",
        "timezone": "Asia/Shanghai",
    },
    "信息不全的新用户": {"sex": "female", "timezone": "Asia/Shanghai"},
}


def show(name: str, result: FirstPlan) -> None:
    print(f"\n## {name}\n")
    tier = result.tier.model_dump(mode="json")["tier"]
    print(f"- 分层：{tier}（{result.tier.status}）；提示：{'、'.join(result.notices) or '无'}")
    if result.questions:
        asks = "；".join(
            f"{q.item.removeprefix('profile.')}（{q.fallback}）" for q in result.questions
        )
        print(f"- 追问：{asks}")
    if result.constraints:
        rules = "；".join(f"{c.field} {c.op} {deep_thaw(c.value)}" for c in result.constraints)
        print(f"- 安全官约束：{rules}")
    rejected = [v for v in result.verdicts if v.result == "reject"]
    print(f"- 复核：{len(result.verdicts) - len(rejected)} 条通过，{len(rejected)} 条拒绝")
    if result.plan is None:
        print("- 计划：不给（只提供提示与基线测试）")
    else:
        print("- 第 1 版计划：")
        for card in result.cards:
            label = PLAN_FIELD_NAMES.get(card.field.value, card.field.value)
            value: Any = deep_thaw(card.to_value)
            if card.field.value == "exercise_selection":
                value = "、".join(f"{v['slot']}：{v['name']}" for v in value)
            elif card.field.value in ("session_slots", "minimum_task", "alternative_version"):
                value = (
                    f"{len(value) if isinstance(value, list) else len(value.get('hotel', []))} 项"
                )
            source = f"依据 {card.source.ref}（{card.source.review}）"
            print(f"  - {label}：{value}  ←  {source}；{card.facts[0]}")
    tests = default_baseline()
    names = "、".join(tests.get(t).name for t in result.baseline.test_ids)
    reason = f"（{result.baseline.skipped_reason}）" if result.baseline.skipped_reason else ""
    print(f"- 基线测试：{names}{reason}")


def main() -> None:
    rulebook, skills = load_rules(RULES_DIR), load_skills()
    print("# 冲刺 2 演示：建档 → 筛查 → 追问 → 约束 → 提案 → 复核 → 第一版计划 → 依据卡 → 基线测试")
    for name, profile in PERSONAS.items():
        result = first_plan(profile, as_of=AS_OF, tz=TZ, rulebook=rulebook, skills=skills)
        show(name, result)


if __name__ == "__main__":
    main()
