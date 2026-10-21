---
id: weekly_structure
owner_agent: training_planner
version: 1.0.0
description: 第一版每周训练结构：力量训练次数、全身槽位、用力程度与每周有氧分钟数
triggers:
  intents: [goal.create, plan.adjust]
applies_to:
  scene_packs: [fat_loss]
  population_tiers: [a, a_prime, b, collaborative]
requires_data: [tier, constraints, current_plan, profile.available_days, profile.current_cardio_minutes]
produces: [training_days_per_week, session_slots, intensity_target, weekly_cardio_minutes]
priority: 10
prescriptive: true
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_weekly_structure.py
---

# 每周结构

参数取自 R-TR-START；安全官的约束（R-SF-TIER-LIMITS）优先，取更严者。

- **力量训练**：每周 2 次全身；可练天数达到 5 天时 3 次（WHO：成人每周至少 2 天肌肉力量活动）
- **槽位**：下蹲、髋铰链、推、拉、核心各 1 个，每个 2 组 8–12 次，RPE 6–7；具体动作由动作教练按器材与身体限制选择
- **有氧**：从“当前每周运动分钟 + 30”开始，至少 90 分钟（B 与协作层至少 60）；不超过安全官给的每周上限
- **调整三件套**：每项写明预期效果与回退条件，14 天后复核

不知道可练天数时按每周 2 次；不知道当前运动量时按 0 计，从下限开始。
