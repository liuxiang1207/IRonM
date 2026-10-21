---
id: diet_mode
owner_agent: nutritionist
version: 1.0.0
description: 饮食模式：用户选过的优先，否则按饮食精度选，默认最轻
triggers:
  intents: [goal.create, plan.adjust]
applies_to:
  scene_packs: [fat_loss]
  population_tiers: [a, a_prime, b, collaborative]
requires_data: [precision, current_plan, profile.diet_mode_preference]
produces: [diet_mode]
priority: 20
prescriptive: true
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_diet_mode.py
---

# 饮食模式

三种模式（IronM 方案 v3.0 §6.3）：精确（热量 + 蛋白质）、简化（餐盘与手掌份量）、习惯（每次只改一个习惯），随时切换。

- 用户选过的模式优先
- 否则按饮食精度（R-NU-DIET-MODE）：P3 精确、P2 简化、P0 与 P1 习惯——记录越少，要求越轻
