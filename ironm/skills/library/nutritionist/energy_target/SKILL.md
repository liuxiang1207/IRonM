---
id: energy_target
owner_agent: nutritionist
version: 1.0.0
description: 能量目标：用热量计算给出每日热量与蛋白质目标，缺口按同时满足的所有分层取最小
triggers:
  intents: [goal.create, plan.adjust]
applies_to:
  scene_packs: [fat_loss]
  population_tiers: [a, a_prime, b, collaborative]
requires_data: [tier, tier_matched, current_plan, profile.birth_date, profile.sex, profile.height_cm, profile.weight_kg, profile.activity, profile.timezone, profile.high_protein]
produces: [daily_kcal_target, daily_protein_g_target]
priority: 10
prescriptive: true
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_energy_target.py
---

# 能量目标

全部由热量计算（§25、§26，规则 R-CAL-*）得出，Skill 不另设数字：

- 静息能量消耗 × 身体活动水平 − 缺口；缺口按同时满足的所有分层取最小；不低于热量底线与静息能量消耗
- 蛋白质取占比、每千克参考体重与老年推荐量中最大者
- 协作层（医生指导下用减重药）：目标是“每天至少要吃到的量”，不是上限
- 信息不全时“本次无意见”并写明缺什么（追问由评估师负责）

结果标为估算；两周后按体重趋势复核。
