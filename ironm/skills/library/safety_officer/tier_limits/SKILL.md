---
id: tier_limits
owner_agent: safety_officer
version: 1.0.0
description: 分层限制：按人群分层给出用力程度上限、有氧分钟数每周增量与上限
triggers:
  intents: [goal.create, review.weekly, review.biweekly, training.too_tired, plan.adjust]
applies_to:
  scene_packs: [fat_loss, strength, fitness, habits, maintenance]
  population_tiers: [a, a_prime, b, collaborative]
requires_data: [tier]
produces: []
priority: 10
prescriptive: false
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_tier_limits.py
---

# 分层限制

参数取自规则 R-SF-TIER-LIMITS：

- A、A′：用力程度上限 RPE 8；每周有氧分钟数每周最多增加 30 分钟
- B、协作：用力程度上限 RPE 7；每周最多增加 20 分钟；每周有氧不超过 150 分钟

分层未知或不可给计划时“本次无意见”（处方类 Skill 本来就不会加载）。
红旗筛查由安全通道在路由前完成，这里不重复。
