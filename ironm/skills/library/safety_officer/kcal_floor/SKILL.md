---
id: kcal_floor
owner_agent: safety_officer
version: 1.0.0
description: 热量底线：给出每日热量目标的下限约束
triggers:
  intents: [goal.create, review.biweekly, nutrition.plateau, plan.adjust]
applies_to:
  scene_packs: [fat_loss, strength, fitness, habits, maintenance]
  population_tiers: [a, a_prime, b, collaborative]
requires_data: [profile.sex]
produces: []
priority: 20
prescriptive: false
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_kcal_floor.py
---

# 热量底线

每日热量目标不低于 R-CAL-FLOOR：女 1200、男 1500 千卡。性别未知时取两者中较高的（更保守）。
热量计算本身还保证不低于静息能量消耗（§26），这里只给出按性别的底线，供复核提案时比对。
