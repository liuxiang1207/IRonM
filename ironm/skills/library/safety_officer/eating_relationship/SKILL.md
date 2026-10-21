---
id: eating_relationship
owner_agent: safety_officer
version: 1.0.1
description: 饮食关系警示：有进食障碍史或信号时，不设热量缺口、每周最多称重 1 次
triggers:
  intents: [goal.create, review.weekly, review.biweekly, nutrition.plateau, plan.adjust]
applies_to:
  scene_packs: [fat_loss]
  population_tiers: [a, a_prime, b, collaborative, c, d, not_applicable, paused]
  unknown_tier: true
requires_data: [profile.conditions, profile.derived_conditions]
produces: []
priority: 5
prescriptive: false
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_eating_relationship.py
---

# 饮食关系警示

依据：IronM 方案 v3.0 §14“饮食关系警示”——食物恐惧、催吐、过度补偿时停止热量缺口与频繁称重建议，转 C 层并提示专业支持。

有进食障碍史（eating_disorder_history）或进食障碍信号（eating_disorder_signal）时：

- 每日热量目标：不设缺口
- 称重日：每周最多 1 次（频繁称重可能加重对体重的焦虑）

适用于所有分层（含分层未知）：C 层不给计划，但记录与提醒类 Skill 仍会运行，这些约束限制它们的提醒内容。
聊天中识别出的进食障碍信号由编排写入快照画像的健康状况（spec §32），这里一并读取。
