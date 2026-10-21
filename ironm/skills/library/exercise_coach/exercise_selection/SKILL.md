---
id: exercise_selection
owner_agent: exercise_coach
version: 1.0.0
description: 选动作：把每个训练槽位落成具体动作，附要点与替代，并给出最低任务与酒店版本
triggers:
  intents: [goal.create, plan.adjust, training.too_tired]
applies_to:
  scene_packs: [fat_loss]
  population_tiers: [a, a_prime, b, collaborative]
requires_data: [current_plan, profile.environment, profile.restrictions, profile.training_level]
produces: [exercise_selection, minimum_task, alternative_version]
priority: 10
prescriptive: true
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_exercise_selection.py
---

# 选动作

从动作库（§27）按用户的训练环境与身体限制选择：

- 每个槽位取难度不超过用户水平、最接近其水平的候选，附技术要点、谨慎提示与最多 2 个替代动作（难度相近的优先；环境里可选动作少时至少 1 个）
- 最低任务：下蹲、推、髋铰链的默认安全动作各 2 组（R-EX-MINIMUM），没时间或状态差时至少做这些
- 酒店版本：只用酒店能找到的条件（徒手、椅子）重新选一遍

不知道训练环境时按酒店（最少器材）选；不知道水平时按 1（最容易）；没有当前计划的槽位时“本次无意见”。
