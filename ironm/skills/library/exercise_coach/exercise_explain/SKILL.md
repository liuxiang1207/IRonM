---
id: exercise_explain
owner_agent: exercise_coach
version: 1.0.0
description: 动作讲解：回答某个动作怎么做、常见错误、怎么自检、做不了时换成什么
triggers:
  intents: [exercise_technique]
applies_to:
  scene_packs: [fat_loss, strength, fitness, habits, maintenance]
  population_tiers: [a, a_prime, b, collaborative, c]
  unknown_tier: true
requires_data: [tier, profile.environment, profile.restrictions, profile.training_level]
produces: []
priority: 10
prescriptive: false
mode: answer
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_exercise_explain.py
---

# 动作讲解

回答“这个动作怎么做 / 做不对怎么办 / 做不了换什么”。内容全部来自动作库（§27），不另编：

- **结论**：一句话说清这个动作做什么、适合谁
- **依据**：动作库中该条目的编号与名称；用户的环境、身体限制（若影响选择）
- **怎么做**：动作库的三条技术要点
- **注意**：动作库的常见错误；用户有相关限制时，附动作库审定的提醒原文
- **不确定**：动作库里没有这个动作、或无法确定用户说的是哪一个时，如实说明并请用户描述

选择规则：
- 用户点了具体动作（带条目编号进入）时，直接讲解那一个，不再猜
- 用户只说名称时，按名称在动作库中匹配；匹配到多个时，给出最接近的一个并说明还有哪些相近的
- 用户说“我做不了”时，给降阶动作；说“太简单”时，给进阶动作

不做：
- 不判断伤病，不建议就医时机之外的诊断（涉及疼痛时按安全规则附提示）
- 不给具体重量与组次（那是计划链路的事），只说明动作本身
