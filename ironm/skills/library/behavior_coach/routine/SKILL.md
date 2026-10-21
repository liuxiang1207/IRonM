---
id: routine
owner_agent: behavior_coach
version: 1.0.0
description: 固定的称重日、每周签到日与提醒时间
triggers:
  intents: [goal.create, plan.adjust]
applies_to:
  scene_packs: [fat_loss, strength, fitness, habits, maintenance]
  population_tiers: [a, a_prime, b, collaborative]
requires_data: [tier, constraints, current_plan]
produces: [weigh_in_days, checkin_day, reminder_schedule]
priority: 10
prescriptive: true
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_routine.py
---

# 作息：称重、签到与提醒

参数取自 R-HB-ROUTINE（搭建计划表 D44、D55）：

- **称重**：默认每周 2 次，周一、周四早上空腹；固定日子比每天称更容易坚持，也够判断趋势
- **签到**：每周日晚上，约 1 分钟（疲劳、睡眠、按计划吃了几天、困难与不适）
- **提醒**：只在称重日早上、签到日晚上、漏打卡或没签到的第二天早上提醒，每天最多 1 条
- 安全官给了称重次数上限（如有进食障碍史时每周最多 1 次）时，只保留排在前面的称重日
