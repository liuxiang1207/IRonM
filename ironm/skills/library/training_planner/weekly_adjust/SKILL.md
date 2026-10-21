---
id: weekly_adjust
owner_agent: training_planner
version: 1.0.0
description: 每周训练复盘：按决策表调整有氧分钟数与每个动作的组数、次数
triggers:
  intents: [review.weekly]
applies_to:
  scene_packs: [fat_loss, strength, fitness, habits, maintenance]
  population_tiers: [a, a_prime, b, collaborative]
requires_data: [tier, constraints, current_plan, review.completion, review.prev_completion, review.avg_rpe, review.fatigue, review.fatigue_baseline, review.sleep_quality, review.partial_week, review.menstruation, review.recovery_weeks_before]
produces: [weekly_cardio_minutes, session_slots]
priority: 10
prescriptive: true
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_weekly_adjust.py
---

# 每周训练复盘

参数取自 R-RV-WEEKLY，按顺序取第一条命中的（spec [2026-10-21] §17）：

1. 签到说"只记录了一部分"：不调整；完成率数据不足：不调整
2. 疲劳比前两次签到的平均高 2 分及以上，或睡眠质量 ≤2：有氧减 20%，每个动作减 1 组（不少于 1 组）
3. 本周平均 RPE ≥9：每个动作减 1 组
4. 完成率连续两周低于 50%：有氧减 20%，不低于起点
5. 完成率 ≥90%，且平均 RPE ≤7（没记 RPE 时要求连续两周 ≥90%）：有氧加一档；力量先加组到 3 组，再把次数上限加到 15
6. 其余：不变

标了经期的一周不做第 2–4 条的下调。有氧加档不超过安全官的上限，也不超过 R-TR-START 的 cardio_target_min。
