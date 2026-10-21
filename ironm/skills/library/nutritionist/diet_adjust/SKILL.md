---
id: diet_adjust
owner_agent: nutritionist
version: 1.0.0
description: 两周饮食复盘（只对减脂）：按决策表调整每日热量目标，每次 150 千卡，不破底线
triggers:
  intents: [review.biweekly]
applies_to:
  scene_packs: [fat_loss]
  population_tiers: [a, a_prime]
requires_data: [tier, constraints, current_plan, review.red_flag, review.meal_days_min, review.weigh_ins_min, review.loss_kg_per_week, review.ci_half_width, review.completion, review.adherence_days, review.prev_slow, review.days_since_start, review.partial_week, review.menstruation]
produces: [daily_kcal_target]
priority: 10
prescriptive: true
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_diet_adjust.py
---

# 两周饮食复盘

参数取自 R-RV-DIET，按顺序取第一条命中的（spec [2026-10-21] §17）：

1. 期间有红旗或新诊断：饮食不变
2. 数据不足（某一周饮食记录少于 4 天或没称重，或体重趋势的可信范围半宽大于 0.25 千克 / 周）：饮食不变，推迟到下次
3. 训练完成率低于 50%，或按计划吃的天数少于每周 3 天：饮食不变，先降计划难度
4. 训练完成率 ≥70%、按计划吃 ≥4 天 / 周，减重不到目标下限的一半，且上个周期也是这样：每日热量减 150 千卡，不低于热量底线
5. 满 4 周后减重快于每周 1.0 千克：每日热量加 150 千卡
6. 其余：不变

只对 A、A′ 层：B 与协作层不走常规饮食复盘。签到说"只记录了一部分"或标了经期时，不做第 4 条的下调。
