---
id: intake_questions
owner_agent: assessor
version: 1.0.0
description: 建档追问：找出分层与热量计算缺少的信息，按对安全的影响排序，每次最多问 3 项
triggers:
  intents: [goal.create, profile.update, review.weekly]
applies_to:
  scene_packs: [fat_loss, strength, fitness, habits, maintenance]
  population_tiers: [a, a_prime, b, collaborative, c, d, not_applicable, paused]
  unknown_tier: true
requires_data: [profile.conditions, profile.birth_date, profile.height_cm, profile.weight_kg, profile.sex, profile.activity, profile.waist_cm]
produces: []
priority: 10
prescriptive: false
review: approved
reviewed_by: Claude（项目负责人委托，2026-10-21）
tests: tests/skills/test_intake_questions.py
---

# 建档追问

只问会改变**安全判断或计划结构**的信息，每次最多 3 项（spec §29），按下表顺序：

| 顺序 | 信息 | 为什么需要 | 没回答时 |
|---|---|---|---|
| 1 | 健康状况问卷 | 决定是否在 C 层等（R-SF-TIER） | 不给计划，直到回答 |
| 2 | 出生日期 | 未成年人暂不开放；65 岁以上为 B 层 | 不给计划，直到回答 |
| 3 | 身高 | 计算 BMI | 不给计划，直到回答 |
| 4 | 体重 | 计算 BMI 与热量 | 不给计划，直到回答 |
| 5 | 性别 | 热量公式与底线 | 不给热量目标，直到回答 |
| 6 | 活动水平 | 每日总消耗 | 按久坐估算（偏保守） |
| 7 | 腰围 | 区分 A 与 A′ | 按 A′ 保守假设（缺口更小） |

都齐了则“本次无意见”（信息够用）。追问不影响安全：急症拦截在路由之前独立运行。
