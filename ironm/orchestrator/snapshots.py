"""问答链路的只读快照（spec [2026-10-21] 统一链路）。

和计划链路的快照是同一个契约，只是数据来源不同：问答时读用户的档案、分层与计划，
不生成任何新数据。没有计划的用户 current_plan 为空；精度默认 P0（没有近期记录）。
"""

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ironm.contracts.common import Domain, Precision
from ironm.contracts.screening import TierResult
from ironm.contracts.snapshot import Snapshot

__all__ = ["chat_snapshot"]


def chat_snapshot(
    profile: Mapping[str, Any],
    tier: TierResult,
    now: datetime,
    *,
    timezone: str = "Asia/Shanghai",
    user_id: str = "u1",
    plan: Mapping[str, Any] | None = None,
    precision: Mapping[Domain, Precision] | None = None,
) -> Snapshot:
    """由档案与分层构造快照；问答类 Skill 只能读自己声明的字段。"""
    data: dict[str, Any] = {
        "snapshot_id": f"chat-{now.isoformat()}",
        "user_id": user_id,
        "as_of": now.isoformat(),
        "timezone": timezone,
        "tier": tier.model_dump(mode="json")["tier"],
        "tier_matched": [t.value for t in tier.matched],
        "precision_by_domain": {
            d.value: (precision or {}).get(d, Precision.P0).value for d in Domain
        },
        "profile": dict(profile),
        "metrics": [],
        "current_plan": plan,
        "constraints": [],
        "prior_verdicts": [],
    }
    return Snapshot.model_validate_json(json.dumps(data, ensure_ascii=False))
