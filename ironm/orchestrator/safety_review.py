"""急症判定的第二层：模型复核（spec [2026-10-21] §12）。

词表判为“无”或“红旗”时，由模型再判一次，只用来提高级别，不能降低。
- 判为急症或红旗时必须给出类别，类别必须是回答模板里已有的；否则不采纳这次复核。
- 模型不可用、输出不合规时，只用词表的结果。
- 复核提高级别时，条目编号加上 EM-MODEL-REVIEW / RF-MODEL-REVIEW，便于追查。
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ironm.contracts.model_call import ModelStage
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model
from ironm.safety.emergency import Assessment

__all__ = ["REVIEW_RULES", "review_safety"]

PROMPT_PATH = Path(__file__).parent / "prompts" / "safety_review.md"
PROMPT_TEMPLATE = "safety_review@1.0.0"
REVIEW_RULES = {"emergency": "EM-MODEL-REVIEW", "red_flag": "RF-MODEL-REVIEW"}
_RANK = {"none": 0, "red_flag": 1, "emergency": 2}


class _Review(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["emergency", "red_flag", "none"]
    categories: list[str] = []


def review_safety(
    text: str,
    assessment: Assessment,
    *,
    gateway: Gateway | None,
    categories: frozenset[str],
    request_id: str,
) -> Assessment:
    if gateway is None or assessment.level == "emergency":
        return assessment  # 没有模型，或已是最高级别
    payload = {"message": text, "categories": sorted(categories)}
    result = gateway.complete(
        ModelStage.SAFETY_REVIEW,
        request_id=request_id,
        template=PROMPT_TEMPLATE,
        messages=[
            Message("system", PROMPT_PATH.read_text(encoding="utf-8")),
            Message("user", json.dumps(payload, ensure_ascii=False)),
        ],
        validate=json_model(_Review),
    )
    review = result.value
    if review is None or _RANK[review.level] <= _RANK[assessment.level]:
        return assessment
    if not review.categories or not set(review.categories) <= categories:
        return assessment  # 类别缺失或不在模板里：不采纳
    level: Literal["emergency", "red_flag"] = review.level  # type: ignore[assignment]
    return Assessment(
        level=level,
        rule_ids=(*assessment.rule_ids, REVIEW_RULES[level]),
        categories=tuple(dict.fromkeys([*assessment.categories, *review.categories])),
    )
