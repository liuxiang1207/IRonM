"""安全决定（spec §24）：三档输出、是否给计划、提示、拒答与回答段落。

只看规则：急症词表、拒答词表、分层结果与回答模板；不调用模型。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from ironm.contracts.common import ContractModel
from ironm.contracts.screening import Tier, TierResult
from ironm.contracts.unknown import is_known
from ironm.safety.emergency import TERMS_PATH, Assessment, TermBook, load_terms
from ironm.safety.matching import TermLoadError
from ironm.safety.refusals import REFUSALS_PATH, RefusalBook, RefusalCheck, load_refusals
from ironm.safety.replies import (
    REPLIES_PATH,
    NoticeCode,
    Replies,
    ReplySection,
    four_section,
    load_replies,
    notice_sections,
    refusal_sections,
)

__all__ = ["SafetyDecision", "SafetyKit", "decide", "default_kit", "load_kit", "plan_allowed"]

type Mode = Literal["full", "conservative", "pause_refer"]

# 分层 → 档位、是否给计划、提示（暂停与未知单独处理）
_BY_TIER: dict[Tier, tuple[Mode, bool, tuple[NoticeCode, ...]]] = {
    Tier.A: ("full", True, ()),
    Tier.A_PRIME: ("full", True, ()),
    Tier.B: ("conservative", True, ("tier_b",)),
    Tier.COLLABORATIVE: ("conservative", True, ("tier_collaborative",)),
    Tier.NOT_APPLICABLE: ("conservative", False, ("fat_loss_not_applicable",)),
    Tier.C: ("conservative", False, ("tier_c",)),
    Tier.D: ("conservative", False, ("tier_d",)),
    Tier.PAUSED: ("pause_refer", False, ("paused_until_clearance",)),
}


def plan_allowed(tier: Tier) -> bool:
    """这一分层是否可以给训练或饮食计划（热量目标等计算以此为准）。"""
    return _BY_TIER[tier][1]


@dataclass(frozen=True)
class SafetyKit:
    """安全模块用到的三份文件；加载时互相校验。"""

    terms: TermBook
    refusals: RefusalBook
    replies: Replies


def load_kit(
    terms_path: Path = TERMS_PATH,
    refusals_path: Path = REFUSALS_PATH,
    replies_path: Path = REPLIES_PATH,
) -> SafetyKit:
    """服务启动时调用：任何一份有误、或急症类别没有对应模板，都抛 TermLoadError。"""
    kit = SafetyKit(
        load_terms(terms_path), load_refusals(refusals_path), load_replies(replies_path)
    )
    used = {r.category for r in kit.terms.rules}
    templated = set(kit.replies.categories)
    if used != templated:
        raise TermLoadError(
            f"急症类别与回答模板不一致：缺模板 {sorted(used - templated)}，"
            f"多余模板 {sorted(templated - used)}"
        )
    return kit


@cache
def default_kit() -> SafetyKit:
    return load_kit()


class SafetyDecision(ContractModel):
    """一句话的安全决定。sections 为由模板生成的回答段落，按顺序展示；不含用户原话。"""

    mode: Mode
    plan_allowed: bool
    notices: tuple[NoticeCode, ...]
    assessment: Assessment
    refusals: RefusalCheck
    sections: tuple[ReplySection, ...]


def decide(text: str, tier: TierResult, kit: SafetyKit | None = None) -> SafetyDecision:
    kit = kit or default_kit()
    assessment = kit.terms.assess(text)
    refusals = kit.refusals.detect(text)
    missing: Sequence[str] = ()
    mode: Mode
    plan: bool
    notices: tuple[NoticeCode, ...]
    if assessment.level != "none":
        mode, plan, notices = "pause_refer", False, ()
    elif not is_known(tier.tier):
        mode, plan, notices, missing = "conservative", False, ("tier_unknown",), tier.missing
    else:
        mode, plan, notices = _BY_TIER[tier.tier]
    sections = (
        four_section(kit.replies, assessment)
        + refusal_sections(kit.replies, refusals.kinds)
        + notice_sections(kit.replies, notices, missing)
    )
    return SafetyDecision(
        mode=mode,
        plan_allowed=plan,
        notices=notices,
        assessment=assessment,
        refusals=refusals,
        sections=sections,
    )
