"""安全回答模板与拼装（spec §24）：红旗四段式、四项拒答、分层提示。

文字来自 replies.yaml，由人维护；拼装只做取用与替换占位符，不含用户原话。
四段式的拼装在 App 离线版（app/src/safety/emergency.ts）中逐步对应实现。
"""

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from ironm.contracts.common import ContractModel
from ironm.safety.emergency import Assessment
from ironm.safety.matching import Term, load_file
from ironm.safety.refusals import KINDS, RefusalKind

__all__ = [
    "NOTICES",
    "REPLIES_PATH",
    "Replies",
    "ReplySection",
    "four_section",
    "load_replies",
    "notice_sections",
    "refusal_sections",
]

REPLIES_PATH = Path(__file__).with_name("replies.yaml")

type SectionKind = Literal["conclusion", "stop", "cannot_judge", "help", "refusal", "notice"]
type NoticeCode = Literal[
    "paused_until_clearance",
    "tier_b",
    "tier_collaborative",
    "tier_c",
    "fat_loss_not_applicable",
    "tier_d",
    "tier_unknown",
]
NOTICES: tuple[NoticeCode, ...] = NoticeCode.__value__.__args__
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def _placeholders(text: str, allowed: Iterable[str], where: str) -> str:
    bad = sorted(set(_PLACEHOLDER.findall(text)) - set(allowed))
    if bad or "{" in _PLACEHOLDER.sub("", text) or "}" in _PLACEHOLDER.sub("", text):
        raise ValueError(f"{where} 使用了未规定的占位符或括号：{bad or text}")
    return text


class ReplySection(ContractModel):
    kind: SectionKind
    title: Term
    text: Term


class LevelText(ContractModel):
    conclusion: Term
    stop: Term
    cannot_judge: Term
    help: Term


class CategoryText(ContractModel):
    stop: Term
    department: Term
    bring: tuple[Term, ...] = Field(min_length=1)
    conclusion: Term | None = None  # 替换级别的结论句（如心理危机用关心的语气）
    help_extra: Term | None = None  # 放在专业帮助的开头（如热线电话）


class Titles(ContractModel):
    conclusion: Term
    stop: Term
    cannot_judge: Term
    help: Term
    refusal: Term
    notice: Term


class Replies(ContractModel):
    version: Term
    review: Literal["pending_review", "approved"]
    titles: Titles
    levels: dict[Literal["emergency", "red_flag"], LevelText]
    categories: dict[Term, CategoryText] = Field(min_length=1)
    refusals: dict[RefusalKind, Term]
    notices: dict[NoticeCode, Term]
    missing_labels: dict[Term, Term]

    @model_validator(mode="after")
    def _check(self) -> Self:
        required: list[tuple[str, set[str]]] = [
            ("levels", {"emergency", "red_flag"} - set(self.levels)),
            ("refusals", {str(k) for k in KINDS} - set(self.refusals)),
            ("notices", {str(n) for n in NOTICES} - set(self.notices)),
        ]
        for name, missing in required:
            if missing:
                raise ValueError(f"{name} 缺少：{sorted(missing)}")
        for level, t in self.levels.items():
            for field in ("conclusion", "stop", "cannot_judge"):
                _placeholders(getattr(t, field), (), f"levels.{level}.{field}")
            _placeholders(t.help, ("{departments}", "{bring}"), f"levels.{level}.help")
        for name, cat in self.categories.items():
            if any("、" in b for b in cat.bring):
                raise ValueError(f"categories.{name}.bring 的每一项不能含顿号（拼接时用顿号分隔）")
        texts = [
            *(self.titles.model_dump().values()),
            *(c.stop for c in self.categories.values()),
            *(c.department for c in self.categories.values()),
            *(c.conclusion for c in self.categories.values() if c.conclusion),
            *(c.help_extra for c in self.categories.values() if c.help_extra),
            *(b for c in self.categories.values() for b in c.bring),
            *self.refusals.values(),
            *self.missing_labels.values(),
        ]
        for text in texts:
            _placeholders(text, (), "模板文字")
        for code, text in self.notices.items():
            _placeholders(text, ("{missing}",) if code == "tier_unknown" else (), f"notices.{code}")
        return self


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def four_section(replies: Replies, assessment: Assessment) -> tuple[ReplySection, ...]:
    """红旗四段式：安全结论 → 现在需要停止的 → 我不能判断的 → 建议的专业帮助。"""
    if assessment.level == "none":
        return ()
    level = replies.levels[assessment.level]
    cats = [replies.categories[c] for c in assessment.categories]
    stop = "".join(_unique([level.stop, *(c.stop for c in cats)]))
    departments = "或".join(_unique(c.department for c in cats))
    bring = "、".join(_unique(b for c in cats for b in c.bring))
    help_text = level.help.replace("{departments}", departments).replace("{bring}", bring)
    help_text = "".join(_unique(c.help_extra for c in cats if c.help_extra)) + help_text
    # 第一个带自己结论句的类别优先（按命中条目的顺序）
    conclusion = next((c.conclusion for c in cats if c.conclusion), level.conclusion)
    t = replies.titles
    return (
        ReplySection(kind="conclusion", title=t.conclusion, text=conclusion),
        ReplySection(kind="stop", title=t.stop, text=stop),
        ReplySection(kind="cannot_judge", title=t.cannot_judge, text=level.cannot_judge),
        ReplySection(kind="help", title=t.help, text=help_text),
    )


def refusal_sections(replies: Replies, kinds: Sequence[RefusalKind]) -> tuple[ReplySection, ...]:
    return tuple(
        ReplySection(kind="refusal", title=replies.titles.refusal, text=replies.refusals[k])
        for k in kinds
    )


def notice_sections(
    replies: Replies, codes: Sequence[NoticeCode], missing: Sequence[str] = ()
) -> tuple[ReplySection, ...]:
    labels = "、".join(replies.missing_labels.get(m, m) for m in missing)
    return tuple(
        ReplySection(
            kind="notice",
            title=replies.titles.notice,
            text=replies.notices[c].replace("{missing}", labels),
        )
        for c in codes
    )


def load_replies(path: Path) -> Replies:
    return load_file(path, Replies, "安全回答模板")
