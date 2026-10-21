"""画像候选（spec [2026-10-21] §8）：从对话中发现画像变化，用户确认后才写入画像。

- 只认用户说自己的情况（说的是旁人时不记）；说法前紧挨着否定词时不算。
- 新增健康状况立即生效（只会让分层更严），同时请用户确认；用户否认时撤回。
- 其余变化（训练环境、可练天数）先作为候选，确认后才写入；没有确认的候选不进入画像。
提取只查词表，不调用模型，在回答前同步完成（耗时可以忽略）。
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.contracts.memory import Candidate
from ironm.contracts.screening import Condition
from ironm.core.ids import IdGenerator
from ironm.core.log import get_logger
from ironm.core.yamlfile import load_yaml
from ironm.memory.repo import MemoryRepo
from ironm.safety.matching import check_term, clauses
from ironm.safety.subject import subject_of

__all__ = [
    "PROFILE_TERMS_PATH",
    "Change",
    "ProfileService",
    "ProfileTerms",
    "extract",
    "load_terms",
]

PROFILE_TERMS_PATH = Path(__file__).with_name("profile_terms.yaml")

_log = get_logger("ironm.memory")

type Environment = Literal["home", "gym", "hotel"]
_DIGITS = "一二三四五六七"
_DAYS_RE = re.compile(rf"每周(?:只能|能|可以|最多|大概)?练([1-7{_DIGITS}])天")


class ProfileTerms(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    conditions: dict[Condition, tuple[str, ...]]
    stricter_than: dict[Condition, tuple[Condition, ...]] = {}
    environment: dict[Environment, tuple[str, ...]]
    negation: tuple[str, ...] = Field(min_length=1)

    def model_post_init(self, _: Any) -> None:
        words = [*self.negation]
        words += [w for ws in self.conditions.values() for w in ws]
        words += [w for ws in self.environment.values() for w in ws]
        for w in words:
            check_term(w)


def load_terms(path: Path = PROFILE_TERMS_PATH) -> ProfileTerms:
    return ProfileTerms.model_validate(load_yaml(path))


@dataclass(frozen=True)
class Change:
    field: Literal["conditions", "environment", "available_days"]
    op: Literal["add", "set"]
    value: Any
    immediate: bool  # 健康状况：立即生效并请确认


def _said(text: str, words: tuple[str, ...], negation: tuple[str, ...]) -> bool:
    for w in words:
        start = text.find(w)
        while start != -1:
            if not any(text[:start].endswith(n) for n in negation):
                return True
            start = text.find(w, start + 1)
    return False


def extract(text: str, terms: ProfileTerms) -> list[Change]:
    """一句话里的画像变化。说的是旁人时没有变化。"""
    if subject_of(text) != "self":
        return []
    joined = "".join(clauses(text))
    neg = terms.negation
    found = [c for c, words in terms.conditions.items() if _said(joined, words, neg)]
    looser = {lo for c in found for lo in terms.stricter_than.get(c, ())}
    changes = [Change("conditions", "add", c.value, True) for c in found if c not in looser]
    for env, words in terms.environment.items():
        if _said(joined, words, neg):
            changes.append(Change("environment", "set", env, False))
            break
    m = _DAYS_RE.search(joined)
    if m:
        raw = m.group(1)
        days = int(raw) if raw.isdigit() else _DIGITS.index(raw) + 1
        changes.append(Change("available_days", "set", days, False))
    return changes


def _apply(profile: dict[str, Any], c: Change | Candidate) -> dict[str, Any]:
    data = dict(profile)
    if c.op == "add":
        current = data.get(c.field)
        items = list(current) if isinstance(current, list) else []
        if c.value not in items:
            items.append(c.value)
        data[c.field] = items
    else:
        data[c.field] = c.value
    return data


def _revert(profile: dict[str, Any], c: Candidate) -> dict[str, Any]:
    data = dict(profile)
    current = data.get(c.field)
    if c.op == "add" and isinstance(current, list):
        data[c.field] = [v for v in current if v != c.value]
    return data


class ProfileService:
    def __init__(
        self,
        store: MemoryRepo,
        *,
        terms: ProfileTerms,
        ids: IdGenerator,
        now: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._terms = terms
        self._ids = ids
        self._now = now

    def profile(self, user_id: str) -> dict[str, Any]:
        return self._store.profile(user_id)[1]

    def note(self, user_id: str, request_id: str, text: str) -> list[Candidate]:
        """记下这句话里的画像变化；健康状况立即写入画像。返回新的候选。"""
        out = []
        for change in extract(text, self._terms):
            version, profile = self._store.profile(user_id)
            current = profile.get(change.field)
            already = (
                change.value in current
                if change.op == "add" and isinstance(current, list)
                else current == change.value
            )
            if already:
                continue
            candidate = Candidate(
                candidate_id=self._ids.new_id(),
                user_id=user_id,
                request_id=request_id,
                field=change.field,
                op=change.op,
                value=change.value,
                immediate=change.immediate,
                status="pending",
                at=self._now(),
            )
            self._store.add_candidate(candidate)
            if change.immediate:
                updated = _apply(profile, change)
                reason = f"candidate:{candidate.candidate_id}"
                self._store.save_profile(user_id, version, updated, reason, self._now())
            out.append(candidate)
        if out:
            _log.info("memory.candidates", request_id=request_id, count=len(out))
        return out

    def decide(self, user_id: str, candidate_id: str, accept: bool) -> Candidate | None:
        """确认或拒绝一个待确认的候选；不是本人的或已处理过的返回 None。"""
        c = self._store.candidate(candidate_id)
        if c is None or c.user_id != user_id or c.status != "pending":
            return None
        version, profile = self._store.profile(user_id)
        reason = f"decision:{candidate_id}"
        if accept and not c.immediate:
            self._store.save_profile(user_id, version, _apply(profile, c), reason, self._now())
        if not accept and c.immediate:
            self._store.save_profile(user_id, version, _revert(profile, c), reason, self._now())
        self._store.decide(candidate_id, "confirmed" if accept else "rejected", self._now())
        return self._store.candidate(candidate_id)
