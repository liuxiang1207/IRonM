"""对话记忆的数据结构（spec [2026-10-21] §8）：消息、摘要与画像候选。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

__all__ = ["Candidate", "Role", "Status", "StoredMessage", "Summary"]

type Role = Literal["user", "assistant"]
type Status = Literal["pending", "confirmed", "rejected"]


@dataclass(frozen=True)
class StoredMessage:
    seq: int
    role: Role
    text: str
    intent: str | None


@dataclass(frozen=True)
class Summary:
    upto_seq: int
    text: str
    source: Literal["model", "rule"]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    user_id: str
    request_id: str
    field: str
    op: Literal["set", "add"]
    value: Any
    immediate: bool
    status: Status
    at: datetime
