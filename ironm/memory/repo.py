"""记忆层用到的存储接口（spec [2026-10-21] §8）。由存储层实现，记忆层只依赖这个接口。"""

from datetime import datetime
from typing import Any, Literal, Protocol

from ironm.contracts.memory import Candidate, Role, Status, StoredMessage, Summary


class MemoryRepo(Protocol):
    def add_message(
        self, user_id: str, request_id: str, role: Role, text: str, intent: str | None, at: datetime
    ) -> int: ...

    def messages_after(self, user_id: str, after_seq: int) -> list[StoredMessage]: ...

    def latest_summary(self, user_id: str) -> Summary | None: ...

    def add_summary(
        self, user_id: str, upto_seq: int, text: str, source: Literal["model", "rule"], at: datetime
    ) -> None: ...

    def profile(self, user_id: str) -> tuple[int, dict[str, Any]]: ...

    def save_profile(
        self, user_id: str, expected_version: int, data: dict[str, Any], reason: str, at: datetime
    ) -> int: ...

    def add_candidate(self, c: Candidate) -> None: ...

    def candidates(self, user_id: str, status: Status | None = None) -> list[Candidate]: ...

    def candidate(self, candidate_id: str) -> Candidate | None: ...

    def decide(self, candidate_id: str, status: Status, at: datetime) -> None: ...
