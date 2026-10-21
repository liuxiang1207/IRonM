"""召回候选的大模型重排（spec [2026-10-21] §13）：按“这一段能不能帮这个角色回答这个问题”打分。

模型不可用、漏打或多打了段落、分数越界时返回 None，由检索改用覆盖比例打分。
"""

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ironm.contracts.model_call import ModelStage
from ironm.knowledge.kb import Passage
from ironm.llm.gateway import Gateway, Message
from ironm.llm.validate import json_model

__all__ = ["ModelReranker"]

PROMPT_PATH = Path(__file__).parent / "prompts" / "rerank.md"
PROMPT_TEMPLATE = "rerank@1.0.0"
EXCERPT_CHARS = 400  # 每个候选只给模型看开头这么多字，控制长度


class _Scores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: dict[str, float]


class ModelReranker:
    def __init__(self, gateway: Gateway, *, request_id: str, question: str, role: str) -> None:
        self._gateway, self._request_id = gateway, request_id
        self._question, self._role = question, role

    def score(self, query: str, passages: Sequence[Passage]) -> list[float] | None:
        payload = {
            "question": self._question,
            "query": query,
            "role": self._role,
            "passages": [
                {"id": p.passage_id, "title": p.title, "text": p.text[:EXCERPT_CHARS]}
                for p in passages
            ],
        }
        result = self._gateway.complete(
            ModelStage.RERANK,
            request_id=self._request_id,
            template=PROMPT_TEMPLATE,
            messages=[
                Message("system", PROMPT_PATH.read_text(encoding="utf-8")),
                Message("user", json.dumps(payload, ensure_ascii=False)),
            ],
            validate=json_model(_Scores),
        )
        got = result.value
        if got is None or set(got.scores) != {p.passage_id for p in passages}:
            return None
        if not all(0 <= v <= 1 for v in got.scores.values()):
            return None
        return [got.scores[p.passage_id] for p in passages]
