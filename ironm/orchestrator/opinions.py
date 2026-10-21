"""问答的意见收集（spec [2026-10-21] §12）：选中的专业 Agent 全部并行作答，互不通信。

每个 Agent 只拿到问题、自己声明能读的画像字段与页面条目编号，看不到其他 Agent 的输出；
各自检索本领域知识、调用本领域工具、生成带依据的意见。单个 Agent 出错为缺失，不影响其他。
问答不加载 Skill。
"""

from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ironm.agents.answering import SPECS, AgentAnswer, answer_as
from ironm.contracts.roles import AgentId
from ironm.core.log import get_logger
from ironm.knowledge.kb import Embedder, Retriever
from ironm.knowledge.qa_cards import CardBook
from ironm.llm.gateway import Gateway
from ironm.rules.loader import RuleBook

__all__ = ["Collected", "collect_opinions"]

_log = get_logger("ironm.opinions")


@dataclass(frozen=True)
class Collected:
    answers: tuple[AgentAnswer, ...]  # 按给出的 Agent 顺序（主领域在前）

    @property
    def with_opinion(self) -> tuple[AgentAnswer, ...]:
        return tuple(a for a in self.answers if a.opinion is not None)

    @property
    def missing(self) -> tuple[AgentAnswer, ...]:
        return tuple(a for a in self.answers if a.opinion is None)


def collect_opinions(
    agents: Iterable[AgentId],
    *,
    text: str,
    profile: Mapping[str, Any],
    context_item: str | None,
    retriever: Retriever,
    cards: CardBook,
    rulebook: RuleBook,
    gateway: Gateway | None,
    request_id: str,
    as_of: datetime,
    tz: ZoneInfo,
    queries: Sequence[str] = (),
    embedder: Embedder | None = None,
) -> Collected:
    chosen = [a for a in dict.fromkeys(agents) if a in SPECS]
    if not chosen:
        return Collected(())

    def run(agent: AgentId) -> AgentAnswer:
        try:
            return answer_as(
                agent,
                text=text,
                profile=profile,
                context_item=context_item,
                retriever=retriever,
                cards=cards,
                rulebook=rulebook,
                gateway=gateway,
                request_id=request_id,
                as_of=as_of,
                tz=tz,
                queries=queries,
                embedder=embedder,
            )
        except Exception as e:  # 单个 Agent 出错为缺失，不影响其他
            _log.exception("opinions.agent_failed", e, request_id=request_id)
            return AgentAnswer(agent, "missing", None, (), (), f"agent_error:{type(e).__name__}")

    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        answers = tuple(pool.map(run, chosen))
    return Collected(answers)
