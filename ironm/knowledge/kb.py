"""知识库与检索（spec [2026-10-21] §12）：召回（关键词 + 向量）→ 重排。

- 知识库是一组段落，每段标上领域（负责该领域的专业角色）、出处、证据级别与审定状态。
  初始内容来自问答卡与动作库；资料目录（不进仓库）里的段落随配置加载。
- 召回：关键词按中文两字片段做 BM25；配置了向量模型时再做向量召回，两路按倒数排名融合（RRF）。
- 重排：默认按“问题的片段被段落覆盖的比例”打分，可换成重排模型；低于门槛的段落不算依据。
检索只查表与计算，不调用大模型。
"""

import json
import math
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, ValidationError

from ironm.contracts.common import ContractModel
from ironm.contracts.roles import AgentId
from ironm.core.yamlfile import load_yaml
from ironm.knowledge.exercises import Library
from ironm.knowledge.qa_cards import CardBook
from ironm.rules.loader import RuleBook
from ironm.safety.matching import clauses

__all__ = [
    "RETRIEVAL_RULE",
    "Embedder",
    "KnowledgeBase",
    "KnowledgeLoadError",
    "Passage",
    "Reranker",
    "Retrieval",
    "RetrievalParams",
    "Retrieved",
    "Retriever",
    "load_passages",
    "retriever_from_rules",
]

RETRIEVAL_RULE = "R-QA-RETRIEVAL"
RESERVED_PREFIXES = ("card:", "exercise:")

type Level = Literal["guideline", "review", "study", "expert", "card", "library"]
type Text = Annotated[str, Field(min_length=1)]

_K1, _B = 1.5, 0.75  # BM25 参数
_RRF_K = 60  # 倒数排名融合的常数


class KnowledgeLoadError(ValueError):
    pass


class Passage(ContractModel):
    """一段知识。出处给用户看；证据级别决定表述的语气（指南说“建议”，单项研究说“有研究显示”）。"""

    passage_id: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:\-]{0,120}$")]
    domain: AgentId
    title: Text
    text: Text
    source: Text
    level: Level
    review: Literal["pending_review", "approved"]


def _grams(text: str) -> list[str]:
    """统一文字后取两字片段（中文）与整词（英文、数字）。"""
    joined = "".join(clauses(text))
    grams = [joined[i : i + 2] for i in range(len(joined) - 1)]
    return grams or ([joined] if joined else [])


class Embedder(Protocol):
    """向量模型（如本地 bge-m3）。没有配置时只做关键词召回。"""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    """重排：给每段打 0–1 的相关分；不可用时返回 None，由检索改用覆盖比例打分。"""

    def score(self, query: str, passages: Sequence[Passage]) -> list[float] | None: ...


# 问句里的虚词与口头语：不表示问的是什么，算覆盖比例前去掉（只用于没有大模型重排时的打分）
_QUESTION_TERMS = (
    "怎么样",
    "怎么",
    "什么",
    "多少",
    "哪个",
    "哪些",
    "哪种",
    "为什么",
    "可以",
    "能不能",
    "是不是",
    "有没有",
    "应该",
    "需要",
    "比较",
    "合适",
    "一下",
    "还是",
    "吗",
    "呢",
    "吧",
    "啊",
    "呀",
    "要",
    "能",
    "会",
    "该",
    "好",
    "的",
    "了",
    "我",
    "你",
    "是",
    "有",
    "在",
    "就",
    "都",
    "也",
    "很",
    "太",
    "想",
    "做",
    "这个",
    "那个",
    "这",
    "那",
)
_QUESTION_WORDS = re.compile("|".join(sorted(_QUESTION_TERMS, key=len, reverse=True)))


class CoverageReranker:
    """默认重排：问题的两字片段被段落覆盖的比例，按片段的稀有程度（IDF）加权。

    “怎么、多少”这类到处都有的问句片段权重低，“肌酸、蛋白”这类专业词权重高。
    """

    def __init__(self, idf: dict[str, float] | None = None, unseen: float = 0.0) -> None:
        self._idf = idf or {}
        self._unseen = unseen

    def score(self, query: str, passages: Sequence[Passage]) -> list[float]:
        # 知识库里从没出现过的片段（如“跑鞋”）最稀有：按最高权重计入，没被覆盖就拉低分数，
        # 避免只凭“什么”这类常见片段就判为全覆盖
        q = set(_grams(_QUESTION_WORDS.sub("，", query)))
        weight = {g: self._idf.get(g, self._unseen) for g in q}
        total = sum(weight.values())
        if total == 0:
            return [0.0 for _ in passages]
        out = []
        for p in passages:
            body = set(_grams(p.title + p.text))
            out.append(sum(w for g, w in weight.items() if g in body) / total)
        return out


@dataclass(frozen=True)
class Retrieved:
    passage: Passage
    score: float  # 重排分


class KnowledgeBase:
    def __init__(self, passages: Iterable[Passage]) -> None:
        self.passages = tuple(passages)
        ids = [p.passage_id for p in self.passages]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise KnowledgeLoadError(f"知识段落编号重复：{dup}")
        self._docs = [Counter(_grams(p.title + p.text)) for p in self.passages]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(self._docs)
        self._idf = {g: math.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}
        self.unseen_idf = math.log(1 + (n + 0.5) / 0.5)  # 没出现过的片段的权重

    @property
    def idf(self) -> dict[str, float]:
        return dict(self._idf)

    @classmethod
    def from_sources(
        cls, cards: CardBook, library: Library, extra: Iterable[Passage] = ()
    ) -> "KnowledgeBase":
        """初始内容：问答卡（各领域与安全类）+ 动作库（动作教练）+ 资料目录里的段落。"""
        passages = [
            Passage(
                passage_id=f"card:{c.id}",
                domain=c.agent,
                title=c.id,
                text="".join(v for _, v in c.parts()),
                source="；".join(c.sources),
                level="card",
                review="approved",
            )
            for c in cards.cards
        ]
        for e in library.exercises:
            body = (
                f"{e.name}。要点：{'；'.join(e.key_points)}。常见错误：{'；'.join(e.common_errors)}。"
                f"自检：{'；'.join(e.self_check)}。"
            )
            passages.append(
                Passage(
                    passage_id=f"exercise:{e.id}",
                    domain=AgentId.EXERCISE_COACH,
                    title=e.name,
                    text=body,
                    source="IronM 动作库",
                    level="library",
                    review="approved",
                )
            )
        return cls([*passages, *extra])

    def bm25(self, query: str, allowed: frozenset[int]) -> list[tuple[int, float]]:
        q = _grams(query)
        scores = []
        for i in allowed:
            doc, length = self._docs[i], self._lengths[i]
            s = 0.0
            for g in q:
                tf = doc.get(g, 0)
                if tf:
                    norm = tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * length / (self._avg or 1)))
                    s += self._idf.get(g, 0.0) * norm
            if s > 0:
                scores.append((i, s))
        return sorted(scores, key=lambda x: -x[1])


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


@dataclass(frozen=True)
class Retrieval:
    """一次检索的结果与过程（过程写进运行记录，只有段落编号与分数，不含查询原文）。"""

    passages: tuple[Retrieved, ...]
    keyword: tuple[str, ...] = ()  # 关键词一路召回的段落编号（各查询合并）
    vector: tuple[str, ...] = ()  # 向量一路召回的段落编号；没有向量模型或不可用时为空
    reranked_by: Literal["model", "coverage", "none"] = "none"


class Retriever:
    """多查询、两路召回（关键词 + 向量）→ 倒数排名融合 → 重排（spec [2026-10-21] §12、§13）。

    - 每条查询各走两路；所有结果按排名融合、去重，取前若干条作为候选。
    - 重排：调用方给出的重排器（大模型）优先；它不可用时按覆盖比例打分。只保留审定过、
      不低于门槛的段落。
    - 问答卡优先级最低：本领域有其他过门槛的依据时，不用问答卡。
    - must：一定要进入依据的段落（命中触发词的安全类卡），不受门槛与优先级限制。
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        embedder: Embedder | None = None,
        recall_k: int = 20,
        candidates: int = 20,
        keep_k: int = 4,
        min_score: float = 0.2,
        model_min_score: float = 0.5,
        retry_after_s: float = 300.0,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self._kb = kb
        self._embedder = embedder
        self._coverage = CoverageReranker(kb.idf, kb.unseen_idf)
        self._recall_k, self._candidates, self._keep_k = recall_k, candidates, keep_k
        self._min, self._model_min = min_score, model_min_score
        self._retry_after, self._timer = retry_after_s, timer
        self._vectors: list[list[float]] | None = None
        self._failed_at: float | None = None
        self._lock = threading.Lock()

    def _passage_vectors(self) -> list[list[float]] | None:
        """段落向量：第一次用到时计算并缓存；失败后隔一段时间再试，期间只用关键词一路。"""
        if self._embedder is None:
            return None
        with self._lock:
            if self._vectors is not None:
                return self._vectors
            if self._failed_at is not None and self._timer() - self._failed_at < self._retry_after:
                return None
            try:
                vectors = self._embedder.embed([p.title + p.text for p in self._kb.passages])
            except Exception:  # 向量服务不可用：这一路跳过，不影响回答
                self._failed_at = self._timer()
                return None
            if len(vectors) != len(self._kb.passages):
                self._failed_at = self._timer()
                return None
            self._vectors = vectors
            return vectors

    def _vector_lists(
        self, queries: Sequence[str], allowed: frozenset[int], embedder: Embedder | None
    ) -> list[list[int]]:
        vectors = self._passage_vectors()
        if vectors is None:
            return []
        try:
            qvs = (embedder or self._embedder).embed(list(queries))  # type: ignore[union-attr]
        except Exception:
            return []
        lists = []
        for qv in qvs:
            sims = sorted(((i, _cosine(qv, vectors[i])) for i in allowed), key=lambda x: -x[1])
            lists.append([i for i, _ in sims[: self._recall_k]])
        return lists

    def retrieve(
        self,
        queries: Sequence[str],
        domain: AgentId,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        must: Iterable[str] = (),
    ) -> Retrieval:
        """queries 第一条为主查询（重排按它判断）；embedder 用于本次查询的向量（可带请求编号）。"""
        queries = [q for q in dict.fromkeys(queries) if q.strip()]
        allowed = frozenset(
            i
            for i, p in enumerate(self._kb.passages)
            if p.domain == domain and p.review == "approved"
        )
        if not allowed or not queries:
            return Retrieval(())
        keyword = [[i for i, _ in self._kb.bm25(q, allowed)[: self._recall_k]] for q in queries]
        vector = self._vector_lists(queries, allowed, embedder)
        fused: dict[int, float] = {}
        for ranked in (*keyword, *vector):
            for rank, i in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1 / (_RRF_K + rank + 1)
        order = sorted(fused, key=lambda i: -fused[i])[: self._candidates]
        candidates = [self._kb.passages[i] for i in order]

        def ids(lists: list[list[int]]) -> tuple[str, ...]:
            return tuple(
                dict.fromkeys(self._kb.passages[i].passage_id for ranked in lists for i in ranked)
            )

        forced_ids = list(dict.fromkeys(must))
        forced = [
            Retrieved(self._kb.passages[i], 1.0)
            for i in sorted(allowed)
            if self._kb.passages[i].passage_id in forced_ids
        ]
        kept, by = self._rerank(queries, candidates, reranker)
        kept = [r for r in kept if r.passage.passage_id not in forced_ids]
        own = [r for r in kept if r.passage.level != "card"]
        chosen = own or kept  # 问答卡优先级最低：有其他依据时不用卡
        return Retrieval(
            passages=tuple([*forced, *chosen][: max(self._keep_k, len(forced))]),
            keyword=ids(keyword),
            vector=ids(vector),
            reranked_by=by if candidates else "none",
        )

    def _rerank(
        self, queries: Sequence[str], candidates: list[Passage], reranker: Reranker | None
    ) -> tuple[list[Retrieved], Literal["model", "coverage"]]:
        if not candidates:
            return [], "coverage"
        scores = reranker.score(queries[0], candidates) if reranker is not None else None
        if scores is not None and len(scores) == len(candidates):
            threshold, by = self._model_min, "model"
        else:
            # 大模型重排不可用：按覆盖比例打分，取各条查询里最高的分
            per_query = [self._coverage.score(q, candidates) for q in queries]
            scores = [max(col) for col in zip(*per_query, strict=True)]
            threshold, by = self._min, "coverage"
        kept = sorted(
            (Retrieved(p, s) for p, s in zip(candidates, scores, strict=True) if s >= threshold),
            key=lambda r: -r.score,
        )
        return kept, by  # type: ignore[return-value]

    def passage(self, passage_id: str) -> Passage | None:
        return next((p for p in self._kb.passages if p.passage_id == passage_id), None)

    def search(self, query: str, domain: AgentId) -> list[Retrieved]:
        """单条查询、默认重排（兼容旧调用）。"""
        return list(self.retrieve([query], domain).passages)


def load_passages(folder: Path | None) -> list[Passage]:
    """读取资料目录下的全部 *.yaml（每个文件为 {passages: [...]}）；目录为空或不存在时返回空。"""
    if folder is None or not folder.is_dir():
        return []
    out: list[Passage] = []
    for path in sorted(folder.glob("*.yaml")):
        try:
            data = load_yaml(path)
            items = data.get("passages", []) if isinstance(data, dict) else None
            if not isinstance(items, list):
                raise KnowledgeLoadError("应为 {passages: [...]}")
            loaded = [Passage.model_validate_json(json.dumps(i)) for i in items]
            reserved = [p.passage_id for p in loaded if p.passage_id.startswith(RESERVED_PREFIXES)]
            if reserved:
                # 这两个前缀留给仓库自带的问答卡与动作库（可完整展示）；外部资料不能冒用
                raise KnowledgeLoadError(f"编号不能以 card: 或 exercise: 开头：{reserved}")
            out.extend(loaded)
        except (ValidationError, ValueError) as e:
            raise KnowledgeLoadError(f"{path}：知识段落有误：{e}") from e
    return out


class RetrievalParams(ContractModel):
    max_queries: Annotated[int, Field(ge=1, le=6)]  # 改写后的查询条数上限（含主查询）
    recall_k: Annotated[int, Field(ge=1, le=200)]
    candidates: Annotated[int, Field(ge=1, le=100)]
    keep_k: Annotated[int, Field(ge=1, le=20)]
    min_score: Annotated[float, Field(ge=0, le=1)]
    model_min_score: Annotated[float, Field(ge=0, le=1)]


def retriever_from_rules(
    kb: KnowledgeBase, rulebook: RuleBook, *, embedder: Embedder | None = None
) -> Retriever:
    """按规则表 R-QA-RETRIEVAL 的参数建检索器；参数不合规时抛错（启动时失败）。"""
    p = rulebook.get(RETRIEVAL_RULE).params_as(RetrievalParams)
    return Retriever(
        kb,
        embedder=embedder,
        recall_k=p.recall_k,
        candidates=p.candidates,
        keep_k=p.keep_k,
        min_score=p.min_score,
        model_min_score=p.model_min_score,
    )
