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
import statistics
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
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
    "OncePerRequestEmbedder",
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
    """一段知识。出处给用户看；证据级别决定表述的语气（指南说“建议”，单项研究说“有研究显示”）。

    后五个字段为可选，供人工维护时使用：
    - keywords：用户实际会说的口语说法，检索时加权命中
    - caution：需要一并告诉用户的注意事项
    - see_also：相关段落编号（内容上互相补充或对照）
    - reviewed_at：最后审定日期；超过一年由检查脚本提示复审
    - superseded_by：被哪一段取代；有值时不再作为依据
    """

    passage_id: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:\-]{0,120}$")]
    domain: AgentId
    title: Text
    text: Text
    source: Text
    level: Level
    review: Literal["pending_review", "approved"]
    keywords: tuple[Text, ...] = ()
    caution: Text | None = None
    see_also: tuple[str, ...] = ()
    reviewed_at: date | None = None
    superseded_by: str | None = None


def _indexable(p: Passage) -> str:
    """参与检索的文本：标题 + 正文 + 口语关键词（用户在问答里多半用关键词里的说法）。"""
    return " ".join([p.title, p.text, *p.keywords])


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


# 问句里的虚词与口头语：不表示问的是什么，算覆盖比例前去掉（只用于没有大模型重排时的打分）。
# 只去掉多字的问句词与句末语气词；“有”“能”“要”这类单字常在实词里（有氧、能量、要点），不去
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
    "的",
    "我",
    "你",
    "这个",
    "那个",
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
        # 知识库里从没出现过的片段（如“跑鞋”）也计入，没被覆盖就拉低分数，
        # 避免只凭“什么”这类常见片段就判为全覆盖
        q = set(_grams(_QUESTION_WORDS.sub("，", query)))
        weight = {g: self._idf.get(g, self._unseen) for g in q}
        total = sum(weight.values())
        if total == 0:
            return [0.0 for _ in passages]
        out = []
        for p in passages:
            body = set(_grams(_indexable(p)))
            out.append(sum(w for g, w in weight.items() if g in body) / total)
        return out


@dataclass(frozen=True)
class Retrieved:
    passage: Passage
    score: float  # 重排分


class KnowledgeBase:
    def __init__(self, passages: Iterable[Passage]) -> None:
        # 被取代的段落保留（历史回答要能追溯），但不再参与检索
        self.passages = tuple(p for p in passages if p.superseded_by is None)
        ids = [p.passage_id for p in self.passages]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise KnowledgeLoadError(f"知识段落编号重复：{dup}")
        self._docs = [Counter(_grams(_indexable(p))) for p in self.passages]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(self._docs)
        self._idf = {g: math.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}
        # 没出现过的片段按中位权重计入：比“什么”这类常见片段重，又不至于压过真正的专业词
        self.unseen_idf = statistics.median(self._idf.values()) if self._idf else 0.0

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


type VectorStatus = Literal["ok", "off", "warming", "unavailable"]


@dataclass(frozen=True)
class Retrieval:
    """一次检索的结果与过程（过程写进运行记录，只有段落编号与分数，不含查询原文）。"""

    passages: tuple[Retrieved, ...]
    keyword: tuple[str, ...] = ()  # 关键词一路召回的段落编号（各查询合并）
    vector: tuple[str, ...] = ()  # 向量一路召回的段落编号；没有向量模型或不可用时为空
    reranked_by: Literal["model", "coverage", "none"] = "none"
    # off：没有配置向量服务；warming：段落向量还在计算（服务正常）；unavailable：服务出错或在冷却
    vector_status: VectorStatus = "off"


class OncePerRequestEmbedder:
    """同一次请求里，几个 Agent 用同一组查询：向量只算一次（失败也只失败一次）。"""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._done: dict[tuple[str, ...], list[list[float]] | BaseException] = {}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        key = tuple(texts)
        with self._lock:  # 同一组查询的并发请求排队，第一个算完后其余直接取结果
            if key not in self._done:
                try:
                    self._done[key] = self._inner.embed(list(texts))
                except Exception as e:
                    self._done[key] = e
            got = self._done[key]
        if isinstance(got, BaseException):
            raise got
        return got


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
        self._computing = False
        self._lock = threading.Lock()

    def _cooling(self) -> bool:
        return self._failed_at is not None and self._timer() - self._failed_at < self._retry_after

    def _fail(self) -> None:
        with self._lock:
            self._failed_at = self._timer()

    def warm(self) -> bool:
        """预先计算段落向量（服务启动后在后台调用）；返回是否已可用。"""
        vectors, _ = self._passage_vectors()
        return vectors is not None

    def _passage_vectors(self) -> tuple[list[list[float]] | None, VectorStatus]:
        """段落向量：第一次用到时计算并缓存；正在计算时其他请求不等待，先只用关键词一路
        （记为预热中，不算服务出错）；失败后隔一段时间再试。"""
        if self._embedder is None:
            return None, "off"
        with self._lock:
            if self._cooling():
                return None, "unavailable"
            if self._computing:
                return None, "warming"
            if self._vectors is not None:
                return self._vectors, "ok"
            self._computing = True
        vectors: list[list[float]] | None = None
        try:
            # 与关键词一路用同样的文本：标题 + 正文 + 口语关键词
            got = self._embedder.embed([_indexable(p) for p in self._kb.passages])
            dims = {len(v) for v in got}
            if len(got) == len(self._kb.passages) and len(dims) == 1 and 0 not in dims:
                vectors = got
        except Exception:  # 向量服务不可用：这一路跳过，不影响回答
            vectors = None
        with self._lock:
            self._computing = False
            if vectors is None:
                self._failed_at = self._timer()
            else:
                self._vectors = vectors
        return vectors, "ok" if vectors is not None else "unavailable"

    def _vector_lists(
        self, queries: Sequence[str], allowed: frozenset[int], embedder: Embedder | None
    ) -> tuple[list[list[int]], VectorStatus]:
        vectors, status = self._passage_vectors()
        source = embedder or self._embedder
        if vectors is None or source is None:
            return [], status
        try:
            qvs = source.embed(list(queries))
            if len(qvs) != len(queries) or any(len(qv) != len(vectors[0]) for qv in qvs):
                raise ValueError("向量维度不一致")
        except Exception:  # 查询向量失败（含维度对不上）：这一路跳过，并进入冷却
            self._fail()
            return [], "unavailable"
        lists = []
        for qv in qvs:
            sims = sorted(((i, _cosine(qv, vectors[i])) for i in allowed), key=lambda x: -x[1])
            lists.append([i for i, _ in sims[: self._recall_k]])
        return lists, "ok"

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
        vector, vector_status = self._vector_lists(queries, allowed, embedder)
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
            vector_status=vector_status,
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
