"""知识库与检索（spec [2026-10-21] §12）：召回（关键词 + 向量）→ 重排。

- 知识库是一组段落，每段标上领域（负责该领域的专业角色）、出处、证据级别与审定状态。
  初始内容来自问答卡与动作库；资料目录（不进仓库）里的段落随配置加载。
- 召回：关键词按中文两字片段做 BM25；配置了向量模型时再做向量召回，两路按倒数排名融合（RRF）。
- 重排：默认按“问题的片段被段落覆盖的比例”打分，可换成重排模型；低于门槛的段落不算依据。
检索只查表与计算，不调用大模型。
"""

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, ValidationError

from ironm.contracts.common import ContractModel
from ironm.contracts.roles import AgentId
from ironm.core.yamlfile import load_yaml
from ironm.knowledge.exercises import Library
from ironm.knowledge.qa_cards import CardBook
from ironm.safety.matching import clauses

__all__ = [
    "Embedder",
    "KnowledgeBase",
    "KnowledgeLoadError",
    "Passage",
    "Reranker",
    "Retrieved",
    "Retriever",
    "load_passages",
]

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
    """重排：给每段打 0–1 的相关分。"""

    def score(self, query: str, passages: Sequence[Passage]) -> list[float]: ...


class CoverageReranker:
    """默认重排：问题的两字片段被段落覆盖的比例，按片段的稀有程度（IDF）加权。

    “怎么、多少”这类到处都有的问句片段权重低，“肌酸、蛋白”这类专业词权重高。
    """

    def __init__(self, idf: dict[str, float] | None = None) -> None:
        self._idf = idf or {}

    def score(self, query: str, passages: Sequence[Passage]) -> list[float]:
        q = set(_grams(query))
        weight = {g: self._idf.get(g, 0.0) for g in q}
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


class Retriever:
    """按领域召回、融合、重排。只返回审定过且重排分不低于门槛的段落。"""

    def __init__(
        self,
        kb: KnowledgeBase,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        recall_k: int = 20,
        keep_k: int = 4,
        min_score: float = 0.3,
    ) -> None:
        self._kb = kb
        self._embedder = embedder
        self._reranker = reranker or CoverageReranker(kb.idf)
        self._recall_k, self._keep_k, self._min = recall_k, keep_k, min_score
        self._vectors = (
            embedder.embed([p.title + p.text for p in kb.passages]) if embedder else None
        )

    def search(self, query: str, domain: AgentId) -> list[Retrieved]:
        allowed = frozenset(
            i
            for i, p in enumerate(self._kb.passages)
            if p.domain == domain and p.review == "approved"
        )
        if not allowed:
            return []
        lists = [[i for i, _ in self._kb.bm25(query, allowed)[: self._recall_k]]]
        if self._embedder is not None and self._vectors is not None:
            [qv] = self._embedder.embed([query])
            sims = sorted(
                ((i, _cosine(qv, self._vectors[i])) for i in allowed), key=lambda x: -x[1]
            )
            lists.append([i for i, _ in sims[: self._recall_k]])
        fused: dict[int, float] = {}
        for ranked in lists:
            for rank, i in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1 / (_RRF_K + rank + 1)
        candidates = [self._kb.passages[i] for i, _ in sorted(fused.items(), key=lambda x: -x[1])]
        scores = self._reranker.score(query, candidates)
        kept = sorted(
            (Retrieved(p, s) for p, s in zip(candidates, scores, strict=True) if s >= self._min),
            key=lambda r: -r.score,
        )
        return kept[: self._keep_k]


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
            out.extend(Passage.model_validate_json(json.dumps(i)) for i in items)
        except (ValidationError, ValueError) as e:
            raise KnowledgeLoadError(f"{path}：知识段落有误：{e}") from e
    return out
