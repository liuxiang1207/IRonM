"""测试数据文件（红旗、对抗、黄金用例）的格式、一致性检查与被测功能解析。

被测功能尚未实现（status: pending）时，resolve() 抛 NotImplementedError，用例据此标为预期失败；
只有“目标模块或函数不存在”才算未实现，模块内部的其他导入错误照常抛出。
"""

import importlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ironm.rules.loader import RuleBook, load_yaml

type Text = Annotated[str, Field(min_length=1)]
type Status = Literal["pending", "active"]
type Review = Literal["pending_review", "approved"]


class CaseError(ValueError):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---- 红旗测试集 ----

POSITIVE_CATEGORIES = frozenset(
    {
        "cardiovascular",
        "syncope",
        "respiratory",
        "neuro",
        "musculoskeletal",
        "heat",
        "metabolic",
        "pregnancy",
        "eating_disorder",
        "other_person",
    }
)
NEGATIVE_CATEGORIES = frozenset({"negation", "keyword_only", "benign", "resolved"})


class RedFlagCase(_Model):
    id: Annotated[str, Field(pattern=r"^RF-\d{3}$")]
    text: Text
    expect: Literal["emergency", "red_flag", "none"]
    category: Text
    note: Text | None = None
    pending: Text | None = None  # 被测功能已启用、但这条依赖的部分尚未实现：写明原因，标为预期失败


class RedFlagFile(_Model):
    target: Text
    status: Status
    review: Review
    cases: tuple[RedFlagCase, ...]


# ---- 对抗题库 ----

ADVERSARIAL_CATEGORIES = frozenset(
    {
        "extreme_weight_loss",
        "purge_compensation",
        "exercise_chest_pain",
        "underweight_cut",
        "banned_drugs",
        "pseudoscience",
        "diagnosis_inducement",
        "weight_drug_dose",
        "prescription_treatment",
        "minor",
        "pregnancy",
        "eating_disorder_signal",
    }
)


class AdversarialChecks(_Model):
    """机器判定（spec [2026-10-21] §9）。

    contains 每组至少出现一个说法；forbids 是一票否决的正则。
    """

    level: Literal["emergency", "red_flag", "none"] | None = None
    contains: tuple[tuple[Text, ...], ...] = Field(min_length=1)
    forbids: tuple[Text, ...] = ()


class AdversarialCase(_Model):
    id: Annotated[str, Field(pattern=r"^ADV-\d{3}$")]
    category: Text
    prompt: Text
    action: Literal["emergency", "refuse_refer", "correct", "no_plan_refer", "safe_alternative"]
    must: tuple[Text, ...] = Field(min_length=1)  # 回答必须包含的要点
    hard_fail: tuple[Text, ...] = Field(min_length=1)  # 一票否决：命中任一项该题判失败
    note: Text | None = None
    checks: AdversarialChecks | None = None  # 问答实现后由机器判定


class AdversarialFile(_Model):
    target: Text
    status: Status
    review: Review
    cases: tuple[AdversarialCase, ...]


# ---- 黄金用例 ----


class GoldenCase(_Model):
    id: Annotated[str, Field(pattern=r"^G(-[A-Z0-9]+)+$")]
    title: Text
    given: dict[str, object]
    expect: dict[str, object]
    twin_of: str | None = None  # 孪生用例：只改一个条件，期望应随之改变


class GoldenFile(_Model):
    target: Text
    status: Status
    reason: Text | None = None
    inject: tuple[Literal["rulebook"], ...] = ()  # 运行时额外传给被测函数的参数
    rules: tuple[Text, ...]
    cases: tuple[GoldenCase, ...] = Field(min_length=1)


def load_case_file[M: BaseModel](path: Path, model: type[M]) -> M:
    return model.model_validate(load_yaml(path))


def check_unique_ids(ids: Iterable[str]) -> None:
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            raise CaseError(f"用例编号重复：{i}")
        seen.add(i)


def check_golden(files: Iterable[GoldenFile], rulebook: RuleBook) -> None:
    files = list(files)
    check_unique_ids(c.id for f in files for c in f.cases)
    for f in files:
        for rule_id in f.rules:
            try:
                rulebook.get(rule_id)
            except KeyError as e:
                raise CaseError(f"{f.target} 引用了不存在的规则：{rule_id}") from e
        by_id = {c.id: c for c in f.cases}
        for c in f.cases:
            if c.twin_of is None:
                continue
            twin = by_id.get(c.twin_of)
            if twin is None:
                raise CaseError(f"{c.id} 的孪生用例 {c.twin_of} 不在同一文件中")
            keys = set(c.given) | set(twin.given)
            diff = [k for k in keys if c.given.get(k) != twin.given.get(k)]
            if len(diff) != 1:
                raise CaseError(f"{c.id} 与 {twin.id} 应恰好只差一个条件，实际差 {sorted(diff)}")
            if c.expect == twin.expect:
                raise CaseError(f"{c.id} 与 {twin.id} 只差一个条件，期望却相同")


def resolve(target: str, status: Status) -> Callable[..., object]:
    module_name, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        missing_is_target = e.name is not None and module_name.startswith(e.name)
        if status == "pending" and missing_is_target:
            raise NotImplementedError(f"{target} 尚未实现") from e
        raise
    fn: object = getattr(module, attr, None)
    if fn is None:
        if status == "pending":
            raise NotImplementedError(f"{target} 尚未实现")
        raise CaseError(f"{target} 不存在")
    if not callable(fn):
        raise CaseError(f"{target} 不可调用")
    return fn
