"""安全模块的词表匹配（spec §21、§22、§24）：急症词表与拒答词表共用。

匹配方式刻意保持简单（文字包含、分句内多组同时出现、血压数值、紧挨着的否定词），
App 离线版（app/src/safety/emergency.ts）逐步对应实现，结果必须一致。
"""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from ironm.contracts.common import ContractModel
from ironm.core.yamlfile import load_yaml

# 统一后（全角已转半角）的分句符号与空白。前端离线版使用同一组字符。
# 不含顿号：“胸口、左胳膊都疼”是同一分句，拆开会让组合说法匹配不上（冲刺 2 末审查 H1）
CLAUSE_SEPARATORS = ",。!?;\n\r~"
_SPACES = [0x20, 0x09, 0x0C, 0x0B, 0xA0, 0x1680, *range(0x2000, 0x200C), 0x2028, 0x2029]
WHITESPACE = "".join(map(chr, [*_SPACES, 0x202F, 0x205F, 0x3000, 0xFEFF]))
_CLAUSE_RE = re.compile(f"[{re.escape(CLAUSE_SEPARATORS)}]+")
_SPACE_RE = re.compile(f"[{re.escape(WHITESPACE)}]+")
# 血压：120/80 这样的分数，或“高压 190”“低压 120”；位数在代码里判断，正则保持两端都能用的写法
_BP_PAIR_RE = re.compile(r"([0-9]+)/([0-9]+)")
_BP_HIGH_RE = re.compile(r"高压([0-9]+)")
_BP_LOW_RE = re.compile(r"低压([0-9]+)")


# 小号与半角标点转成对应的常用写法（冲刺 2 复核 N4）：｡ ﹐ ﹒ ﹔ ﹖ ﹗
PUNCTUATION = {0xFF61: 0x3002, 0xFE50: 0x2C, 0xFE52: 0x2E, 0xFE54: 0x3B, 0xFE56: 0x3F, 0xFE57: 0x21}


def unify(text: str) -> str:
    """全角 ASCII（U+FF01–FF5E）转半角、全角空格转空格、小号与半角标点转常用写法、
    英文字母转小写，其余字符不变（上标数字等不转）。

    不用 Unicode 规范化：不同运行环境的 Unicode 版本不同，结果会不一致（冲刺 2 末审查 L1）。
    App 离线版按同样的码点规则实现。
    """
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        elif code == 0x3000:
            code = 0x20
        else:
            code = PUNCTUATION.get(code, code)
        if 0x41 <= code <= 0x5A:
            code += 0x20
        out.append(chr(code))
    return "".join(out)


def clauses(text: str) -> list[str]:
    """统一文字后按分句符号切开，再去掉空白；空分句丢弃。"""
    parts = (_SPACE_RE.sub("", p) for p in _CLAUSE_RE.split(unify(text)))
    return [p for p in parts if p]


def check_term(term: str) -> str:
    if not term:
        raise ValueError("说法不能为空")
    if clauses(term) != [term]:
        raise ValueError(f"说法“{term}”不是统一后的写法，或含有分句符号、空白")
    return term


type Term = Annotated[str, Field(min_length=1)]


def _bp_value(digits: str) -> int | None:
    return int(digits) if 2 <= len(digits) <= 3 else None


class BloodPressure(ContractModel):
    context: tuple[Term, ...] = Field(min_length=1)  # 同一分句里须出现的字样，如“血压”
    systolic_at_least: int = Field(gt=0)
    diastolic_at_least: int = Field(gt=0)

    def matches(self, clause: str) -> bool:
        if not any(c in clause for c in self.context):
            return False
        systolic: list[int | None] = [_bp_value(m.group(1)) for m in _BP_PAIR_RE.finditer(clause)]
        diastolic: list[int | None] = [_bp_value(m.group(2)) for m in _BP_PAIR_RE.finditer(clause)]
        systolic += [_bp_value(m.group(1)) for m in _BP_HIGH_RE.finditer(clause)]
        diastolic += [_bp_value(m.group(1)) for m in _BP_LOW_RE.finditer(clause)]
        return any(v is not None and v >= self.systolic_at_least for v in systolic) or any(
            v is not None and v >= self.diastolic_at_least for v in diastolic
        )


class Negation(ContractModel):
    """否定与“以前发生过”（spec §22）。只取消紧挨着的一处说法；拿不准时按出现处理。"""

    before: tuple[Term, ...] = Field(min_length=1)  # 说法前紧挨着这些词：这一处不算
    exceptions: tuple[Term, ...] = ()  # 但前面以这些词结尾时仍算（“差点没晕倒”）
    # 说法后紧跟这个字：以前发生过（拒答词表等不需要时不写）
    past_after: Term | None = None
    past_not_followed_by: tuple[Term, ...] = ()  # 再接这些字时不是“以前发生过”（“晕倒过去”）
    # 同一分句或后面的分句有复发说法：以前发生过的也照常算（“以前胸口疼过，今天又疼了”）……
    recurrence: tuple[Term, ...] = ()
    # ……但后面那个分句说的是别的部位时不算（“以前晕倒过，最近腿也疼”）
    body_parts: tuple[Term, ...] = ()
    # 同一分句或前一个分句有近期字样：以前发生过的也照常算（“今天跑步的时候，晕倒过一次”）……
    recent: tuple[Term, ...] = ()
    # ……但所在分句有久远的时间、又没有近期字样时不算（“去年晕倒过一次，最近想开始减肥”）
    remote: tuple[Term, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        past = [] if self.past_after is None else [self.past_after]
        words = [*self.before, *self.exceptions, *past, *self.past_not_followed_by]
        words += [*self.recurrence, *self.body_parts, *self.recent, *self.remote]
        for word in words:
            check_term(word)
        return self

    def past_counts(self, parts: list[str]) -> list[bool]:
        """每个分句里“以前发生过”的说法是否照常算。"""

        def has(clause: str, words: tuple[str, ...]) -> bool:
            return any(w in clause for w in words)

        marks: list[bool] = []
        for i, clause in enumerate(parts):
            later = parts[i + 1 :]
            again = has(clause, self.recurrence) or any(
                has(c, self.recurrence) and not has(c, self.body_parts) for c in later
            )
            here = has(clause, self.recent)
            near = here or (i > 0 and has(parts[i - 1], self.recent))
            marks.append(again or (near and (here or not has(clause, self.remote))))
        return marks

    def counts(self, clause: str, start: int, end: int, past_counts: bool) -> bool:
        """clause[start:end] 这一处说法算不算出现。"""
        head = clause[:start]
        negated = any(head.endswith(w) for w in self.before)
        if negated and not any(head.endswith(w) for w in self.exceptions):
            return False
        tail = clause[end:]
        if past_counts or self.past_after is None or not tail.startswith(self.past_after):
            return True
        rest = tail[len(self.past_after) :]
        return any(rest.startswith(w) for w in self.past_not_followed_by)

    def found(
        self, clause: str, term: str, past_counts: bool, not_followed_by: tuple[str, ...] = ()
    ) -> bool:
        """term 在分句里是否有一处算出现；后面紧跟 not_followed_by 中的字时这一处不算。"""
        i = clause.find(term)
        while i != -1:
            end = i + len(term)
            followed = any(clause.startswith(w, end) for w in not_followed_by)
            if not followed and self.counts(clause, i, end, past_counts):
                return True
            i = clause.find(term, i + 1)
        return False


class Symptom(ContractModel):
    """一组同义说法：terms 任一出现即命中；combos 每项是几组词，同一分句里每组都出现才命中。"""

    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    name: Term
    terms: tuple[Term, ...] = ()
    combos: tuple[tuple[tuple[Term, ...], ...], ...] = ()
    blood_pressure: BloodPressure | None = None
    # 说法后紧跟这些字时这一处不算（如“吃完就吐”后接“槽”）
    not_followed_by: tuple[Term, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not (self.terms or self.combos or self.blood_pressure):
            raise ValueError(f"症状 {self.id} 没有任何说法")
        for combo in self.combos:
            if len(combo) < 2 or any(not group for group in combo):
                raise ValueError(f"症状 {self.id} 的组合至少两组，且每组至少一个说法")
        words = [*self.terms, *(t for combo in self.combos for g in combo for t in g)]
        words += self.not_followed_by
        if self.blood_pressure is not None:
            words += self.blood_pressure.context
        for word in words:
            check_term(word)
        return self

    def matches(self, clause: str, negation: Negation, past_counts: bool) -> bool:
        def found(term: str) -> bool:
            return negation.found(clause, term, past_counts, self.not_followed_by)

        if any(found(t) for t in self.terms):
            return True
        if any(all(any(found(t) for t in g) for g in combo) for combo in self.combos):
            return True
        # 血压按数值判断，不受否定影响
        return self.blood_pressure is not None and self.blood_pressure.matches(clause)


class Phrasebook(ContractModel):
    """词表的公共部分：版本、审定状态、否定规则与一组症状（或信号）。"""

    version: Term
    review: Literal["pending_review", "approved"]
    negation: Negation
    symptoms: tuple[Symptom, ...] = Field(min_length=1)

    def present(self, text: str) -> frozenset[str]:
        """这句话里出现的症状编号。"""
        parts = clauses(text)
        neg = self.negation
        marked = list(zip(parts, neg.past_counts(parts), strict=True))
        return frozenset(
            s.id for s in self.symptoms if any(s.matches(c, neg, r) for c, r in marked)
        )

    def check_rules(self, rules: Iterable[tuple[str, tuple[str, ...]]]) -> None:
        """条目编号唯一、引用的症状都存在、每个症状至少被一个条目使用。"""
        rules = list(rules)
        unique((s.id for s in self.symptoms), "症状")
        unique((rule_id for rule_id, _ in rules), "条目")
        known = {s.id for s in self.symptoms}
        used: set[str] = set()
        for rule_id, refs in rules:
            missing = [s for s in refs if s not in known]
            if missing:
                raise ValueError(f"条目 {rule_id} 引用了不存在的症状：{missing}")
            if len(set(refs)) != len(refs):
                raise ValueError(f"条目 {rule_id} 重复引用了同一症状")
            used.update(refs)
        unused = [s.id for s in self.symptoms if s.id not in used]
        if unused:
            raise ValueError(f"症状没有被任何条目使用：{unused}")


class TermLoadError(ValueError):
    pass


def unique(ids: Iterable[str], kind: str) -> None:
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"{kind}编号重复：{i}")
        seen.add(i)


def load_file[M: BaseModel](path: Path, model: type[M], label: str) -> M:
    """读取并严格校验一份安全文件；任何问题都抛 TermLoadError。"""
    try:
        data = load_yaml(path)
        return model.model_validate_json(json.dumps(data))
    except (OSError, ValueError, TypeError) as e:
        # ValidationError 是 ValueError 的子类
        detail = e if not isinstance(e, ValidationError) else _summary(e)
        raise TermLoadError(f"{path}：{label}有误：{detail}") from e
    except yaml.YAMLError as e:
        raise TermLoadError(f"{path}：{label}格式错误：{e}") from e


def _summary(e: ValidationError) -> str:
    return "；".join(f"{'.'.join(map(str, err['loc']))}：{err['msg']}" for err in e.errors())
