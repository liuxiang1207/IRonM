"""模型选型评测。用法：make eval-models（密钥取自环境变量，见 models.yaml 的 api_key_env）；
修正判分规则后用 python -m evals.model_selection.run --rescore <结果目录> 重新判分，不调用接口。

每道题只调用一次：超时或出错记为该题失败并单独计数，不重试。
结果写到 data/evals/model_selection/<时间>/：每个配置一份逐题明细 .jsonl，汇总 report.md。
"""

import json
import math
import os
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.model_selection.scoring import (
    ExtractExpect,
    Price,
    Usage,
    Verdict,
    cost_usd,
    score_extraction,
    score_text,
)
from ironm.core.clock import SystemClock
from ironm.rules.loader import load_yaml

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "questions.yaml"
MODELS_PATH = HERE / "models.yaml"
OUT_ROOT = Path("data/evals/model_selection")
MAX_TOKENS = 3000
TIMEOUT_S = 180

type Transport = Callable[[str, dict[str, str], dict[str, object]], dict[str, object]]
type Groups = tuple[tuple[str, ...], ...]
type Kind = Literal["extraction", "qa", "adversarial"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Prices(_Model):
    offpeak: Price
    peak: Price


class ModelConfig(_Model):
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")]
    thinking: bool
    extra_body: dict[str, object]
    price_usd_per_mtok: Prices


class ExtractionQ(_Model):
    id: str
    text: str
    expect: ExtractExpect


class TextQ(_Model):
    id: str
    prompt: str
    must_any: Groups = Field(min_length=1)
    forbid: tuple[str, ...] = ()
    source: str | None = None  # 对抗题对应的题库编号


class QuestionSet(_Model):
    extraction: tuple[ExtractionQ, ...]
    qa: tuple[TextQ, ...]
    adversarial: tuple[TextQ, ...]


class _ModelList(_Model):
    models: tuple[ModelConfig, ...]


def load_questions(path: Path) -> QuestionSet:
    return QuestionSet.model_validate(load_yaml(path))


def load_models(path: Path) -> tuple[ModelConfig, ...]:
    return _ModelList.model_validate(load_yaml(path)).models


SYSTEM_EXTRACT = """\
你负责把用户的一句话转成一条结构化记录。只输出一个 JSON 对象，不要输出其他文字：
{"type": <类型>, "payload": {<字段>}}

类型与字段（单位统一：千克、厘米、千卡、克、秒）：
- weight：weight_kg
- meal：slot（breakfast / lunch / dinner / snack），kcal，protein_g
- workout_check：status（done / half / skipped），effort（1–5 的吃力程度）
- set_log：exercise_id（squat / bench_press / deadlift / pull_up / push_up / row /
  overhead_press / lunge），set_index（第几组），reps，load_kg（自重为 0）
- checkin：fatigue、sleep_quality、energy、hunger（均为 1–5），diet_adherence_days（0–7），
  clothes_fit（looser / same / tighter）
- symptom：text（原话），severity（1–5）
- circumference：waist_cm，hip_cm，chest_cm，upper_arm_cm，thigh_cm
- baseline_test：test_id（pushup_max 俯卧撑最多个数 / plank_hold 平板支撑时长 /
  squat_60s 60 秒深蹲个数），value，unit（reps / s）

规则：用户没有明确说出的数值一律不要填写，不要估算或编造；需要换算单位时换算到上述单位。"""

SYSTEM_QA = "你是一名严谨的健身与营养助手。用中文简洁回答（150 字以内），给出有依据的建议。"


@dataclass(frozen=True)
class CallResult:
    text: str | None
    usage: Usage
    latency_s: float
    error: str | None


def http_transport(url: str, headers: dict[str, str], body: dict[str, object]) -> dict[str, object]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data: dict[str, object] = json.loads(resp.read())
        return data


def _usage(resp: dict[str, object]) -> Usage:
    u = resp.get("usage")
    if not isinstance(u, dict):
        raise ValueError("响应缺少 usage")
    hit = int(u.get("prompt_cache_hit_tokens", 0))
    miss = int(u.get("prompt_cache_miss_tokens", int(u["prompt_tokens"]) - hit))
    return Usage(input_hit=hit, input_miss=miss, output=int(u["completion_tokens"]))


def call(cfg: ModelConfig, system: str, user: str, transport: Transport) -> CallResult:
    headers = {
        "Authorization": f"Bearer {os.environ[cfg.api_key_env]}",
        "Content-Type": "application/json",
    }
    body: dict[str, object] = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": MAX_TOKENS,
        **cfg.extra_body,
    }
    start = time.perf_counter()
    try:
        resp = transport(f"{cfg.base_url}/chat/completions", headers, body)
        choices = resp["choices"]
        assert isinstance(choices, list)
        text = str(choices[0]["message"]["content"])
        usage = _usage(resp)
    except Exception as e:
        return CallResult(None, Usage(), time.perf_counter() - start, f"{type(e).__name__}: {e}")
    return CallResult(text, usage, time.perf_counter() - start, None)


def _judge(q: ExtractionQ | TextQ, answer: str) -> Verdict:
    if isinstance(q, ExtractionQ):
        return score_extraction(answer, q.expect)
    return score_text(answer, q.must_any, q.forbid)


@dataclass(frozen=True)
class Record:
    """一道题的调用与判分结果。"""

    model: str
    kind: Kind
    id: str
    answer: str | None
    error: str | None
    latency_s: float
    usage: Usage
    passed: bool
    refused: bool
    reasons: tuple[str, ...]
    cost_offpeak_usd: float
    cost_peak_usd: float


@dataclass(frozen=True)
class Row:
    """一个配置的汇总。"""

    name: str
    extraction_acc: float
    qa_acc: float
    adversarial_pass: float
    over_refusal: float
    errors: int
    latency_p50_s: float | None
    latency_p95_s: float | None
    avg_output_tokens: float
    avg_cost_offpeak_usd: float
    avg_cost_peak_usd: float


def _run_one(cfg: ModelConfig, kind: Kind, q: ExtractionQ | TextQ, transport: Transport) -> Record:
    system, user = (SYSTEM_EXTRACT, q.text) if isinstance(q, ExtractionQ) else (SYSTEM_QA, q.prompt)
    r = call(cfg, system, user, transport)
    v = Verdict(False, ("调用失败",)) if r.text is None else _judge(q, r.text)
    prices = cfg.price_usd_per_mtok
    return Record(
        model=cfg.name,
        kind=kind,
        id=q.id,
        answer=r.text,
        error=r.error,
        latency_s=round(r.latency_s, 3),
        usage=r.usage,
        passed=v.passed,
        refused=v.refused,
        reasons=v.reasons,
        cost_offpeak_usd=cost_usd(r.usage, prices.offpeak),
        cost_peak_usd=cost_usd(r.usage, prices.peak),
    )


def _pct(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * p) - 1)]  # 最近秩法


def _summary(cfg: ModelConfig, records: list[Record]) -> Row:
    def rate(rs: list[Record], flag: Callable[[Record], bool]) -> float:
        return sum(flag(r) for r in rs) / len(rs)

    def of(kind: Kind) -> list[Record]:
        return [r for r in records if r.kind == kind]

    latencies = [r.latency_s for r in records if r.error is None]
    n = len(records)
    return Row(
        name=cfg.name,
        extraction_acc=rate(of("extraction"), lambda r: r.passed),
        qa_acc=rate(of("qa"), lambda r: r.passed),
        adversarial_pass=rate(of("adversarial"), lambda r: r.passed),
        over_refusal=rate(of("qa"), lambda r: r.refused),
        errors=sum(r.error is not None for r in records),
        latency_p50_s=_pct(latencies, 0.5),
        latency_p95_s=_pct(latencies, 0.95),
        avg_output_tokens=sum(r.usage.output for r in records) / n,
        avg_cost_offpeak_usd=sum(r.cost_offpeak_usd for r in records) / n,
        avg_cost_peak_usd=sum(r.cost_peak_usd for r in records) / n,
    )


def _report(rows: list[Row]) -> str:
    def pct(v: float) -> str:
        return f"{v * 100:.0f}%"

    def num(v: float | None, digits: int = 2) -> str:
        return "—" if v is None else f"{v:.{digits}f}"

    lines = [
        "| 配置 | 抽取准确率 | 问答命中率 | 对抗通过率 | 误拒率 | 失败 | 延迟中位 s | 延迟 P95 s"
        " | 平均输出 tokens | 单题成本 非高峰 $ | 单题成本 高峰 $ |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | {pct(r.extraction_acc)} | {pct(r.qa_acc)} | {pct(r.adversarial_pass)}"
            f" | {pct(r.over_refusal)} | {r.errors} | {num(r.latency_p50_s)}"
            f" | {num(r.latency_p95_s)} | {num(r.avg_output_tokens, 0)}"
            f" | {num(r.avg_cost_offpeak_usd, 6)} | {num(r.avg_cost_peak_usd, 6)} |"
        )
    return "\n".join(lines) + "\n"


def _write(out_dir: Path, results: dict[str, list[Record]], rows: list[Row]) -> None:
    for name, records in results.items():
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_report(rows), encoding="utf-8")


def _items(questions: QuestionSet) -> list[tuple[Kind, ExtractionQ | TextQ]]:
    return [
        *(("extraction", q) for q in questions.extraction),
        *(("qa", q) for q in questions.qa),
        *(("adversarial", q) for q in questions.adversarial),
    ]


def rescore(out_dir: Path, models: Sequence[ModelConfig], questions: QuestionSet) -> list[Row]:
    """按当前判分标准重新判已保存的回答，不调用接口。用于修正判分规则后的复核。"""
    results: dict[str, list[Record]] = {}
    for cfg in models:
        saved = {}
        for line in (out_dir / f"{cfg.name}.jsonl").read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            saved[data["id"]] = Record(**{**data, "usage": Usage(**data["usage"]), "reasons": ()})
        records = []
        for kind, q in _items(questions):
            old = saved[q.id]
            v = Verdict(False, ("调用失败",)) if old.answer is None else _judge(q, old.answer)
            records.append(
                replace(old, kind=kind, passed=v.passed, refused=v.refused, reasons=v.reasons)
            )
        results[cfg.name] = records
    rows = [_summary(cfg, results[cfg.name]) for cfg in models]
    _write(out_dir, results, rows)
    return rows


def run_all(
    models: Sequence[ModelConfig],
    questions: QuestionSet,
    out_dir: Path,
    transport: Transport,
    workers: int = 4,
) -> list[Row]:
    for cfg in models:
        if cfg.api_key_env not in os.environ:
            raise KeyError(f"缺少环境变量 {cfg.api_key_env}（{cfg.name}）")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[Record]] = {}
    for cfg in models:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_one, cfg, kind, q, transport) for kind, q in _items(questions)
            ]
            results[cfg.name] = [fut.result() for fut in futures]
    rows = [_summary(cfg, results[cfg.name]) for cfg in models]
    _write(out_dir, results, rows)
    return rows


if __name__ == "__main__":
    if sys.argv[1:2] == ["--rescore"]:
        target = Path(sys.argv[2])
        names = {p.stem for p in target.glob("*.jsonl")}
        cfgs = [m for m in load_models(MODELS_PATH) if m.name in names]
        rescore(target, cfgs, load_questions(QUESTIONS_PATH))
        print((target / "report.md").read_text(encoding="utf-8"))
        sys.exit(0)
    selected = set(sys.argv[1:])
    models = [m for m in load_models(MODELS_PATH) if not selected or m.name in selected]
    stamp = SystemClock().now().strftime("%Y%m%dT%H%M%SZ")
    out = OUT_ROOT / stamp
    run_all(models, load_questions(QUESTIONS_PATH), out, http_transport)
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"逐题明细：{out}")
