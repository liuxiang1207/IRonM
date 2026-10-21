"""FastAPI 应用入口。路由只做输入校验与调用，不放业务逻辑。"""

import json
import os
import re
import sys
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, closing
from datetime import timedelta
from pathlib import Path
from typing import IO, Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ironm import __version__
from ironm.contracts.events import EVENT_ADAPTER, EnergyCheckEvent, Event, WeightEvent
from ironm.core.clock import Clock, SystemClock, local_day_bounds
from ironm.core.config import Settings, load_settings
from ironm.core.log import configure_logging, get_logger
from ironm.engine.energy import EnergyAdjustment, energy_today
from ironm.engine.metrics_view import build_metrics, lookback_days
from ironm.engine.rule_params import validate_rule_params
from ironm.knowledge.exercises import LIBRARY_PATH, load_library
from ironm.rules.loader import load_rules
from ironm.safety.decision import load_kit
from ironm.safety.emergency import TERMS_PATH
from ironm.safety.refusals import REFUSALS_PATH
from ironm.safety.replies import REPLIES_PATH
from ironm.skills.loader import SKILLS_DIR, load_skills, validate_skill_rules
from ironm.store.db import connect
from ironm.store.events import AppendResult, EventStore
from ironm.store.migrate import MIGRATIONS_DIR, migrate

log = get_logger("ironm.api")

# 错误位置里只保留合规的字段名；用户自造的键名可能就是健康数据原文
_SAFE_LOC_RE = re.compile(r"[A-Za-z0-9_]{1,64}")


def _safe_loc(loc: tuple[int | str, ...]) -> list[int | str]:
    return [p if isinstance(p, int) or _SAFE_LOC_RE.fullmatch(p) else "<field>" for p in loc]


MAX_BATCH = 500
_EARLIER_WEIGHTS = 10  # 回看范围之外再取几次称重：足以找到一次可信读数
MAX_BODY_BYTES = 2 * 1024 * 1024
_USER_RE = re.compile(r"[A-Za-z0-9_.:\-]{1,64}")
_NOT_OWNER = {"loc": ["user_id"], "msg": "记录不属于当前用户", "type": "forbidden_user"}
_STATUS_CODES = {"created": 201, "duplicate": 200, "conflict": 409}


def _invalid(errors: list[dict[str, Any]]) -> JSONResponse:
    return JSONResponse(status_code=422, content={"status": "invalid", "errors": errors})


# 校验错误的说明一律用自己的话，不用 pydantic 的原文：
# 原文可能带出提交的取值（类型名、标识里的字符等）。
# 限值只取自契约本身（ctx 中的 le / ge 等），不是用户输入。
_MESSAGES: dict[str, str] = {
    "missing": "缺少该字段",
    "extra_forbidden": "未定义的字段",
    "union_tag_invalid": "记录类型不在支持范围内",
    "union_tag_not_found": "缺少记录类型",
    "json_invalid": "请求体不是合法 JSON",
    "uuid_parsing": "不是合法的记录标识（UUID）",
    "uuid_type": "不是合法的记录标识（UUID）",
    "datetime_parsing": "不是合法的时间",
    "datetime_type": "不是合法的时间",
    "string_type": "应为文字",
    "int_type": "应为整数",
    "float_type": "应为数字",
    "bool_type": "应为真或假",
    "finite_number": "应为有限的数字",
    "literal_error": "取值不在允许范围内",
    "enum": "取值不在允许范围内",
    "string_too_short": "不能为空",
    "value_error": "取值不合规",
}
_LIMITS = {
    "less_than_equal": ("le", "应不大于 {}"),
    "less_than": ("lt", "应小于 {}"),
    "greater_than_equal": ("ge", "应不小于 {}"),
    "greater_than": ("gt", "应大于 {}"),
}


def _safe_msg(err: Any) -> str:
    kind = str(err["type"])
    if kind in _LIMITS:
        key, template = _LIMITS[kind]
        limit = err.get("ctx", {}).get(key)
        if isinstance(limit, int | float):
            return template.format(limit)
    return _MESSAGES.get(kind, f"取值不合规（{kind}）")


def _errors(e: ValidationError) -> list[dict[str, Any]]:
    """校验错误：只给字段位置、原因与类型，不带取值。"""
    return [
        {"loc": _safe_loc(err["loc"]), "msg": _safe_msg(err), "type": err["type"]}
        for err in e.errors(include_input=False, include_url=False)
    ]


_TZ_INVALID = {
    "loc": ["query", "tz"],
    "msg": "时区无效，应为标准名称（如 Asia/Shanghai）",
    "type": "timezone",
}


def _zone(name: str) -> ZoneInfo | None:
    """解析时区名；不合规（含过长、路径式输入）返回 None，由调用方回 422。"""
    if len(name) > 64:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, RecursionError):
        return None


def _reason_errors(result: AppendResult) -> list[dict[str, Any]]:
    return [{"loc": [result.field or ""], "msg": result.reason, "type": "invalid_event"}]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def create_app(
    settings: Settings,
    clock: Clock,
    log_stream: IO[str] = sys.stderr,
    migrations_dir: Path = MIGRATIONS_DIR,
    terms_path: Path = TERMS_PATH,
    refusals_path: Path = REFUSALS_PATH,
    replies_path: Path = REPLIES_PATH,
    library_path: Path = LIBRARY_PATH,
    skills_dir: Path = SKILLS_DIR,
) -> FastAPI:
    """配置、规则表、安全文件、动作库或 Skill 有误时在这里直接失败。

    迁移在服务启动时执行，失败则启动失败。
    """
    configure_logging(settings, clock, log_stream)
    rulebook = load_rules(settings.rules_dir)
    validate_rule_params(rulebook)
    # 急症词表、拒答词表与回答模板：启动时校验并互相核对（spec §21–24）
    load_kit(terms_path, refusals_path, replies_path)
    load_library(library_path)  # 动作库：启动时校验（spec §27）
    # Skill 头信息与它们用到的规则参数：启动时校验（spec §28、§30）
    validate_skill_rules(load_skills(skills_dir), rulebook)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        with closing(connect(settings.db_path)) as conn:
            migrate(conn, migrations_dir, clock)
        yield

    app = FastAPI(title="IronM", version=__version__, lifespan=lifespan)

    def event_store() -> Iterator[EventStore]:
        # 每个请求一个连接
        with closing(connect(settings.db_path)) as conn:
            yield EventStore(conn, clock)

    async def raw_body(request: Request) -> bytes:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="请求过大")  # 先看声明的大小，不先读进内存
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="请求过大")
        return body

    def current_user(x_ironm_user: Annotated[str | None, Header()] = None) -> str:
        """当前用户。登录功能完成前由请求头 X-IronM-User 给出；之后改为由登录身份决定。"""
        if x_ironm_user is None or not _USER_RE.fullmatch(x_ironm_user):
            raise HTTPException(status_code=401, detail="缺少当前用户")
        return x_ironm_user

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """存活检查：只说明进程在运行，不检查依赖。"""
        return {"status": "ok", "version": __version__}

    @app.post("/events")
    def submit_event(
        body: Annotated[bytes, Depends(raw_body)],
        user: Annotated[str, Depends(current_user)],
        store: Annotated[EventStore, Depends(event_store)],
    ) -> JSONResponse:
        start = time.perf_counter()
        try:
            event = EVENT_ADAPTER.validate_json(body)
        except ValidationError as e:
            log.info("events.submitted", status="invalid", duration_ms=_ms(start))
            return _invalid(_errors(e))
        if event.user_id != user:
            log.info("events.submitted", status="forbidden", duration_ms=_ms(start))
            return JSONResponse(
                status_code=403, content={"status": "forbidden", "errors": [_NOT_OWNER]}
            )
        [result] = store.append_many([event])
        log.info(
            "events.submitted",
            event_id=str(result.event_id),
            status=result.status,
            duration_ms=_ms(start),
        )
        if result.status == "invalid":
            return _invalid(_reason_errors(result))
        content = {"event_id": str(result.event_id), "status": result.status}
        return JSONResponse(status_code=_STATUS_CODES[result.status], content=content)

    @app.post("/events/batch")
    def submit_batch(
        body: Annotated[bytes, Depends(raw_body)],
        user: Annotated[str, Depends(current_user)],
        store: Annotated[EventStore, Depends(event_store)],
    ) -> JSONResponse:
        """离线队列上传：逐条处理、逐条返回；只有整批格式不对时才整批拒绝。"""
        start = time.perf_counter()
        try:
            doc = json.loads(body)
        except (ValueError, RecursionError):
            return _invalid([{"loc": [], "msg": "请求体不是合法 JSON", "type": "json_invalid"}])
        if (
            not isinstance(doc, dict)
            or set(doc) != {"events"}
            or not isinstance(doc["events"], list)
        ):
            msg = '请求体必须是 {"events": [...]}'
            return _invalid([{"loc": ["events"], "msg": msg, "type": "batch_shape"}])
        items: list[object] = doc["events"]
        if not 1 <= len(items) <= MAX_BATCH:
            msg = f"每批 1–{MAX_BATCH} 条，实际 {len(items)} 条"
            return _invalid([{"loc": ["events"], "msg": msg, "type": "batch_size"}])

        results: dict[int, dict[str, object]] = {}
        parsed: list[tuple[int, Event]] = []
        for i, item in enumerate(items):
            try:
                event = EVENT_ADAPTER.validate_json(json.dumps(item))
            except ValidationError as e:
                results[i] = {
                    "index": i,
                    "event_id": None,
                    "status": "invalid",
                    "errors": _errors(e),
                }
                continue
            if event.user_id != user:
                results[i] = {
                    "index": i,
                    "event_id": str(event.event_id),
                    "status": "invalid",
                    "errors": [_NOT_OWNER],
                }
                continue
            parsed.append((i, event))
        outcomes = store.append_many([e for _, e in parsed])
        for (i, _), res in zip(parsed, outcomes, strict=True):
            results[i] = {"index": i, "event_id": str(res.event_id), "status": res.status}
            if res.status == "invalid":
                results[i]["errors"] = _reason_errors(res)
        ordered = [results[i] for i in range(len(items))]
        for r in ordered:
            log.info("events.submitted", event_id=r["event_id"], status=r["status"])  # type: ignore[arg-type]
        log.info("events.batch_submitted", count=len(items), duration_ms=_ms(start))
        return JSONResponse(status_code=200, content={"results": ordered})

    @app.get("/energy/today")
    def energy_today_view(
        tz: str,
        user_id: Annotated[str, Depends(current_user)],
        store: Annotated[EventStore, Depends(event_store)],
    ) -> JSONResponse:
        """用户所在时区今天的精力选择与训练量调整；未选择时 selected 为 false，不给调整。"""
        zone = _zone(tz)
        if zone is None:
            return _invalid([_TZ_INVALID])
        today = clock.now().astimezone(zone).date()
        start, end = local_day_bounds(today, zone)
        checks = [
            e
            for e in store.list_for_user(user_id, types={"energy_check"}, since=start, until=end)
            if isinstance(e, EnergyCheckEvent)
        ]
        result = energy_today(checks, day=today, tz=zone, rulebook=rulebook)
        if not result.selected or not isinstance(result.adjustment, EnergyAdjustment):
            return JSONResponse(content={"selected": False})
        content = {
            "selected": True,
            "level": result.level,
            "adjustment": result.adjustment.model_dump(),
        }
        return JSONResponse(content=content)

    @app.get("/metrics")
    def metrics_view(
        tz: str,
        user_id: Annotated[str, Depends(current_user)],
        store: Annotated[EventStore, Depends(event_store)],
    ) -> JSONResponse:
        """当前用户的指标：四个领域的精度与指标结果列表。只读取计算所需的最近一段记录。"""
        zone = _zone(tz)
        if zone is None:
            return _invalid([_TZ_INVALID])
        as_of = clock.now()
        since = as_of - timedelta(days=lookback_days(rulebook))
        events = store.list_for_user(user_id, since=since)
        # 更正可能把发生时间改到范围之外，仍要一并取得，否则原记录会被当作有效
        have = {e.event_id for e in events}
        events += [e for e in store.corrections_of(user_id, have) if e.event_id not in have]
        # 范围内没有称重时，用范围之外最近的几次区分“已过期”与“没有数据”
        earlier = [
            e
            for e in store.latest_before(user_id, "weight", since, _EARLIER_WEIGHTS)
            if isinstance(e, WeightEvent)
        ]
        report = build_metrics(
            events, as_of=as_of, tz=zone, rulebook=rulebook, earlier_weights=earlier
        )
        return JSONResponse(content=report.model_dump(mode="json"))

    return app


app = create_app(load_settings(os.environ), SystemClock())
