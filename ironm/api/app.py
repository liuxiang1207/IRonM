"""FastAPI 应用入口。路由只做输入校验与调用，不放业务逻辑。"""

import json
import os
import re
import sys
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import IO, Annotated, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ironm import __version__
from ironm.contracts.events import EVENT_ADAPTER, EnergyCheckEvent, Event
from ironm.core.clock import Clock, SystemClock, local_day_bounds
from ironm.core.config import Settings, load_settings
from ironm.core.log import configure_logging, get_logger
from ironm.engine.energy import EnergyAdjustment, energy_today
from ironm.engine.rule_params import validate_rule_params
from ironm.rules.loader import load_rules
from ironm.store.db import connect
from ironm.store.events import AppendResult, EventStore
from ironm.store.migrate import MIGRATIONS_DIR, migrate

log = get_logger("ironm.api")

# 错误位置里只保留合规的字段名；用户自造的键名可能就是健康数据原文
_SAFE_LOC_RE = re.compile(r"[A-Za-z0-9_]{1,64}")


def _safe_loc(loc: tuple[int | str, ...]) -> list[int | str]:
    return [p if isinstance(p, int) or _SAFE_LOC_RE.fullmatch(p) else "<field>" for p in loc]


MAX_BATCH = 500
MAX_BODY_BYTES = 2 * 1024 * 1024
_USER_RE = re.compile(r"[A-Za-z0-9_.:\-]{1,64}")
_NOT_OWNER = {"loc": ["user_id"], "msg": "记录不属于当前用户", "type": "forbidden_user"}
_STATUS_CODES = {"created": 201, "duplicate": 200, "conflict": 409}


def _invalid(errors: list[dict[str, Any]]) -> JSONResponse:
    return JSONResponse(status_code=422, content={"status": "invalid", "errors": errors})


def _safe_msg(err: Any) -> str:
    """原因说明里若带出了提交的取值（如类型名写成一句话、标识里的字符），改用通用说明。"""
    msg = str(err["msg"])
    value = err.get("input")
    echoed = isinstance(value, str) and value != "" and value in msg
    if echoed or not msg.isascii():
        return f"取值不合规（{err['type']}）"
    return msg


def _errors(e: ValidationError) -> list[dict[str, Any]]:
    """校验错误：只给字段位置、原因与类型，不带取值。"""
    return [
        {"loc": _safe_loc(err["loc"]), "msg": _safe_msg(err), "type": err["type"]}
        for err in e.errors(include_context=False, include_url=False)
    ]


def _reason_errors(result: AppendResult) -> list[dict[str, Any]]:
    return [{"loc": [result.field or ""], "msg": result.reason, "type": "invalid_event"}]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def create_app(
    settings: Settings,
    clock: Clock,
    log_stream: IO[str] = sys.stderr,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> FastAPI:
    """配置或规则表有误时在这里直接失败；迁移在服务启动时执行，失败则启动失败。"""
    configure_logging(settings, clock, log_stream)
    rulebook = load_rules(settings.rules_dir)
    validate_rule_params(rulebook)

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
        try:
            zone = ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError):
            msg = "时区无效，应为标准名称（如 Asia/Shanghai）"
            return _invalid([{"loc": ["query", "tz"], "msg": msg, "type": "timezone"}])
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

    return app


app = create_app(load_settings(os.environ), SystemClock())
