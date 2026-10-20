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

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ironm import __version__
from ironm.contracts.events import EVENT_ADAPTER, EnergyCheckEvent, Event
from ironm.core.clock import Clock, SystemClock, local_day_bounds
from ironm.core.config import Settings, load_settings
from ironm.core.log import configure_logging, get_logger
from ironm.engine.energy import EnergyAdjustment, energy_today
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
_STATUS_CODES = {"created": 201, "duplicate": 200, "conflict": 409}


def _invalid(errors: list[dict[str, Any]]) -> JSONResponse:
    return JSONResponse(status_code=422, content={"status": "invalid", "errors": errors})


def _errors(e: ValidationError) -> list[dict[str, Any]]:
    """校验错误：只给字段位置、原因与类型，不带取值。"""
    return [
        {"loc": _safe_loc(err["loc"]), "msg": err["msg"], "type": err["type"]}
        for err in e.errors(include_input=False, include_context=False, include_url=False)
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
        return await request.body()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """存活检查：只说明进程在运行，不检查依赖。"""
        return {"status": "ok", "version": __version__}

    @app.post("/events")
    def submit_event(
        body: Annotated[bytes, Depends(raw_body)],
        store: Annotated[EventStore, Depends(event_store)],
    ) -> JSONResponse:
        start = time.perf_counter()
        try:
            event = EVENT_ADAPTER.validate_json(body)
        except ValidationError as e:
            log.info("events.submitted", status="invalid", duration_ms=_ms(start))
            return _invalid(_errors(e))
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
        store: Annotated[EventStore, Depends(event_store)],
    ) -> JSONResponse:
        """离线队列上传：逐条处理、逐条返回；只有整批格式不对时才整批拒绝。"""
        start = time.perf_counter()
        try:
            doc = json.loads(body)
        except ValueError:
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
                parsed.append((i, EVENT_ADAPTER.validate_json(json.dumps(item))))
            except ValidationError as e:
                results[i] = {
                    "index": i,
                    "event_id": None,
                    "status": "invalid",
                    "errors": _errors(e),
                }
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
        user_id: str, tz: str, store: Annotated[EventStore, Depends(event_store)]
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
