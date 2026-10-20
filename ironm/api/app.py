"""FastAPI 应用入口。路由只做输入校验与调用，不放业务逻辑。"""

import os
import re
import sys
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import IO, Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ironm import __version__
from ironm.contracts.events import EVENT_ADAPTER
from ironm.core.clock import Clock, SystemClock
from ironm.core.config import Settings, load_settings
from ironm.core.log import configure_logging, get_logger
from ironm.rules.loader import load_rules
from ironm.store.db import connect
from ironm.store.events import (
    EventConflictError,
    EventStore,
    FutureEventError,
    InvalidCorrectionError,
)
from ironm.store.migrate import MIGRATIONS_DIR, migrate

log = get_logger("ironm.api")

# 错误位置里只保留合规的字段名；用户自造的键名可能就是健康数据原文
_SAFE_LOC_RE = re.compile(r"[A-Za-z0-9_]{1,64}")


def _safe_loc(loc: tuple[int | str, ...]) -> list[int | str]:
    return [p if isinstance(p, int) or _SAFE_LOC_RE.fullmatch(p) else "<field>" for p in loc]


def _invalid(errors: list[dict[str, Any]]) -> JSONResponse:
    return JSONResponse(status_code=422, content={"status": "invalid", "errors": errors})


def create_app(
    settings: Settings,
    clock: Clock,
    log_stream: IO[str] = sys.stderr,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> FastAPI:
    """配置或规则表有误时在这里直接失败；迁移在服务启动时执行，失败则启动失败。"""
    configure_logging(settings, clock, log_stream)
    load_rules(settings.rules_dir)

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

        def done(status: str, code: int, event_id: str | None = None, **extra: Any) -> JSONResponse:
            ms = round((time.perf_counter() - start) * 1000, 1)
            log.info("events.submitted", event_id=event_id, status=status, duration_ms=ms)
            content = {"event_id": event_id, "status": status, **extra}
            return JSONResponse(status_code=code, content=content)

        try:
            event = EVENT_ADAPTER.validate_json(body)
        except ValidationError as e:
            errors = [
                {"loc": _safe_loc(err["loc"]), "msg": err["msg"], "type": err["type"]}
                for err in e.errors(include_input=False, include_context=False, include_url=False)
            ]
            log.info("events.submitted", status="invalid", duration_ms=0)
            return _invalid(errors)

        event_id = str(event.event_id)
        try:
            created = store.append(event)
        except EventConflictError:
            return done("conflict", 409, event_id)
        except (FutureEventError, InvalidCorrectionError) as e:
            loc = ["recorded_at"] if isinstance(e, FutureEventError) else ["corrects"]
            log.info("events.submitted", event_id=event_id, status="invalid", duration_ms=0)
            return _invalid([{"loc": loc, "msg": str(e), "type": type(e).__name__}])
        return done("created", 201, event_id) if created else done("duplicate", 200, event_id)

    return app


app = create_app(load_settings(os.environ), SystemClock())
