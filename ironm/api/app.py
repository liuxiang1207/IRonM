"""FastAPI 应用入口。路由只做输入校验与调用，不放业务逻辑。"""

import json
import os
import re
import sqlite3
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, closing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import IO, Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ironm import __version__
from ironm.contracts.events import EVENT_ADAPTER, EnergyCheckEvent, Event, WeightEvent
from ironm.contracts.memory import Candidate
from ironm.contracts.model_call import ModelStage
from ironm.contracts.screening import SafetyFlag
from ironm.core.clock import Clock, SystemClock, local_day_bounds
from ironm.core.config import Settings, load_settings
from ironm.core.ids import IdGenerator, RandomIdGenerator
from ironm.core.log import configure_logging, get_logger
from ironm.engine.energy import EnergyAdjustment, energy_today
from ironm.engine.metrics_view import build_metrics, lookback_days
from ironm.engine.review_facts import RETURN_RULE, ReturnParams, return_state, training_days
from ironm.engine.rule_params import validate_rule_params
from ironm.knowledge.baseline import BASELINE_PATH, load_baseline
from ironm.knowledge.exercises import LIBRARY_PATH, load_library
from ironm.knowledge.kb import KnowledgeBase, load_passages, retriever_from_rules
from ironm.knowledge.qa_cards import load_cards
from ironm.llm.config import load_models
from ironm.llm.gateway import Gateway, GatewayEmbedder
from ironm.memory.conversation import MEMORY_RULE, MemoryParams
from ironm.memory.profile import ProfileService
from ironm.memory.profile import load_terms as load_profile_terms
from ironm.orchestrator.chat import (
    MAX_TEXT_CHARS,
    ChatDeps,
    ChatReply,
    after_reply,
    load_answer_texts,
    respond,
)
from ironm.orchestrator.domains import DOMAIN_RULE, DomainParams
from ironm.orchestrator.intent import INTENT_RULE, IntentParams, load_intent_book
from ironm.orchestrator.jobs import JobContext, plan_jobs, run_job
from ironm.orchestrator.onboarding import Intake, history_view, onboard, plan_view, today_tasks
from ironm.orchestrator.pages import PAGE_INTENTS, PageContext
from ironm.orchestrator.qa import RunRecord
from ironm.rules.loader import load_rules
from ironm.safety.decision import load_kit
from ironm.safety.emergency import TERMS_PATH
from ironm.safety.refusals import REFUSALS_PATH
from ironm.safety.replies import REPLIES_PATH
from ironm.scheduler.runner import Scheduler, TickResult
from ironm.skills.loader import SKILLS_DIR, load_skills, validate_skill_rules
from ironm.store.auth import AuthStore
from ironm.store.db import connect
from ironm.store.events import AppendResult, EventStore
from ironm.store.jobs import JobStore
from ironm.store.memory import MemoryStore
from ironm.store.migrate import MIGRATIONS_DIR, migrate
from ironm.store.plans import PlanStore, VersionConflict
from ironm.store.reviews import DecisionStore
from ironm.store.safety import ModelCallStore, RunStore, SafetyFlagStore

log = get_logger("ironm.api")

# 网页版（spec [2026-10-21] §19）：静态页面，只调用下面这些接口
WEB_DIR = Path(__file__).parent / "web"

# 错误位置里只保留合规的字段名；用户自造的键名可能就是健康数据原文
_SAFE_LOC_RE = re.compile(r"[A-Za-z0-9_]{1,64}")


def _safe_loc(loc: tuple[int | str, ...]) -> list[int | str]:
    return [p if isinstance(p, int) or _SAFE_LOC_RE.fullmatch(p) else "<field>" for p in loc]


MAX_BATCH = 500
_EARLIER_WEIGHTS = 10  # 回看范围之外再取几次称重：足以找到一次可信读数
MAX_BODY_BYTES = 2 * 1024 * 1024
_USER_RE = re.compile(r"[A-Za-z0-9_.:\-]{1,64}")


class LoginRequest(BaseModel):
    """登录：设备码与设备名（如“我的 iPhone”）。"""

    model_config = ConfigDict(extra="forbid")

    code: Annotated[str, Field(min_length=1, max_length=32)]
    device: Annotated[str, Field(min_length=1, max_length=40)]


class _LoginGuard:
    """10 分钟内错 10 次设备码，接下来 10 分钟拒绝任何登录（spec [2026-10-21] §14）。"""

    WINDOW = timedelta(minutes=10)
    LIMIT = 10

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._fails: list[datetime] = []
        self._until: datetime | None = None
        self._lock = threading.Lock()

    def locked(self) -> bool:
        with self._lock:
            return self._until is not None and self._clock.now() < self._until

    def fail(self) -> None:
        with self._lock:
            now = self._clock.now()
            self._fails = [t for t in self._fails if now - t < self.WINDOW] + [now]
            if len(self._fails) >= self.LIMIT:
                self._until, self._fails = now + self.WINDOW, []


class ModeRequest(BaseModel):
    """出差 / 节假日 / 恢复正常；出差与节假日要给结束日期（最多 60 天）。"""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["travel", "holiday", "normal"]
    until: date | None = None

    @model_validator(mode="after")
    def _check(self) -> "ModeRequest":
        if self.mode != "normal" and self.until is None:
            raise ValueError("出差与节假日要给结束日期")
        return self


def _midnight_utc(day: date, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, datetime.min.time(), tzinfo=tz)


class ChatRequest(BaseModel):
    """对话请求：用户的一句话（去掉首尾空白后 1–2000 个字）。"""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    context: PageContext | None = None  # 从功能页带入的页面与条目标识


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
    baseline_path: Path = BASELINE_PATH,
    skills_dir: Path = SKILLS_DIR,
    ids: IdGenerator | None = None,
    environ: Mapping[str, str] | None = None,
) -> FastAPI:
    """配置、规则表、安全文件、动作库或 Skill 有误时在这里直接失败。

    迁移在服务启动时执行，失败则启动失败。
    """
    configure_logging(settings, clock, log_stream)
    rulebook = load_rules(settings.rules_dir)
    validate_rule_params(rulebook)
    # 急症词表、拒答词表与回答模板：启动时校验并互相核对（spec §21–24）
    kit = load_kit(terms_path, refusals_path, replies_path)
    id_gen = ids or RandomIdGenerator()
    # 问答链路（spec [2026-10-21] §2–§4）：模型配置、意图词表与参数、固定说法，启动时校验
    models = load_models(settings.models_path)
    intents = load_intent_book()
    rulebook.get(INTENT_RULE).params_as(IntentParams)
    rulebook.get(DOMAIN_RULE).params_as(DomainParams)
    texts = load_answer_texts()
    cards = load_cards()  # 问答卡（spec [2026-10-21] §6）
    profile_terms = load_profile_terms()  # 画像候选的说法（§8）
    rulebook.get(MEMORY_RULE).params_as(MemoryParams)
    env = os.environ if environ is None else environ
    library = load_library(library_path)  # 动作库：启动时校验（spec §27）
    rulebook.get(RETURN_RULE).params_as(ReturnParams)  # 中断回归参数：启动时校验（spec §17）
    # 知识库与检索（spec [2026-10-21] §12）：问答卡 + 动作库 + 资料目录的段落，启动时建索引
    # 向量一路（spec §13）：配置了向量服务的密钥才启用；段落向量第一次检索时计算，调用只记日志
    vectors = bool(env.get(models.routes[ModelStage.EMBEDDING].api_key_env))
    index_gateway = Gateway(models, environ=env, clock=clock, ids=id_gen)
    retriever = retriever_from_rules(
        KnowledgeBase.from_sources(cards, library, load_passages(settings.kb_dir)),
        rulebook,
        embedder=GatewayEmbedder(index_gateway, "kb-index") if vectors else None,
    )
    load_baseline(baseline_path)  # 基线测试方案：启动时校验（spec §31）
    # Skill 头信息与它们用到的规则参数：启动时校验（spec §28、§30）
    skills = load_skills(skills_dir)
    validate_skill_rules(skills, rulebook)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        with closing(connect(settings.db_path)) as conn:
            migrate(conn, migrations_dir, clock)
            JobStore(conn).reset_running(clock.now())  # 上次没执行完的任务重新排队
        if vectors:
            # 启动后在后台预先计算知识段落的向量：不拖慢启动，第一个提问也不用等（spec §13）
            threading.Thread(target=retriever.warm, name="kb-warm", daemon=True).start()
        stop = threading.Event()
        if settings.scheduler:
            threading.Thread(target=_loop, args=(stop,), name="scheduler", daemon=True).start()
        yield
        stop.set()

    jobs_ctx = JobContext(rulebook=rulebook, skills=skills, ids=id_gen)

    def tick() -> TickResult:
        """执行一轮定时任务（spec [2026-10-21] §16）：登记到期任务，按到期时间先后执行。"""
        with closing(connect(settings.db_path)) as conn:
            scheduler = Scheduler(
                repo=JobStore(conn),
                plan=lambda now: plan_jobs(conn, now, jobs_ctx),
                run=lambda job, now: run_job(conn, job, now, jobs_ctx),
                clock=clock,
            )
            return scheduler.tick()

    def _loop(stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                tick()
            except Exception as e:  # 一轮出错不停止后台循环
                log.error("scheduler.tick_failed", status="error", error_type=type(e).__name__)
            stop.wait(settings.scheduler_interval_s)

    app = FastAPI(title="IronM", version=__version__, lifespan=lifespan)
    app.state.tick = tick  # 管理员命令与测试可直接执行一轮定时任务

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

    def current_user(
        authorization: Annotated[str | None, Header()] = None,
        x_ironm_user: Annotated[str | None, Header()] = None,
    ) -> str:
        """当前用户（spec [2026-10-21] §14）：带登录凭证时按凭证认人；正式环境只认凭证，
        开发与测试环境也可用请求头 X-IronM-User 直接指定。"""
        if authorization is not None:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                with closing(connect(settings.db_path)) as conn:
                    found = AuthStore(conn, clock).user_for(token.strip())
                if found is not None:
                    return found[0]
            raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
        if settings.env != "prod" and x_ironm_user and _USER_RE.fullmatch(x_ironm_user):
            return x_ironm_user
        raise HTTPException(status_code=401, detail="请先登录")

    guard = _LoginGuard(clock)

    @app.post("/auth/login")
    def login(body: Annotated[bytes, Depends(raw_body)]) -> JSONResponse:
        """设备码换这台设备的登录凭证（spec §14）。"""
        try:
            req = LoginRequest.model_validate_json(body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail="请填写设备码与设备名") from e
        if guard.locked():
            log.info("auth.login", status="locked")
            raise HTTPException(status_code=429, detail="错误次数太多，请 10 分钟后再试")
        with closing(connect(settings.db_path)) as conn:
            got = AuthStore(conn, clock).login(req.code, req.device)
        if got is None:
            guard.fail()
            log.info("auth.login", status="rejected")
            raise HTTPException(status_code=401, detail="设备码无效、已用过或已过期")
        user_id, token = got
        log.info("auth.login", status="ok", user_id=user_id)
        return JSONResponse({"user_id": user_id, "token": token})

    @app.get("/auth/devices")
    def list_devices(user: Annotated[str, Depends(current_user)]) -> JSONResponse:
        with closing(connect(settings.db_path)) as conn:
            devices = AuthStore(conn, clock).devices(user)
        return JSONResponse(
            [
                {"device_id": d.device_id, "name": d.name, "created_at": d.created_at.isoformat()}
                for d in devices
            ]
        )

    @app.delete("/auth/devices/{device_id}", status_code=204)
    def revoke_device(device_id: str, user: Annotated[str, Depends(current_user)]) -> Response:
        """撤销自己的一台设备；不是自己的设备一律当作不存在。"""
        ok = False
        if _USER_RE.fullmatch(device_id):
            with closing(connect(settings.db_path)) as conn:
                ok = AuthStore(conn, clock).revoke(user, device_id)
        if not ok:
            raise HTTPException(status_code=404, detail="没有这台设备")
        return Response(status_code=204)

    def chat_deps(conn: sqlite3.Connection) -> ChatDeps:
        """一次请求（或回答后的后台任务）用到的依赖；都用同一个连接。"""
        flags = SafetyFlagStore(conn, id_gen)
        calls = ModelCallStore(conn)
        runs = RunStore(conn)
        memory = MemoryStore(conn)

        def record_flag(u: str, req: str, flag: SafetyFlag) -> None:
            flags.add(u, req, flag)

        def record_run(u: str, record: RunRecord) -> None:
            runs.add(u, record.request_id, record.model_dump_json())

        return ChatDeps(
            kit=kit,
            clock=clock,
            ids=id_gen,
            record_flag=record_flag,
            record_run=record_run,
            intents=intents,
            cards=cards,
            rulebook=rulebook,
            texts=texts,
            memory=memory,
            profiles=ProfileService(memory, terms=profile_terms, ids=id_gen, now=clock.now),
            gateway=Gateway(models, environ=env, clock=clock, ids=id_gen, record=calls.add),
            retriever=retriever,
            vectors=vectors,
        )

    def save_turn(user: str, text: str, reply: ChatReply) -> None:
        """回答之后在后台保存这一轮对话并按需压缩（spec [2026-10-21] §8）。"""
        with closing(connect(settings.db_path)) as conn:
            after_reply(user, text, reply, chat_deps(conn), now=clock.now)

    @app.post("/chat")
    def chat(
        body: Annotated[bytes, Depends(raw_body)],
        user: Annotated[str, Depends(current_user)],
        background: BackgroundTasks,
    ) -> JSONResponse:
        """对话（spec [2026-10-21] §3–§8）：先过急症与拒答检查，再识别意图、编排回答。"""
        try:
            request = ChatRequest.model_validate_json(body)
        except ValidationError as e:
            log.info("chat.submitted", status="invalid")
            return _invalid(_errors(e))
        with closing(connect(settings.db_path)) as conn:
            reply = respond(user, request.text, chat_deps(conn), page_context=request.context)
        background.add_task(save_turn, user, request.text, reply)
        return JSONResponse(content=reply.model_dump(mode="json"))

    @app.post("/pages/{page}/ask")
    def page_ask(
        page: str,
        body: Annotated[bytes, Depends(raw_body)],
        user: Annotated[str, Depends(current_user)],
        background: BackgroundTasks,
    ) -> JSONResponse:
        """功能页提问（spec [2026-10-21] §3、§10）。

        与管家对话用同一套安全前置；只调用本功能的专业 Agent 与安全官。
        """
        if page not in PAGE_INTENTS:
            return JSONResponse(status_code=404, content={"status": "not_found"})
        try:
            request = ChatRequest.model_validate_json(body)
        except ValidationError as e:
            log.info("pages.asked", status="invalid")
            return _invalid(_errors(e))
        if request.context is not None and request.context.page != page:
            return JSONResponse(status_code=422, content={"status": "invalid_context"})
        with closing(connect(settings.db_path)) as conn:
            reply = respond(
                user,
                request.text,
                chat_deps(conn),
                page=page,
                page_context=request.context,
            )
        background.add_task(save_turn, user, request.text, reply)
        return JSONResponse(content=reply.model_dump(mode="json"))

    def candidate_json(c: Candidate) -> dict[str, Any]:
        return {
            "candidate_id": c.candidate_id,
            "field": c.field,
            "value": c.value,
            "immediate": c.immediate,
            "status": c.status,
        }

    @app.get("/profile")
    def get_profile(user: Annotated[str, Depends(current_user)]) -> JSONResponse:
        with closing(connect(settings.db_path)) as conn:
            version, data = MemoryStore(conn).profile(user)
        return JSONResponse(content={"version": version, "data": data})

    @app.get("/profile/candidates")
    def get_candidates(user: Annotated[str, Depends(current_user)]) -> JSONResponse:
        """待确认的画像候选（spec [2026-10-21] §8）。"""
        with closing(connect(settings.db_path)) as conn:
            pending = MemoryStore(conn).candidates(user, "pending")
        return JSONResponse(content={"candidates": [candidate_json(c) for c in pending]})

    @app.post("/profile/candidates/{candidate_id}/{decision}")
    def decide_candidate(
        candidate_id: str, decision: str, user: Annotated[str, Depends(current_user)]
    ) -> JSONResponse:
        """确认（confirm）或否认（reject）一个候选；已生效的健康状况被否认时撤回。"""
        if decision not in ("confirm", "reject"):
            return JSONResponse(status_code=404, content={"status": "not_found"})
        with closing(connect(settings.db_path)) as conn:
            memory = MemoryStore(conn)
            service = ProfileService(memory, terms=profile_terms, ids=id_gen, now=clock.now)
            decided = service.decide(user, candidate_id, decision == "confirm")
        if decided is None:
            return JSONResponse(status_code=404, content={"status": "not_found"})
        return JSONResponse(content=candidate_json(decided))

    @app.get("/chat/runs/{request_id}")
    def chat_run(request_id: str, user: Annotated[str, Depends(current_user)]) -> JSONResponse:
        """按请求编号回放问答运行记录（spec [2026-10-21] §5）；只能看自己的。"""
        with closing(connect(settings.db_path)) as conn:
            found = RunStore(conn).get(request_id)
        if found is None or found[0] != user:
            return JSONResponse(status_code=404, content={"status": "not_found"})
        record = RunRecord.model_validate_json(found[1])
        return JSONResponse(content=record.model_dump(mode="json"))

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        """网站首页：进入网页版（spec §19）。"""
        return RedirectResponse("/web/")

    @app.get("/auth/options")
    def auth_options() -> JSONResponse:
        """网页登录页可用的方式：开发与测试环境另可直接填用户名进入（spec §19）。"""
        return JSONResponse({"dev_user_login": settings.env != "prod"})

    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")

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

    @app.post("/onboarding")
    def onboarding(
        body: Annotated[bytes, Depends(raw_body)],
        user: Annotated[str, Depends(current_user)],
        tz: str = "Asia/Shanghai",
    ) -> JSONResponse:
        """建档并生成第一版计划（spec [2026-10-21] §15）：计划、需要补充的信息或不能给计划。"""
        zone = _zone(tz)
        if zone is None:
            return _invalid([_TZ_INVALID])
        try:
            raw = json.loads(body)
            if not isinstance(raw, dict):
                raise ValueError("请求体应是对象")
            intake = Intake.from_request(raw)
        except ValidationError as e:
            return _invalid(_errors(e))
        except ValueError:
            return _invalid([{"loc": ["body"], "msg": "资料格式不对", "type": "invalid_body"}])
        with closing(connect(settings.db_path)) as conn:
            try:
                result = onboard(
                    user,
                    intake,
                    memory=MemoryStore(conn),
                    plans=PlanStore(conn),
                    flags=SafetyFlagStore(conn, id_gen).for_user(user),
                    rulebook=rulebook,
                    skills=skills,
                    as_of=clock.now(),
                    tz=zone,
                )
            except VersionConflict:
                log.info("plan.onboarding", status="conflict")
                return JSONResponse(status_code=409, content={"status": "conflict"})
        log.info("plan.onboarding", status=result["status"])
        return JSONResponse(content=result)

    @app.get("/plan")
    def get_plan(user: Annotated[str, Depends(current_user)]) -> JSONResponse:
        with closing(connect(settings.db_path)) as conn:
            stored = PlanStore(conn).latest(user)
        if stored is None:
            return JSONResponse(content={"status": "none"})
        return JSONResponse(content={"status": "ok", "plan": plan_view(stored)})

    @app.get("/plan/history")
    def get_plan_history(user: Annotated[str, Depends(current_user)]) -> JSONResponse:
        with closing(connect(settings.db_path)) as conn:
            stored = PlanStore(conn).history(user)
        return JSONResponse(content=history_view(stored))

    @app.get("/plan/today")
    def get_plan_today(tz: str, user: Annotated[str, Depends(current_user)]) -> JSONResponse:
        """今日任务：按当前计划与用户时区的当天日期（spec §15）。"""
        zone = _zone(tz)
        if zone is None:
            return _invalid([_TZ_INVALID])
        today = clock.now().astimezone(zone).date()
        since = _midnight_utc(today - timedelta(days=90), zone)
        with closing(connect(settings.db_path)) as conn:
            stored = PlanStore(conn).latest(user)
            profile = MemoryStore(conn).profile(user)[1]
            events = EventStore(conn, clock).list_for_user(
                user, types={"workout_check", "set_log", "checkin"}, since=since
            )
        p = rulebook.get(RETURN_RULE).params_as(ReturnParams)
        back = return_state(
            training_days(events, zone),
            today,
            gap_days=p.gap_days,
            rebaseline_after_days=p.rebaseline_after_days,
            halve_for_days=p.halve_for_days,
        )
        content = today_tasks(
            stored, today, library=library, back=back, mode=profile.get("plan_mode")
        )
        return JSONResponse(content=content)

    @app.get("/plan/decisions")
    def get_plan_decisions(
        user: Annotated[str, Depends(current_user)], limit: int = 20
    ) -> JSONResponse:
        """复盘与复核的决策日志（给用户看的版本：改了什么、为什么、什么时候再看，spec §17）。"""
        with closing(connect(settings.db_path)) as conn:
            logs = DecisionStore(conn).recent(user, limit=max(1, min(limit, 100)))
        return JSONResponse(
            content=[
                {
                    "kind": d.kind,
                    "period": d.period,
                    "at": d.at.isoformat(),
                    "decision": d.record.get("decision"),
                    "summary": d.record.get("summary"),
                    "suggestion": d.record.get("suggestion"),
                    "changes": d.record.get("changes", []),
                }
                for d in logs
            ]
        )

    @app.post("/plan/mode")
    def set_plan_mode(
        body: Annotated[bytes, Depends(raw_body)], user: Annotated[str, Depends(current_user)]
    ) -> JSONResponse:
        """出差 / 节假日（spec §17）：到结束日后自动恢复；选“正常”立即恢复。"""
        try:
            req = ModeRequest.model_validate_json(body)
        except ValidationError as e:
            return _invalid(_errors(e))
        with closing(connect(settings.db_path)) as conn:
            memory = MemoryStore(conn)
            version, profile = memory.profile(user)
            mode = (
                {"mode": req.mode, "until": req.until.isoformat()}
                if req.mode != "normal" and req.until is not None
                else None
            )
            updated = {**profile, "plan_mode": mode}
            memory.save_profile(user, version, updated, "plan_mode", clock.now())
        log.info("plan.mode", status=req.mode)
        return JSONResponse(content={"mode": mode})

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
