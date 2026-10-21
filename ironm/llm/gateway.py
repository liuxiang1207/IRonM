"""模型网关（spec [2026-10-21] §2）：兼容常见对话接口的模型服务。

- 每个环节按配置选模型；密钥只从环境变量读取，缺失时不发请求、记为出错。
- 超时或出错时最多重试 1 次；输出没通过校验（含被截断）不重试，直接交给调用方兜底。
- 每次调用留一条 ModelCall 记录（只有元数据），并写一行日志；都不含提示词、回答原文与密钥。
- 网关本身从不抛出调用异常：失败时返回的 value 为 None，由调用方使用兜底。
"""

import functools
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ironm.contracts.model_call import ModelCall, ModelStage
from ironm.core.clock import Clock
from ironm.core.ids import IdGenerator
from ironm.core.log import get_logger
from ironm.llm.config import ModelsConfig
from ironm.llm.validate import OutputRejected

MAX_ATTEMPTS = 2  # 首次 + 重试 1 次
EMBED_BATCH = 32  # 向量接口每批的文本数
EMBED_TEMPLATE = "embedding@1.0.0"  # 向量调用没有提示词，记录里用固定的名称

_log = get_logger("ironm.llm")


class TransportTimeout(Exception):
    pass


class MissingApiKey(Exception):
    pass


class Transport(Protocol):
    def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
    ) -> dict[str, object]: ...


def http_transport(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
) -> dict[str, object]:
    data_bytes = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
    except TimeoutError as e:
        raise TransportTimeout from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise TransportTimeout from e
        raise
    if not isinstance(data, dict):
        raise ValueError("响应不是 JSON 对象")
    return data


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Completion[T]:
    """value 为 None 表示失败，调用方应使用兜底。calls 是本次的全部调用记录。"""

    value: T | None
    calls: tuple[ModelCall, ...]


def _content(resp: Mapping[str, object]) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise OutputRejected("响应缺少 choices")
    first = choices[0]
    if first.get("finish_reason") == "length":
        raise OutputRejected("输出被截断")
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OutputRejected("输出为空")
    return content


def _vectors(resp: Mapping[str, object], expected: int) -> list[list[float]]:
    """embeddings 响应：data[i].embedding，按 index 排序；条数或格式不对即拒收。"""
    data = resp.get("data")
    if not isinstance(data, list) or len(data) != expected:
        raise OutputRejected("向量条数不对")
    rows: list[tuple[int, list[float]]] = []
    for n, item in enumerate(data):
        vec = item.get("embedding") if isinstance(item, dict) else None
        index = item.get("index", n) if isinstance(item, dict) else n
        if not isinstance(vec, list) or not vec or not all(isinstance(x, int | float) for x in vec):
            raise OutputRejected("向量格式不对")
        rows.append((index if isinstance(index, int) else n, [float(x) for x in vec]))
    return [v for _, v in sorted(rows, key=lambda r: r[0])]


class GatewayEmbedder:
    """把网关的向量接口包成检索用的向量模型；失败时抛错，由检索跳过向量这一路。"""

    def __init__(self, gateway: "Gateway", request_id: str) -> None:
        self._gateway, self._request_id = gateway, request_id

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        result = self._gateway.embed(texts, request_id=self._request_id)
        if result.value is None:
            raise EmbeddingUnavailable
        return result.value


class EmbeddingUnavailable(Exception):
    pass


def _tokens(resp: Mapping[str, object]) -> dict[str, int]:
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        return {}
    out: dict[str, int] = {}
    for src, dst in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
        value = usage.get(src)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            out[dst] = value
    return out


class Gateway:
    """可被多个线程同时使用（问答时各 Agent 并行调用模型）。调用记录逐条写入、不交叠：
    记录通常写进同一个数据库连接，交叠的事务会出错。"""

    def __init__(
        self,
        config: ModelsConfig,
        *,
        environ: Mapping[str, str],
        clock: Clock,
        ids: IdGenerator,
        transport: Transport = http_transport,
        record: Callable[[ModelCall], None] = lambda _call: None,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._environ = environ
        self._clock = clock
        self._ids = ids
        self._transport = transport
        self._record = record
        self._record_lock = threading.Lock()
        self._timer = timer

    def complete[T](
        self,
        stage: ModelStage,
        *,
        request_id: str,
        template: str,
        messages: Sequence[Message],
        validate: Callable[[str], T],
    ) -> Completion[T]:
        route = self._config.routes[stage]
        body: dict[str, Any] = {
            "model": route.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": route.max_tokens,
            **route.extra_body,
        }
        return self._run(
            stage,
            request_id=request_id,
            template=template,
            path="/chat/completions",
            body=body,
            parse=lambda resp: validate(_content(resp)),
        )

    def embed(
        self, texts: Sequence[str], *, request_id: str, stage: ModelStage = ModelStage.EMBEDDING
    ) -> Completion[list[list[float]]]:
        """向量（spec [2026-10-21] §13）：按通用的 embeddings 接口分批请求；一批失败即整体失败。"""
        route = self._config.routes[stage]
        vectors: list[list[float]] = []
        calls: list[ModelCall] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = list(texts[start : start + EMBED_BATCH])
            done = self._run(
                stage,
                request_id=request_id,
                template=EMBED_TEMPLATE,
                path="/embeddings",
                body={"model": route.model, "input": batch},
                parse=functools.partial(_vectors, expected=len(batch)),
            )
            calls += done.calls
            if done.value is None:
                return Completion(value=None, calls=tuple(calls))
            vectors += done.value
        return Completion(value=vectors, calls=tuple(calls))

    def _run[T](
        self,
        stage: ModelStage,
        *,
        request_id: str,
        template: str,
        path: str,
        body: dict[str, Any],
        parse: Callable[[Mapping[str, object]], T],
    ) -> Completion[T]:
        route = self._config.routes[stage]
        calls: list[ModelCall] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started_at = self._clock.now()
            start = self._timer()
            tokens: dict[str, int] = {}
            value: T | None = None
            error: BaseException | None = None
            try:
                key = self._environ.get(route.api_key_env)
                if not key:
                    raise MissingApiKey
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                resp = self._transport(f"{route.base_url}{path}", headers, body, route.timeout_s)
                tokens = _tokens(resp)
                value = parse(resp)
            except Exception as e:  # 任何失败都只记类型，不外抛
                error = e
            status = (
                "ok"
                if error is None
                else "timeout"
                if isinstance(error, TransportTimeout)
                else "rejected"
                if isinstance(error, OutputRejected)
                else "error"
            )
            retry = status in ("timeout", "error") and not isinstance(error, MissingApiKey)
            last = status == "ok" or not retry or attempt == MAX_ATTEMPTS
            call = ModelCall.model_validate(
                {
                    "call_id": self._ids.new_id(),
                    "request_id": request_id,
                    "stage": stage,
                    "provider": route.provider,
                    "model": route.model,
                    "prompt_template": template,
                    "started_at": started_at,
                    "duration_ms": max(0, round((self._timer() - start) * 1000)),
                    **tokens,
                    "status": status,
                    "fallback_used": last and status != "ok",
                    "error_type": None if error is None else type(error).__name__,
                }
            )
            calls.append(call)
            with self._record_lock:
                self._record(call)
            _log.info(
                "llm.call",
                call_id=call.call_id,
                request_id=request_id,
                status=status,
                attempt=attempt,
                duration_ms=call.duration_ms,
                error_type=call.error_type,
            )
            if last:
                return Completion(value=value if status == "ok" else None, calls=tuple(calls))
        raise AssertionError("unreachable")  # pragma: no cover
