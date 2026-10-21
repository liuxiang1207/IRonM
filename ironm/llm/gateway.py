"""模型网关（spec [2026-10-21] §2）：兼容常见对话接口的模型服务。

- 每个环节按配置选模型；密钥只从环境变量读取，缺失时不发请求、记为出错。
- 超时或出错时最多重试 1 次；输出没通过校验（含被截断）不重试，直接交给调用方兜底。
- 每次调用留一条 ModelCall 记录（只有元数据），并写一行日志；都不含提示词、回答原文与密钥。
- 网关本身从不抛出调用异常：失败时返回的 value 为 None，由调用方使用兜底。
"""

import json
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
                body: dict[str, Any] = {
                    "model": route.model,
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "max_tokens": route.max_tokens,
                    **route.extra_body,
                }
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                resp = self._transport(
                    f"{route.base_url}/chat/completions", headers, body, route.timeout_s
                )
                tokens = _tokens(resp)
                value = validate(_content(resp))
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
