"""结构化日志：每行一条 JSON，时间取自注入的 Clock。

为保证日志不含健康数据原文：事件名必须是固定的点分标识；附加字段只允许标识类（*_id）
和运行类字段；异常只记类型。违反时直接抛 LogFieldError，不做静默过滤。
"""

import json
import logging
import re
from typing import IO

from ironm.core.clock import Clock
from ironm.core.config import Settings

ROOT_LOGGER = "ironm"

_EVENT_RE = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*")
_ID_FIELD_RE = re.compile(r"[a-z][a-z0-9_]*_id")
# 运行类字段。新增前确认取值不可能携带健康数据原文
_RUNTIME_FIELDS = frozenset(
    {"status", "status_code", "method", "path", "duration_ms", "count", "attempt", "error_type"}
)
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

type FieldValue = str | int | float | bool | None


class LogFieldError(ValueError):
    pass


class _JsonFormatter(logging.Formatter):
    def __init__(self, clock: Clock) -> None:
        super().__init__()
        self._clock = clock

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, object] = {
            "ts": self._clock.now().isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.msg,
        }
        line.update(getattr(record, "ironm_fields", {}))
        return json.dumps(line, ensure_ascii=False)


class _IronmHandler(logging.StreamHandler[IO[str]]):
    """标记本模块安装的 handler，便于重新配置时替换。"""


def configure_logging(settings: Settings, clock: Clock, stream: IO[str]) -> None:
    """配置 ironm 日志输出；重复调用会替换上一次的配置。"""
    root = logging.getLogger(ROOT_LOGGER)
    for h in [h for h in root.handlers if isinstance(h, _IronmHandler)]:
        root.removeHandler(h)
    handler = _IronmHandler(stream)
    handler.setFormatter(_JsonFormatter(clock))
    root.addHandler(handler)
    root.setLevel(_LEVELS[settings.log_level])
    root.propagate = False


def _check(event: str, fields: dict[str, FieldValue]) -> None:
    if not _EVENT_RE.fullmatch(event):
        raise LogFieldError(f"日志事件名必须是点分小写标识，不能是自由文本：{event!r}")
    for key, value in fields.items():
        if not (_ID_FIELD_RE.fullmatch(key) or key in _RUNTIME_FIELDS):
            raise LogFieldError(f"日志字段不在允许范围内：{key}")
        if not isinstance(value, str | int | float | bool | None):
            raise LogFieldError(f"日志字段 {key} 只能是标量，实际为 {type(value).__name__}")


class StructLogger:
    def __init__(self, name: str) -> None:
        if name != ROOT_LOGGER and not name.startswith(ROOT_LOGGER + "."):
            raise LogFieldError(f"日志来源必须位于 {ROOT_LOGGER} 之下：{name}")
        self._logger = logging.getLogger(name)

    def _log(self, level: int, event: str, fields: dict[str, FieldValue]) -> None:
        _check(event, fields)
        self._logger.log(level, event, extra={"ironm_fields": fields})

    def debug(self, event: str, **fields: FieldValue) -> None:
        self._log(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: FieldValue) -> None:
        self._log(logging.INFO, event, fields)

    def warning(self, event: str, **fields: FieldValue) -> None:
        self._log(logging.WARNING, event, fields)

    def error(self, event: str, **fields: FieldValue) -> None:
        self._log(logging.ERROR, event, fields)

    def exception(self, event: str, exc: BaseException, **fields: FieldValue) -> None:
        """记录异常：只记类型。异常消息可能含用户输入的原文，不写入日志。"""
        self._log(logging.ERROR, event, {**fields, "error_type": type(exc).__name__})


def get_logger(name: str) -> StructLogger:
    return StructLogger(name)
