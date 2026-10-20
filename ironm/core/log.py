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
# 取值规则：标识与状态只能是不含空格、中文的短标识；路径不得带查询参数
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:\-]{1,128}")
_PATH_RE = re.compile(r"/[A-Za-z0-9_.\-/]{0,255}")
_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
# 运行类字段及其取值规则。新增前确认取值不可能携带健康数据原文
_NUMERIC_FIELDS = frozenset({"status_code", "duration_ms", "count", "attempt"})
_TEXT_FIELDS = {"status": _TOKEN_RE, "error_type": _TOKEN_RE, "path": _PATH_RE}
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
        }
        fields = getattr(record, "ironm_fields", None)
        if fields is None:
            # 绕过结构化接口写的日志：内容可能含健康数据，只记下“有一条被拒”，不输出原文
            line["event"] = "log.rejected_unstructured"
        else:
            line["event"] = record.msg
            line.update(fields)
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
        if value is None:
            continue
        if key in _NUMERIC_FIELDS:
            ok = isinstance(value, int | float) and not isinstance(value, bool)
        elif key == "method":
            ok = value in _METHODS
        elif key in _TEXT_FIELDS:
            ok = isinstance(value, str) and _TEXT_FIELDS[key].fullmatch(value) is not None
        elif _ID_FIELD_RE.fullmatch(key):
            ok = isinstance(value, int) or (
                isinstance(value, str) and _TOKEN_RE.fullmatch(value) is not None
            )
        else:
            raise LogFieldError(f"日志字段不在允许范围内：{key}")
        if not ok:
            # 不在错误信息里回显取值：它可能正是健康数据原文
            raise LogFieldError(f"日志字段 {key} 的取值不合规（{type(value).__name__}）")


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
