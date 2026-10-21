"""模型输出校验：不合规的输出抛 OutputRejected，由网关记为“未通过校验”并交给兜底。"""

import json
import re
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.S)


class OutputRejected(ValueError):
    """输出不合规。消息里不带输出原文。"""


def json_model[M: BaseModel](model: type[M]) -> Callable[[str], M]:
    """把输出解析为 JSON 对象并按 model 校验；允许整段被 ```json 代码块包住。"""

    def parse(text: str) -> M:
        m = _FENCE_RE.match(text.strip())
        body = m.group(1) if m else text
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise OutputRejected("输出不是 JSON") from e
        try:
            return model.model_validate(data)
        except ValidationError as e:
            raise OutputRejected(f"输出不符合 {model.__name__}") from e

    return parse
