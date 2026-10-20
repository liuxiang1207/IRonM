"""运行配置：只从 IRONM_ 前缀的环境变量读取。未知变量或非法取值直接报错，不回退默认值。"""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

PREFIX = "IRONM_"


class ConfigError(ValueError):
    pass


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    db_path: Path = Path("data/ironm.sqlite")


def load_settings(environ: Mapping[str, str]) -> Settings:
    values = {k.removeprefix(PREFIX).lower(): v for k, v in environ.items() if k.startswith(PREFIX)}
    try:
        return Settings.model_validate(values)
    except ValidationError as e:
        problems = [f"{PREFIX}{str(err['loc'][0]).upper()}：{err['msg']}" for err in e.errors()]
        raise ConfigError("配置错误；" + "；".join(problems)) from e
