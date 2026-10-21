"""运行配置：只从 IRONM_ 前缀的环境变量读取。未知变量或非法取值直接报错，不回退默认值。"""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PREFIX = "IRONM_"


class ConfigError(ValueError):
    pass


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    db_path: Path = Path("data/ironm.sqlite")
    rules_dir: Path = Path("rules")
    models_path: Path = Path("config/models.yaml")
    # 知识库资料目录（不进仓库）；不设时只用问答卡与动作库（spec [2026-10-21] §12）
    kb_dir: Path | None = None
    # 后台定时任务（spec [2026-10-21] §16）：复盘、复核、提醒、到期删除、备份；默认关，部署时打开
    scheduler: bool = False
    scheduler_interval_s: int = Field(default=60, ge=5, le=3600)


def load_settings(environ: Mapping[str, str]) -> Settings:
    keys = [k for k in environ if k.startswith(PREFIX)]
    lower = sorted(k for k in keys if k != k.upper())
    if lower:
        raise ConfigError(f"配置错误；配置项名必须全大写：{', '.join(lower)}")
    values = {k.removeprefix(PREFIX).lower(): environ[k] for k in keys}
    try:
        return Settings.model_validate(values)
    except ValidationError as e:
        problems = [f"{PREFIX}{str(err['loc'][0]).upper()}：{err['msg']}" for err in e.errors()]
        raise ConfigError("配置错误；" + "；".join(problems)) from e
