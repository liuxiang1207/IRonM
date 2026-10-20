"""FastAPI 应用入口。业务路由按模块逐步挂载。"""

import os
import sys

from fastapi import FastAPI

from ironm import __version__
from ironm.core.clock import SystemClock
from ironm.core.config import load_settings
from ironm.core.log import configure_logging

# 配置有误时在导入阶段直接失败，服务不会带着错误配置启动
settings = load_settings(os.environ)
configure_logging(settings, SystemClock(), sys.stderr)

app = FastAPI(title="IronM", version=__version__)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """存活检查：只说明进程在运行，不检查依赖。"""
    return {"status": "ok", "version": __version__}
