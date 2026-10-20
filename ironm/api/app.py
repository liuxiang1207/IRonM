"""FastAPI 应用入口。业务路由按模块逐步挂载。"""

from fastapi import FastAPI

from ironm import __version__

app = FastAPI(title="IronM", version=__version__)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """存活检查：只说明进程在运行，不检查依赖。"""
    return {"status": "ok", "version": __version__}
