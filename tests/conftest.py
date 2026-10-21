from collections.abc import Iterator

import pytest

from ironm.llm.config import MODELS_PATH, load_models
from tests.memory_helpers import close_all

# 模型配置里用到的全部密钥（大模型与向量服务）。测试一律不碰真实服务：终端里设了这些密钥，
# 也会在每个测试开始前清掉；需要密钥的测试自己设置（多为假值配合假的传输层）
MODEL_KEYS = sorted({route.api_key_env for route in load_models(MODELS_PATH).routes.values()})


@pytest.fixture(autouse=True)
def _close_memory_dbs() -> Iterator[None]:
    yield
    close_all()


@pytest.fixture(autouse=True)
def _no_real_model_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MODEL_KEYS:
        monkeypatch.delenv(name, raising=False)
