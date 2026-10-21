"""测试不碰真实的模型与向量服务（spec [2026-10-21] §13）。"""

import os

from ironm.llm.config import MODELS_PATH, load_models


def test_model_keys_cleared_inside_tests() -> None:
    """终端里设了模型或向量服务的密钥，每个测试开始前也已清掉（见 tests/conftest.py）。"""
    keys = {route.api_key_env for route in load_models(MODELS_PATH).routes.values()}
    assert {"DEEPSEEK_API_KEY", "MOONSHOT_API_KEY", "SILICONFLOW_API_KEY"} <= keys
    assert sorted(k for k in keys if k in os.environ) == []
