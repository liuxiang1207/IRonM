"""各环节的模型配置（spec [2026-10-21] §2）：连接与模型设置写在配置文件，密钥只从环境变量读取。"""

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ironm.contracts.model_call import ModelStage
from ironm.core.yamlfile import load_yaml

MODELS_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"


class ModelsConfigError(ValueError):
    pass


_HTTPS_URL = r"^https://[A-Za-z0-9.\-]+(:\d+)?(/[A-Za-z0-9._\-/]*)?$"


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ModelRoute(_Strict):
    """一个环节用哪家服务、哪个模型。不含密钥：api_key_env 是环境变量名。"""

    provider: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    base_url: Annotated[str, Field(pattern=_HTTPS_URL)]
    model: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")]
    api_key_env: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")]
    timeout_s: Annotated[float, Field(gt=0, le=120)]
    max_tokens: Annotated[int, Field(gt=0, le=32000)]
    extra_body: dict[str, Any] = {}

    @model_validator(mode="after")
    def _check(self) -> "ModelRoute":
        reserved = {"model", "messages", "max_tokens"} & set(self.extra_body)
        if reserved:
            raise ValueError(f"extra_body 不能覆盖 {sorted(reserved)}")
        return self


class ModelsConfig(_Strict):
    version: str
    routes: dict[ModelStage, ModelRoute]

    @model_validator(mode="after")
    def _all_stages(self) -> "ModelsConfig":
        missing = sorted(s.value for s in ModelStage if s not in self.routes)
        if missing:
            raise ValueError(f"缺少环节的模型配置：{missing}")
        return self


def load_models(path: Path) -> ModelsConfig:
    """读取并严格校验模型配置；任何问题都抛 ModelsConfigError（不回显取值）。"""
    try:
        data = load_yaml(path)
        # 经由 JSON 校验：严格模式下环节名按枚举取值解析
        return ModelsConfig.model_validate_json(json.dumps(data))
    except ValidationError as e:
        errors = e.errors(include_input=False)
        problems = [f"{'.'.join(map(str, err['loc']))}：{err['msg']}" for err in errors]
        raise ModelsConfigError(f"{path}：模型配置有误：" + "；".join(problems)) from e
    except (OSError, ValueError, TypeError) as e:
        raise ModelsConfigError(f"{path}：模型配置无法读取：{type(e).__name__}") from e
