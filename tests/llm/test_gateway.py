"""模型接入（spec [2026-10-21] §2）：按环节配置、超时重试 1 次、校验不过即兜底、只记元数据。"""

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from ironm.contracts.model_call import ModelCall, ModelStage
from ironm.core.clock import FixedClock
from ironm.core.config import Settings
from ironm.core.ids import SeededIdGenerator
from ironm.core.log import configure_logging
from ironm.llm.config import MODELS_PATH, ModelsConfigError, load_models
from ironm.llm.gateway import Gateway, Message, TransportTimeout
from ironm.llm.validate import OutputRejected, json_model

NOW = datetime(2026, 10, 21, 8, tzinfo=UTC)
KEY = "sk-test-secret-value"
ENV = {"DEEPSEEK_API_KEY": KEY, "MOONSHOT_API_KEY": KEY}
PROMPT = "用户原话：我膝盖疼，想练腿"


class Intent(BaseModel):
    intent: str
    confidence: float


def _reply(content: str, finish: str = "stop") -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }


class FakeTransport:
    """依次返回预设结果；元素为异常时抛出。记录每次请求。"""

    def __init__(self, *results: object) -> None:
        self.results = list(results)
        self.requests: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout_s: float
    ) -> dict[str, object]:
        self.requests.append((url, headers, body, timeout_s))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        assert isinstance(result, dict)
        return result


def _gateway(
    transport: FakeTransport, environ: dict[str, str] = ENV
) -> tuple[Gateway, list[ModelCall]]:
    calls: list[ModelCall] = []
    gw = Gateway(
        load_models(MODELS_PATH),
        environ=environ,
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(1),
        transport=transport,
        record=calls.append,
    )
    return gw, calls


def _ask(gw: Gateway) -> Any:
    return gw.complete(
        ModelStage.INTENT,
        request_id="req-1",
        template="intent_l3@1.0.0",
        messages=[Message(role="system", content="识别意图"), Message(role="user", content=PROMPT)],
        validate=json_model(Intent),
    )


def test_bundled_config_covers_every_stage_and_has_no_keys() -> None:
    config = load_models(MODELS_PATH)
    assert set(config.routes) == set(ModelStage)
    text = MODELS_PATH.read_text(encoding="utf-8")
    assert "sk-" not in text and "api_key:" not in text


def test_success_returns_validated_value_and_one_record() -> None:
    gw, calls = _gateway(FakeTransport(_reply('{"intent": "training", "confidence": 0.9}')))
    result = _ask(gw)
    assert result.value == Intent(intent="training", confidence=0.9)
    assert [c.status for c in calls] == ["ok"]
    call = calls[0]
    assert call.request_id == "req-1" and call.stage is ModelStage.INTENT
    assert call.prompt_template == "intent_l3@1.0.0" and call.fallback_used is False
    assert call.input_tokens == 12 and call.output_tokens == 5
    assert call.model == load_models(MODELS_PATH).routes[ModelStage.INTENT].model


def test_request_uses_route_settings_and_key_from_environment() -> None:
    transport = FakeTransport(_reply('{"intent": "x", "confidence": 1}'))
    gw, _ = _gateway(transport)
    _ask(gw)
    route = load_models(MODELS_PATH).routes[ModelStage.INTENT]
    url, headers, body, timeout_s = transport.requests[0]
    assert url == f"{route.base_url}/chat/completions"
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert body["model"] == route.model and timeout_s == route.timeout_s
    assert body["messages"][1] == {"role": "user", "content": PROMPT}


def test_timeout_then_success_retries_once() -> None:
    ok = _reply('{"intent": "x", "confidence": 1}')
    gw, calls = _gateway(FakeTransport(TransportTimeout(), ok))
    result = _ask(gw)
    assert result.value is not None
    assert [c.status for c in calls] == ["timeout", "ok"]
    assert [c.fallback_used for c in calls] == [False, False]


def test_two_failures_fall_back_without_raising() -> None:
    gw, calls = _gateway(FakeTransport(ConnectionError("boom"), TransportTimeout()))
    result = _ask(gw)
    assert result.value is None
    assert [c.status for c in calls] == ["error", "timeout"]
    assert calls[0].error_type == "ConnectionError"
    assert calls[-1].fallback_used is True


def test_invalid_output_is_rejected_without_retry() -> None:
    gw, calls = _gateway(FakeTransport(_reply("不是 JSON")))
    result = _ask(gw)
    assert result.value is None
    assert [(c.status, c.fallback_used) for c in calls] == [("rejected", True)]
    assert calls[0].error_type == "OutputRejected"


def test_truncated_output_is_rejected() -> None:
    complete_json = '{"intent": "x", "confidence": 1}'
    gw, calls = _gateway(FakeTransport(_reply(complete_json, finish="length")))
    assert _ask(gw).value is None
    assert calls[0].status == "rejected"


def test_missing_key_fails_without_network_call() -> None:
    transport = FakeTransport()
    gw, calls = _gateway(transport, environ={})
    assert _ask(gw).value is None
    assert transport.requests == []
    assert [(c.status, c.error_type) for c in calls] == [("error", "MissingApiKey")]


def test_records_and_logs_contain_no_prompt_answer_or_key() -> None:
    stream = io.StringIO()
    configure_logging(Settings(log_level="debug"), FixedClock(NOW), stream)
    answer = '{"intent": "训练知识", "confidence": 0.9}'
    gw, calls = _gateway(FakeTransport(ConnectionError(KEY), _reply(answer)))
    _ask(gw)
    dumped = json.dumps([c.model_dump(mode="json") for c in calls], ensure_ascii=False)
    logs = stream.getvalue()
    assert logs.count('"event": "llm.call"') == 2
    for text in (dumped, logs):
        assert KEY not in text and "膝盖" not in text and "训练知识" not in text


def test_changing_model_name_in_config_changes_records(tmp_path: Path) -> None:
    data = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))
    data["routes"]["intent_l3"]["model"] = "another-model"
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    calls: list[ModelCall] = []
    gw = Gateway(
        load_models(path),
        environ=ENV,
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(1),
        transport=FakeTransport(_reply('{"intent": "x", "confidence": 1}')),
        record=calls.append,
    )
    _ask(gw)
    assert calls[0].model == "another-model"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["routes"].pop("answer"), "answer"),
        (lambda d: d["routes"]["answer"].update(api_key="sk-x"), "api_key"),
        (lambda d: d["routes"]["answer"].update(base_url="http://example.com"), "https"),
        (lambda d: d["routes"]["answer"].update(api_key_env="deepseek"), "api_key_env"),
        (lambda d: d["routes"]["answer"].update(timeout_s=0), "timeout_s"),
        (lambda d: d["routes"]["answer"].update(extra_body={"model": "x"}), "extra_body"),
    ],
)
def test_bad_config_rejected(tmp_path: Path, mutate: Any, message: str) -> None:
    data = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ModelsConfigError, match=message):
        load_models(path)


def test_duration_measured_per_attempt() -> None:
    ticks = iter([0.0, 1.25, 2.0, 2.5])
    calls: list[ModelCall] = []
    gw = Gateway(
        load_models(MODELS_PATH),
        environ=ENV,
        clock=FixedClock(NOW),
        ids=SeededIdGenerator(1),
        transport=FakeTransport(TransportTimeout(), _reply('{"intent": "x", "confidence": 1}')),
        record=calls.append,
        timer=lambda: next(ticks),
    )
    _ask(gw)
    assert [c.duration_ms for c in calls] == [1250, 500]


def test_json_model_accepts_fenced_json_and_rejects_bad_shape() -> None:
    parse = json_model(Intent)
    assert parse('```json\n{"intent": "x", "confidence": 0.5}\n```').intent == "x"
    with pytest.raises(OutputRejected):
        parse('{"intent": "x"}')
    with pytest.raises(OutputRejected):
        parse("[1, 2]")
