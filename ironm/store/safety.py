"""安全标记与模型调用记录（spec [2026-10-21] §2、§3）。只追加，不含用户原话与回答原文。"""

import json
import sqlite3

from ironm.contracts.model_call import ModelCall, ModelStage
from ironm.contracts.screening import SafetyFlag
from ironm.contracts.unknown import is_known
from ironm.core.ids import IdGenerator
from ironm.store.db import from_db_time, to_db_time, transaction


class SafetyFlagStore:
    def __init__(self, conn: sqlite3.Connection, ids: IdGenerator) -> None:
        self._conn = conn
        self._ids = ids

    def add(self, user_id: str, request_id: str, flag: SafetyFlag) -> str:
        flag_id = self._ids.new_id()
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO safety_flags (flag_id, user_id, request_id, at, level, rule_ids,"
                " categories) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    flag_id,
                    user_id,
                    request_id,
                    to_db_time(flag.at),
                    flag.level,
                    json.dumps(list(flag.rule_ids)),
                    json.dumps(list(flag.categories)),
                ),
            )
        return flag_id

    def for_user(self, user_id: str) -> list[SafetyFlag]:
        rows = self._conn.execute(
            "SELECT at, level, rule_ids, categories FROM safety_flags WHERE user_id = ?"
            " ORDER BY at, seq",
            (user_id,),
        ).fetchall()
        return [
            SafetyFlag(
                at=from_db_time(r["at"]),
                level=r["level"],
                rule_ids=tuple(json.loads(r["rule_ids"])),
                categories=tuple(json.loads(r["categories"])),
            )
            for r in rows
        ]


_CALL_COLUMNS = (
    "call_id", "request_id", "stage", "provider", "model", "prompt_template", "started_at",
    "duration_ms", "input_tokens", "output_tokens", "status", "fallback_used", "error_type",
)  # fmt: skip


class ModelCallStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, call: ModelCall) -> None:
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO model_calls (call_id, request_id, stage, provider, model,"
                " prompt_template, started_at, duration_ms, input_tokens, output_tokens, status,"
                " fallback_used, error_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    call.call_id,
                    call.request_id,
                    call.stage.value,
                    call.provider,
                    call.model,
                    call.prompt_template,
                    to_db_time(call.started_at),
                    call.duration_ms,
                    call.input_tokens if is_known(call.input_tokens) else None,
                    call.output_tokens if is_known(call.output_tokens) else None,
                    call.status,
                    int(call.fallback_used),
                    call.error_type,
                ),
            )

    def for_request(self, request_id: str) -> list[ModelCall]:
        rows = self._conn.execute(
            f"SELECT {', '.join(_CALL_COLUMNS)} FROM model_calls WHERE request_id = ? ORDER BY seq",
            (request_id,),
        ).fetchall()
        out = []
        for r in rows:
            data = {k: r[k] for k in _CALL_COLUMNS}
            data["stage"] = ModelStage(r["stage"])
            data["started_at"] = from_db_time(r["started_at"])
            data["fallback_used"] = bool(r["fallback_used"])
            for key in ("input_tokens", "output_tokens"):
                if data[key] is None:
                    del data[key]
            out.append(ModelCall.model_validate(data))
        return out
