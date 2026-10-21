-- 安全标记（spec [2026-10-21] §3）：本人的急症或红旗判定。只追加；不含用户原话。
CREATE TABLE safety_flags (
    seq         INTEGER PRIMARY KEY,
    flag_id     TEXT    NOT NULL UNIQUE,
    user_id     TEXT    NOT NULL,
    request_id  TEXT    NOT NULL,
    at          TEXT    NOT NULL,
    level       TEXT    NOT NULL CHECK (level IN ('emergency', 'red_flag')),
    rule_ids    TEXT    NOT NULL CHECK (json_valid(rule_ids)),
    categories  TEXT    NOT NULL CHECK (json_valid(categories))
) STRICT;

CREATE INDEX safety_flags_user_time ON safety_flags (user_id, at);

CREATE TRIGGER safety_flags_no_update BEFORE UPDATE ON safety_flags
BEGIN
    SELECT RAISE(ABORT, 'safety_flags is append-only');
END;

CREATE TRIGGER safety_flags_no_delete BEFORE DELETE ON safety_flags
BEGIN
    SELECT RAISE(ABORT, 'safety_flags is append-only');
END;

-- 模型调用记录（spec [2026-10-21] §2）：只有元数据，不存提示词与回答原文。只追加。
CREATE TABLE model_calls (
    seq             INTEGER PRIMARY KEY,
    call_id         TEXT    NOT NULL UNIQUE,
    request_id      TEXT    NOT NULL,
    stage           TEXT    NOT NULL,
    provider        TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    prompt_template TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    duration_ms     INTEGER NOT NULL CHECK (duration_ms >= 0),
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    status          TEXT    NOT NULL CHECK (status IN ('ok', 'timeout', 'error', 'rejected')),
    fallback_used   INTEGER NOT NULL CHECK (fallback_used IN (0, 1)),
    error_type      TEXT
) STRICT;

CREATE INDEX model_calls_request ON model_calls (request_id);

CREATE TRIGGER model_calls_no_update BEFORE UPDATE ON model_calls
BEGIN
    SELECT RAISE(ABORT, 'model_calls is append-only');
END;

CREATE TRIGGER model_calls_no_delete BEFORE DELETE ON model_calls
BEGIN
    SELECT RAISE(ABORT, 'model_calls is append-only');
END;
