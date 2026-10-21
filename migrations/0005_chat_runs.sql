-- 问答运行记录（spec [2026-10-21] §5）：按请求编号回放。不含用户原话与回答原文。只追加。
CREATE TABLE chat_runs (
    seq         INTEGER PRIMARY KEY,
    request_id  TEXT    NOT NULL UNIQUE,
    user_id     TEXT    NOT NULL,
    record      TEXT    NOT NULL CHECK (json_valid(record))
) STRICT;

CREATE INDEX chat_runs_user ON chat_runs (user_id, seq);

CREATE TRIGGER chat_runs_no_update BEFORE UPDATE ON chat_runs
BEGIN
    SELECT RAISE(ABORT, 'chat_runs is append-only');
END;

CREATE TRIGGER chat_runs_no_delete BEFORE DELETE ON chat_runs
BEGIN
    SELECT RAISE(ABORT, 'chat_runs is append-only');
END;
