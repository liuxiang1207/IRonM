-- 复盘、到期复核与定时任务（spec [2026-10-21] §16、§17）。

-- 决策日志：每次复盘一条，只追加；record 为 JSON（事实、推断、不确定、决策、三道关的结果）
CREATE TABLE decision_logs (
    seq         INTEGER PRIMARY KEY,
    user_id     TEXT    NOT NULL,
    kind        TEXT    NOT NULL CHECK (kind IN ('weekly', 'biweekly', 'recheck')),
    period      TEXT    NOT NULL,     -- 周期：周一的日期，或复核的调整编号
    record      TEXT    NOT NULL CHECK (json_valid(record)),
    at          TEXT    NOT NULL
) STRICT;

CREATE INDEX decision_logs_user ON decision_logs (user_id, seq);

CREATE TRIGGER decision_logs_no_update BEFORE UPDATE ON decision_logs
BEGIN
    SELECT RAISE(ABORT, 'decision_logs is append-only');
END;

-- 已发布的调整：到复核日检查，保留、撤回或延后
CREATE TABLE adjustments (
    adjustment_id TEXT    PRIMARY KEY,
    user_id       TEXT    NOT NULL,
    field         TEXT    NOT NULL,
    from_value    TEXT    CHECK (from_value IS NULL OR json_valid(from_value)),
    to_value      TEXT    NOT NULL CHECK (json_valid(to_value)),
    direction     TEXT    NOT NULL CHECK (direction IN ('up', 'down', 'same')),
    plan_version  INTEGER NOT NULL,
    review_date   TEXT    NOT NULL,   -- 用户时区的日期
    status        TEXT    NOT NULL CHECK (status IN ('pending', 'kept', 'rolled_back', 'superseded')),
    decided_at    TEXT
) STRICT;

CREATE INDEX adjustments_due ON adjustments (status, review_date);

-- 定时任务：同一用户、同一任务、同一周期只有一行（job_key 唯一）
CREATE TABLE jobs (
    job_key     TEXT    PRIMARY KEY,
    kind        TEXT    NOT NULL,
    user_id     TEXT    NOT NULL,
    period      TEXT    NOT NULL,
    due_at      TEXT    NOT NULL,
    status      TEXT    NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts    INTEGER NOT NULL DEFAULT 0,
    last_error  TEXT,                 -- 只记异常类型
    updated_at  TEXT    NOT NULL
) STRICT;

CREATE INDEX jobs_due ON jobs (status, due_at);
