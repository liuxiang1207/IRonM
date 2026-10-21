-- 计划版本（spec [2026-10-21] §15）：每次生成或调整都写一个新版本，只增不改；回退也是新版本。
-- record 为 JSON：计划版本、依据卡、分层与提示、基线测试方案、生成原因。
CREATE TABLE plan_versions (
    seq         INTEGER PRIMARY KEY,
    user_id     TEXT    NOT NULL,
    version     INTEGER NOT NULL CHECK (version >= 1),
    record      TEXT    NOT NULL CHECK (json_valid(record)),
    reason      TEXT    NOT NULL,     -- onboarding / weekly_review / rollback 等
    at          TEXT    NOT NULL,
    UNIQUE (user_id, version)
) STRICT;

CREATE TRIGGER plan_versions_no_update BEFORE UPDATE ON plan_versions
BEGIN
    SELECT RAISE(ABORT, 'plan_versions is append-only');
END;
