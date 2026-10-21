-- 登录（spec [2026-10-21] §14）：白名单用户、一次性设备码、设备凭证。
-- 设备码与凭证只存 SHA-256 摘要，原文只在生成时交给管理员 / App 一次。
CREATE TABLE users (
    user_id     TEXT    PRIMARY KEY,
    name        TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    disabled_at TEXT                  -- 停用后所有设备立即失效
) STRICT;

CREATE TABLE device_codes (
    code_hash   TEXT    PRIMARY KEY,
    user_id     TEXT    NOT NULL REFERENCES users (user_id),
    created_at  TEXT    NOT NULL,
    expires_at  TEXT    NOT NULL,
    used_at     TEXT                  -- 用过一次即作废
) STRICT;

CREATE TABLE devices (
    device_id   TEXT    PRIMARY KEY,
    user_id     TEXT    NOT NULL REFERENCES users (user_id),
    token_hash  TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    revoked_at  TEXT
) STRICT;

CREATE INDEX devices_user ON devices (user_id);
