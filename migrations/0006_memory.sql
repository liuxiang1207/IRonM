-- 对话记忆（spec [2026-10-21] §8）。消息原文只存在这里，从不写入日志；保留期限与到期删除见后续任务。
CREATE TABLE chat_messages (
    seq         INTEGER PRIMARY KEY,
    user_id     TEXT    NOT NULL,
    request_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    text        TEXT    NOT NULL,
    intent      TEXT,                -- 用户消息的意图（摘要兜底时按话题汇总）
    at          TEXT    NOT NULL
) STRICT;

CREATE INDEX chat_messages_user ON chat_messages (user_id, seq);

-- 摘要：upto_seq 及以前的消息已压缩进 summary
CREATE TABLE chat_summaries (
    seq         INTEGER PRIMARY KEY,
    user_id     TEXT    NOT NULL,
    upto_seq    INTEGER NOT NULL,
    summary     TEXT    NOT NULL,
    source      TEXT    NOT NULL CHECK (source IN ('model', 'rule')),
    at          TEXT    NOT NULL
) STRICT;

CREATE INDEX chat_summaries_user ON chat_summaries (user_id, seq);

-- 画像：每次变化写一个新版本，只追加
CREATE TABLE profiles (
    seq         INTEGER PRIMARY KEY,
    user_id     TEXT    NOT NULL,
    version     INTEGER NOT NULL CHECK (version >= 1),
    data        TEXT    NOT NULL CHECK (json_valid(data)),
    reason      TEXT    NOT NULL,     -- 变化来源：候选编号或 initial
    at          TEXT    NOT NULL,
    UNIQUE (user_id, version)
) STRICT;

CREATE TRIGGER profiles_no_update BEFORE UPDATE ON profiles
BEGIN
    SELECT RAISE(ABORT, 'profiles is append-only');
END;

-- 画像候选：从对话中发现的变化，用户确认后才写入画像（会让分层变严的立即生效）
CREATE TABLE profile_candidates (
    seq          INTEGER PRIMARY KEY,
    candidate_id TEXT    NOT NULL UNIQUE,
    user_id      TEXT    NOT NULL,
    request_id   TEXT    NOT NULL,
    field        TEXT    NOT NULL,
    op           TEXT    NOT NULL CHECK (op IN ('set', 'add')),
    value        TEXT    NOT NULL CHECK (json_valid(value)),
    immediate    INTEGER NOT NULL CHECK (immediate IN (0, 1)),
    status       TEXT    NOT NULL CHECK (status IN ('pending', 'confirmed', 'rejected')),
    at           TEXT    NOT NULL,
    decided_at   TEXT
) STRICT;

CREATE INDEX profile_candidates_user ON profile_candidates (user_id, status);
