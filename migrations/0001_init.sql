-- 记录（事件）：只追加。改错是追加一条 corrects 指向原记录的新记录。
-- 时间一律为 UTC、固定格式（YYYY-MM-DDTHH:MM:SS.ffffffZ），可直接按字符串比较与排序。
CREATE TABLE events (
    seq            INTEGER PRIMARY KEY,
    event_id       TEXT    NOT NULL UNIQUE,  -- 客户端生成，去重键
    user_id        TEXT    NOT NULL,
    type           TEXT    NOT NULL,
    occurred_at    TEXT    NOT NULL,
    recorded_at    TEXT    NOT NULL,
    received_at    TEXT    NOT NULL,         -- 服务端接收时间
    source         TEXT    NOT NULL,
    confirmed      INTEGER NOT NULL CHECK (confirmed IN (0, 1)),
    raw            TEXT,                     -- 用户原话或设备原始字段
    stream_key     TEXT,
    corrects       TEXT    REFERENCES events (event_id),
    precision_hint TEXT,
    payload        TEXT    NOT NULL CHECK (json_valid(payload))
) STRICT;

CREATE INDEX events_user_time ON events (user_id, occurred_at);

CREATE TRIGGER events_no_update BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only');
END;

CREATE TRIGGER events_no_delete BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only');
END;

-- 全文检索：原文与身体不适描述。trigram 分词，中文按子串匹配（检索词至少 3 个字）。
-- 无内容表：只存索引，命中后按 rowid 回到 events 取数据。
CREATE VIRTUAL TABLE events_fts USING fts5 (raw, note, content = '', tokenize = 'trigram');

CREATE TRIGGER events_fts_insert AFTER INSERT ON events
BEGIN
    INSERT INTO events_fts (rowid, raw, note)
    VALUES (new.seq, new.raw, json_extract(new.payload, '$.text'));
END;
