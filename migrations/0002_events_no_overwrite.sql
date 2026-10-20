-- 记录只追加：任何写入方式（含 INSERT OR REPLACE）都不能覆盖同一记录标识的已有记录。
-- “有则替换”先删后插，默认不触发删除触发器，所以在插入前拦截。
CREATE TRIGGER events_no_overwrite BEFORE INSERT ON events
WHEN EXISTS (SELECT 1 FROM events WHERE event_id = new.event_id)
BEGIN
    SELECT RAISE(ABORT, 'events is append-only');
END;
