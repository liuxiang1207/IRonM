-- 写入更正时要检查目标是否已被更正过（一条记录只能被更正一次），为 corrects 建索引。
CREATE INDEX events_corrects ON events (corrects);
