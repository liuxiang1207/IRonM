"""知识库检查（人工维护段落用）：格式、重复、出处、复审期、引用完整、检索缺口。

用法：
    uv run python tools/check_kb.py            # 检查 IRONM_KB_DIR 或 kb/
不检查审定的实质内容，只检查“能不能安全入库”。
"""

import sys
from datetime import date
from pathlib import Path

from ironm.core.config import load_settings
from ironm.knowledge.kb import KnowledgeBase, Retriever, load_passages

REVIEW_MONTHS = 12
MIN_CHARS, MAX_CHARS = 60, 400


def main() -> int:
    folder = load_settings({}).kb_dir or Path("kb")
    if not folder.is_dir():
        print(f"资料目录不存在：{folder}")
        return 1
    # 每个领域一个子目录，段落文件放在子目录下
    passages = [
        p for sub in sorted(d for d in folder.iterdir() if d.is_dir()) for p in load_passages(sub)
    ]
    if not passages:
        print(f"没有段落：{folder}")
        return 1

    problems: list[str] = []
    seen_titles: dict[str, str] = {}
    today = date.today()
    for p in passages:
        if len(p.text) < MIN_CHARS:
            problems.append(f"{p.passage_id}: 正文过短（{len(p.text)} 字）")
        if len(p.text) > MAX_CHARS:
            problems.append(f"{p.passage_id}: 正文过长（{len(p.text)} 字），建议拆成多段")
        if p.review == "approved" and not p.source.strip():
            problems.append(f"{p.passage_id}: 已审定但缺少出处")
        if p.review == "approved" and p.reviewed_at is None:
            problems.append(f"{p.passage_id}: 已审定但没有复审日期")
        if p.reviewed_at and (today - p.reviewed_at).days > REVIEW_MONTHS * 30:
            problems.append(f"{p.passage_id}: 超过 {REVIEW_MONTHS} 个月未复审")
        key = p.title.strip()
        if key in seen_titles:
            problems.append(f"{p.passage_id}: 标题与 {seen_titles[key]} 重复：{key}")
        seen_titles[key] = p.passage_id

    ids = {p.passage_id for p in passages}
    for p in passages:
        for ref in p.see_also:
            if ref not in ids:
                problems.append(f"{p.passage_id}: see_also 指向不存在的段落 {ref}")

    # 检索探针：只看“有没有召回到”，用已审定的段落才有意义；未审定阶段跳过
    retriever = Retriever(KnowledgeBase(passages))
    probes = {
        "一周练几次力量": "trn-principles-002",
        "怎么加重量": "trn-principles-003",
        "组间休息多久": "trn-principles-007",
        "有氧做多少": "trn-principles-001",
    }
    approved = sum(p.review == "approved" for p in passages)
    misses: list[str] = []
    if approved:
        for question, expect in probes.items():
            domain = next(p.domain for p in passages if p.passage_id == expect)
            got = retriever.retrieve([question], domain).passages
            top = [r.passage.passage_id for r in got[:3]]
            if expect not in top:
                misses.append(f"「{question}」应命中 {expect}，实际前 3 为 {top}")
    else:
        print("提示：没有已审定段落，跳过检索探针")

    print(f"段落 {len(passages)} 条；已审定 {sum(p.review == 'approved' for p in passages)} 条")
    for line in problems:
        print("问题:", line)
    for line in misses:
        print("检索:", line)
    if problems or misses:
        return 1
    print("检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
