"""生成 App 离线版的词表数据（spec §22）。用法：python -m ironm.safety.offline [输出路径]。

数据与服务端使用的词表同源；统一文字所用的分句符号与空白一并写入，离线脚本不另设。
"""

import json
import sys
from pathlib import Path

from ironm.safety.emergency import CLAUSE_SEPARATORS, TERMS_PATH, WHITESPACE, TermBook, load_terms

OFFLINE_PATH = Path(__file__).resolve().parents[2] / "app/src/safety/emergency_terms.json"
FORMAT = 1


def render(book: TermBook) -> str:
    data = {
        "format": FORMAT,
        "separators": CLAUSE_SEPARATORS,
        "whitespace": WHITESPACE,
        **book.model_dump(mode="json"),
    }
    return json.dumps(data, ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str]) -> None:
    out = Path(argv[0]) if argv else OFFLINE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(load_terms(TERMS_PATH)), encoding="utf-8")
    print(f"已生成 {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
