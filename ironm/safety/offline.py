"""生成 App 离线版的数据（spec §22、§24）。用法：python -m ironm.safety.offline [输出路径]。

数据与服务端同源：急症词表、统一文字所用的分句符号与空白，以及红旗四段式的模板。
"""

import json
import sys
from pathlib import Path

from ironm.safety.decision import SafetyKit, load_kit
from ironm.safety.emergency import CLAUSE_SEPARATORS, WHITESPACE

OFFLINE_PATH = Path(__file__).resolve().parents[2] / "app/src/safety/emergency_terms.json"
FORMAT = 2


def render(kit: SafetyKit) -> str:
    r = kit.replies
    titles = r.titles.model_dump(include={"conclusion", "stop", "cannot_judge", "help"})
    data = {
        "format": FORMAT,
        "separators": CLAUSE_SEPARATORS,
        "whitespace": WHITESPACE,
        **kit.terms.model_dump(mode="json"),
        "replies": {
            "version": r.version,
            "review": r.review,
            "titles": titles,
            "levels": {k: v.model_dump(mode="json") for k, v in r.levels.items()},
            "categories": {k: v.model_dump(mode="json") for k, v in r.categories.items()},
        },
    }
    return json.dumps(data, ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str]) -> None:
    out = Path(argv[0]) if argv else OFFLINE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(load_kit()), encoding="utf-8")
    print(f"已生成 {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
