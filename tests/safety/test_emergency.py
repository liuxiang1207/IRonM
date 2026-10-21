"""急症词表与匹配（spec [2026-10-20] §21）。红旗测试集本身见 test_red_flags.py。"""

import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from ironm.api.app import create_app
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.safety.emergency import (
    TERMS_PATH,
    Assessment,
    TermLoadError,
    assess,
    clauses,
    load_terms,
)
from tests.test_api_events import NOW

# 不在红旗测试集里的改写说法：同义词、换语序、他人发生。防止词表只贴合测试集。
PARAPHRASES = [
    ("跑着跑着胸口剧痛，到现在都没缓解", "emergency"),
    ("心口疼得厉害，左手臂也麻", "emergency"),
    ("胸前闷得慌，一直冒冷汗", "emergency"),
    ("练完突然昏过去了", "emergency"),
    ("我妈打羽毛球时突然倒地，叫不醒", "emergency"),
    ("刚才看到有人在跑道上晕倒了", "emergency"),
    ("有个会员在跑步机上抽搐", "emergency"),
    ("高温天跑完十公里，人有点神志不清", "emergency"),
    ("嘴唇发青，喘不过气", "emergency"),
    ("深蹲时突然头疼得受不了，从没这么疼过", "emergency"),
    ("右边半边脸麻了，胳膊也没劲", "emergency"),
    ("昨天练了三百个深蹲，今天小便是浓茶色", "emergency"),
    ("我是一型糖尿病，刚跑完出了一身虚汗", "emergency"),
    ("孕30周，今天跳操后感觉胎动变少了", "emergency"),
    ("跑完步高压190，头很晕", "emergency"),
    ("一侧小腿肿起来了，按着疼", "emergency"),
    ("腰痛，大小便都控制不住了", "emergency"),
    ("最近练腿老是头晕", "red_flag"),
    ("这周跑步时心脏乱跳了好几次", "red_flag"),
    ("跑步时左胸口有点闷，停下来就好", "red_flag"),
    ("爬三楼就气短，比以前差很多", "red_flag"),
    ("打篮球崴了脚，肿得不能着地", "red_flag"),
    ("最近吃完饭会抠喉咙吐出来", "red_flag"),
    ("腰疼得厉害，左腿一直发麻", "red_flag"),
    ("膝盖肿了好几天，摸着发烫", "red_flag"),
    ("肩膀疼得睡不着", "red_flag"),
    ("量了血压 185/100，没什么感觉", "red_flag"),
    # 审定后补充（spec §26）
    ("最近真的不想活了", "red_flag"),
    ("我割腕了", "emergency"),
    ("练完全身起风团，喉咙发紧", "emergency"),
    ("运动后全身起风团，喘不上气", "emergency"),
    ("跑完身上起了荨麻疹", "red_flag"),
    ("哮喘发作了，喷了药也不管用", "emergency"),
    ("跑步时哮喘犯了", "red_flag"),
    ("头撞到杠铃，吐了好几次", "emergency"),
    ("卧推时杠铃砸到头了", "red_flag"),
    ("突然一只眼睛黑了看不清", "emergency"),
    ("右手突然没劲，拿不住东西", "emergency"),
    ("血压190/115，喘不上气", "emergency"),
    ("我在吃格列美脲，练完发抖出冷汗", "emergency"),
    ("怀孕了，羊水破了", "emergency"),
    ("怀孕5个月，跑步时有点头晕", "red_flag"),
    ("我中暑了，头痛恶心", "red_flag"),
    ("大太阳下跑步晕倒了", "emergency"),
    ("今天吃了泻药", "red_flag"),
    ("吐血了", "emergency"),
    ("最近大便是黑的，柏油样", "red_flag"),
    ("打了减肥针，肚子疼到后背", "red_flag"),
]

# 含关键字但没有症状：健身语境里的常见说法不能被拦下
ORDINARY = [
    "深蹲动作变形了怎么办",
    "我要去吐槽一下教练",
    "一边跑步一边听歌，小腿有点酸",
    "练完胸部好紧",
    "今天练胸和背，后背有点酸",
    "卧推要不要憋气",
    "跑步时心跳很快正常吗",
    "血压计哪个牌子好",
    "晕车好难受",
    # 审定后修正的误报（spec §26）
    "发力时把气吐出来",
    "呼气时把气吐掉",
    "吃完就吐槽教练",
    "我有胰岛素抵抗，练完腿有点酸",
    "早上迷迷糊糊起来去跑步",
    "喝了可乐老想尿尿",
    "手机没有反应了",
    "我还在坚持跑步",
]


# 否定说法与以前发生过（spec §22）
NEGATIONS = [
    ("没有胸痛", "none"),
    ("今天不头晕", "none"),
    ("没有心慌也没有胸闷", "none"),
    ("怀孕五个月，运动后未见红", "none"),
    ("胸口不疼了", "none"),
    ("没有心慌，但是胸口还在疼", "emergency"),
    ("胸口不闷但是很疼", "red_flag"),  # 同一分句里另一处没被否定
    ("差点没晕倒", "emergency"),
    ("大太阳下跑步，不出汗了", "emergency"),  # 说法本身带否定字
    ("跑完喘不上气，嘴唇发紫", "emergency"),
    ("爬楼梯胸闷，坐下也不缓解", "emergency"),
    ("我朋友以前晕倒过", "none"),
    ("上周胸口疼过，查了没问题", "none"),
    ("以前胸口疼过，今天又疼了", "red_flag"),  # 再次发生：以前发生过的照常算
    ("晕倒过去了", "emergency"),  # “过去”不是以前发生过
    ("没有胸痛。胸口很闷", "red_flag"),  # 否定只影响所在分句
    ("没头痛，血压190/120", "red_flag"),
    (
        "说没有胸闷其实也胸闷",
        "red_flag",
    ),  # 同一说法出现两处，第一处被否定、第二处没有  # 血压数值不受否定影响；头痛被否定，不升为急症
]


@pytest.mark.parametrize(("text", "expect"), NEGATIONS)
def test_negation_and_past(text: str, expect: str) -> None:
    assert assess(text).level == expect


@pytest.mark.parametrize(("text", "expect"), PARAPHRASES)
def test_paraphrases(text: str, expect: str) -> None:
    assert assess(text).level == expect


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_fitness_talk_not_flagged(text: str) -> None:
    assert assess(text) == Assessment(level="none")


def test_emergency_wins_and_all_rules_listed() -> None:
    result = assess("胸口疼，现在还在疼")
    assert result.level == "emergency"
    assert {"EM-CHEST-ONGOING", "RF-CHEST"} <= set(result.rule_ids)
    assert result.categories == ("cardiovascular",)


def test_combo_within_one_clause_only() -> None:
    assert assess("胸口那里有点疼").level == "red_flag"
    assert assess("胸口有点酸，膝盖疼").level == "none"  # “胸口”与“疼”分在两个分句


UNIFIED = [
    "血压 190／120 ＭＭＨＧ，头痛",
    "胸 口 痛",
    "高压１９０",
    "胸\u3000口\u00a0痛",
    "胸\u200b闷",
]


def test_unified_text() -> None:
    assert clauses("血压 190／120 mmHg，头痛！") == ["血压190/120mmhg", "头痛"]
    assert assess("血压 190／120 ＭＭＨＧ，头痛").level == "emergency"
    assert assess("胸 口 痛").level == "red_flag"
    assert assess("高压１９０").level == "red_flag"


BLOOD_PRESSURE = [
    ("血压179/109", "none"),
    ("血压180/90", "red_flag"),
    ("血压150/110", "red_flag"),  # 3 级高血压：舒张压 ≥ 110
    ("血压150/120", "red_flag"),
    ("血压190/120", "red_flag"),
    ("血压190/120，头痛", "emergency"),
    ("190/120，头痛", "none"),  # 没有“血压”等字样，不当作血压
    ("低压125", "red_flag"),
    ("血压1900/120", "red_flag"),  # 收缩压位数不合理被忽略，舒张压仍达阈值
    ("血压1900/80", "none"),
]


@pytest.mark.parametrize(("text", "expect"), BLOOD_PRESSURE)
def test_blood_pressure(text: str, expect: str) -> None:
    assert assess(text).level == expect


def test_result_has_no_user_text() -> None:
    result = assess("我爸刚才跑步时说胸痛，然后就倒下了")
    assert set(Assessment.model_fields) == {"level", "rule_ids", "categories"}
    assert result.model_dump_json().isascii()


# ---- 词表校验 ----


def _book() -> dict[str, Any]:
    return {
        "version": "t",
        "review": "pending_review",
        "negation": {"before": ["没有"], "past_after": "过", "recurrence": ["又"]},
        "symptoms": [
            {"id": "chest_pain", "name": "胸痛", "terms": ["胸痛"]},
            {"id": "ongoing", "name": "持续", "combos": [[["还在"], ["疼", "痛"]]]},
        ],
        "rules": [
            {
                "id": "EM-CHEST",
                "level": "emergency",
                "category": "cardiovascular",
                "title": "胸痛持续",
                "all": ["chest_pain", "ongoing"],
            }
        ],
    }


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "terms.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_minimal_book_loads(tmp_path: Path) -> None:
    book = load_terms(_write(tmp_path, _book()))
    assert book.assess("胸痛，现在还在痛").level == "emergency"
    assert book.assess("胸痛").level == "none"


def _symptom(data: dict[str, Any], **changes: Any) -> dict[str, Any]:
    data["symptoms"][0].update(changes)
    return data


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["symptoms"].append(dict(d["symptoms"][0])), "编号重复"),
        (lambda d: d["rules"][0]["all"].append("fever"), "不存在的症状"),
        (
            lambda d: d["symptoms"].append({"id": "spare", "name": "多余", "terms": ["x"]}),
            "没有被任何条目使用",
        ),
        (lambda d: _symptom(d, terms=[""]), "terms"),
        (lambda d: _symptom(d, terms=["胸痛，胸闷"]), "统一后的写法"),
        (lambda d: _symptom(d, terms=["胸 痛"]), "统一后的写法"),
        (lambda d: _symptom(d, terms=["ＢＰ"]), "统一后的写法"),
        (lambda d: _symptom(d, terms=["Chest"]), "统一后的写法"),
        (lambda d: _symptom(d, terms=[]), "没有任何说法"),
        (lambda d: _symptom(d, combos=[[["胸口"]]]), "至少两组"),
        (lambda d: d["rules"][0].update(level="red_flag"), "前缀"),
        (lambda d: d["rules"][0].update(all=["chest_pain", "chest_pain", "ongoing"]), "重复引用"),
        (lambda d: d["rules"][0].update(extra=1), "extra"),
        (lambda d: d["negation"].update(before=["没 有"]), "统一后的写法"),
        (lambda d: d["negation"].update(before=[]), "before"),
        (lambda d: d.pop("negation"), "negation"),
    ],
)
def test_bad_book_rejected(tmp_path: Path, mutate: Any, message: str) -> None:
    data = _book()
    mutate(data)
    with pytest.raises(TermLoadError, match=message):
        load_terms(_write(tmp_path, data))


def test_duplicate_yaml_key_rejected(tmp_path: Path) -> None:
    path = tmp_path / "terms.yaml"
    path.write_text("version: a\nversion: b\n", encoding="utf-8")
    with pytest.raises(TermLoadError, match="重复的键"):
        load_terms(path)


def test_bundled_book_is_valid_and_reviewed() -> None:
    book = load_terms(TERMS_PATH)
    assert book.review == "approved"
    assert {r.level for r in book.rules} == {"emergency", "red_flag"}


def test_startup_fails_on_bad_book(tmp_path: Path) -> None:
    data = _book()
    data["rules"][0]["all"] = ["missing"]
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})
    with pytest.raises(TermLoadError):
        create_app(
            settings, FixedClock(NOW), log_stream=io.StringIO(), terms_path=_write(tmp_path, data)
        )


def test_bundled_book_round_trips_as_json() -> None:
    """词表能无损转成 JSON（D22 生成前端离线版的前提）。"""
    book = load_terms(TERMS_PATH)
    assert type(book).model_validate_json(json.dumps(book.model_dump(mode="json"))) == book
