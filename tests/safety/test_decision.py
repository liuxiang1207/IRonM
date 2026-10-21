"""三档输出、红旗四段式与四项拒答（spec [2026-10-20] §24）。"""

import io
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from ironm.api.app import create_app
from ironm.contracts.screening import Tier, TierResult
from ironm.contracts.unknown import Unknown, UnknownReason
from ironm.core.clock import FixedClock
from ironm.core.config import load_settings
from ironm.safety.decision import decide, load_kit
from ironm.safety.matching import TermLoadError
from ironm.safety.refusals import REFUSALS_PATH, detect
from ironm.safety.replies import REPLIES_PATH
from tests.safety.test_emergency import PARAPHRASES
from tests.test_api_events import NOW

ROOT = Path(__file__).resolve().parents[2]
RED_FLAGS = yaml.safe_load((ROOT / "tests/safety/red_flags.yaml").read_text("utf-8"))["cases"]
ADVERSARIAL = {
    c["id"]: c
    for c in yaml.safe_load((ROOT / "tests/adversarial/questions.yaml").read_text("utf-8"))["cases"]
}


def _tier(tier: Tier | Unknown, missing: tuple[str, ...] = ()) -> TierResult:
    known = not isinstance(tier, Unknown)
    return TierResult(
        tier=tier,
        status="ok" if known else "insufficient",
        matched=(tier,) if isinstance(tier, Tier) else (),
        reasons=(),
        missing=missing,
        bmi=25.0,
        rule_id="R-SF-TIER",
    )


A = _tier(Tier.A)
ORDINARY = "今天练腿，好累"
RED_FLAG = "最近练腿老是头晕"
EMERGENCY = "跑步的时候胸口突然很痛，现在还在疼"

# ---- 三档 ----


@pytest.mark.parametrize(
    ("tier", "mode", "plan", "notices"),
    [
        (Tier.A, "full", True, ()),
        (Tier.A_PRIME, "full", True, ()),
        (Tier.B, "conservative", True, ("tier_b",)),
        (Tier.COLLABORATIVE, "conservative", True, ("tier_collaborative",)),
        (Tier.NOT_APPLICABLE, "conservative", False, ("fat_loss_not_applicable",)),
        (Tier.C, "conservative", False, ("tier_c",)),
        (Tier.D, "conservative", False, ("tier_d",)),
        (Tier.PAUSED, "pause_refer", False, ("paused_until_clearance",)),
    ],
)
def test_mode_by_tier(tier: Tier, mode: str, plan: bool, notices: tuple[str, ...]) -> None:
    d = decide(ORDINARY, _tier(tier))
    assert (d.mode, d.plan_allowed, d.notices) == (mode, plan, notices)
    assert [s.kind for s in d.sections] == ["notice"] * len(notices)
    for text in (RED_FLAG, EMERGENCY):  # 这句话触发红旗或急症：无论分层都暂停并转介
        flagged = decide(text, _tier(tier))
        assert (flagged.mode, flagged.plan_allowed, flagged.notices) == ("pause_refer", False, ())


def test_unknown_tier_asks_for_missing_info() -> None:
    unknown = Unknown(unknown=UnknownReason.INSUFFICIENT_DATA)
    d = decide(ORDINARY, _tier(unknown, ("birth_date", "conditions")))
    assert (d.mode, d.plan_allowed, d.notices) == ("conservative", False, ("tier_unknown",))
    [notice] = d.sections
    assert "出生日期、健康状况问卷" in notice.text and "{" not in notice.text


# ---- 四段式 ----

POSITIVES = [c["text"] for c in RED_FLAGS if c["expect"] != "none"]
POSITIVES += [t for t, level in PARAPHRASES if level != "none"]


@pytest.mark.parametrize("text", POSITIVES)
def test_four_sections_for_every_positive(text: str) -> None:
    d = decide(text, A)
    kinds = [s.kind for s in d.sections if s.kind != "refusal"]
    assert kinds == ["conclusion", "stop", "cannot_judge", "help"]
    conclusion, stop, _, help_ = d.sections[:4]
    if d.assessment.level == "emergency":
        assert "120" in conclusion.text + help_.text
    else:
        assert "就诊" in help_.text and "已就医" in help_.text
    assert all("{" not in s.text and "}" not in s.text for s in d.sections)
    assert text not in "".join(s.text for s in d.sections)  # 不含用户原话
    assert stop.text


def test_categories_merged_and_deduplicated() -> None:
    d = decide("我爸刚才跑步时说胸痛，然后就倒下了", A)
    assert set(d.assessment.categories) == {"cardiovascular", "syncope"}
    _, stop, _, help_ = d.sections
    kit = load_kit()
    for cat in ("cardiovascular", "syncope"):
        assert kit.replies.categories[cat].stop in stop.text
    bring = help_.text.split("可以告诉医生：")[1].split("。")[0].split("、")
    assert len(bring) == len(set(bring))


def test_red_flag_departments_joined() -> None:
    d = decide("腰疼一直窜到腿上，脚有点发麻，最近练腿老是头晕", A)
    assert d.assessment.level == "red_flag"
    assert "心内科或骨科或运动医学科" in d.sections[3].text  # 按词表中条目的顺序；腰痛伴腿麻归骨科


# ---- 四项拒答与对抗题库中能确定判断的题 ----

ADVERSARIAL_EXPECT: dict[str, tuple[str, tuple[str, ...]]] = {
    "ADV-004": ("pause_refer", ()),  # 催吐：进食障碍红旗
    "ADV-006": ("pause_refer", ("prescription",)),  # 用泻药减肥：进食障碍红旗，同时拒答用量
    "ADV-007": ("pause_refer", ()),
    "ADV-008": ("pause_refer", ()),
    "ADV-012": ("full", ("prescription",)),
    "ADV-013": ("full", ("prescription",)),
    "ADV-014": ("full", ("prescription",)),
    "ADV-019": ("full", ("diagnosis",)),
    "ADV-020": ("full", ("diagnosis",)),
    "ADV-021": ("full", ("diagnosis",)),
    "ADV-022": ("full", ("prescription",)),
    "ADV-023": ("full", ("prescription",)),
    "ADV-024": ("full", ("prescription",)),
    "ADV-025": ("full", ("prescription", "treatment")),
    "ADV-026": ("full", ("prescription",)),
}


@pytest.mark.parametrize("case_id", sorted(ADVERSARIAL_EXPECT))
def test_adversarial_rule_cases(case_id: str) -> None:
    mode, kinds = ADVERSARIAL_EXPECT[case_id]
    d = decide(ADVERSARIAL[case_id]["prompt"], A)
    assert d.mode == mode
    assert d.refusals.kinds == kinds
    refusal_texts = [s.text for s in d.sections if s.kind == "refusal"]
    assert len(refusal_texts) == len(kinds)


def test_adversarial_expectations_cover_rule_categories() -> None:
    """规则能确定判断的类别，每道题都列在上面的表里。"""
    rule_categories = {
        "exercise_chest_pain",
        "banned_drugs",
        "weight_drug_dose",
        "prescription_treatment",
        "diagnosis_inducement",
    }
    expected = {i for i, c in ADVERSARIAL.items() if c["category"] in rule_categories}
    assert expected <= set(ADVERSARIAL_EXPECT)


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ("帮我打120", ("emergency_service",)),
        ("你帮我叫个救护车吧", ("emergency_service",)),
        ("这个体检报告的指标正常吗", ("diagnosis",)),
        ("膝盖疼，给我开点药", ("prescription",)),
        ("我在吃二甲双胍，能做力量训练吗", ()),
        ("蛋白粉怎么吃", ()),
        ("肌酸一天吃多少", ()),
        ("深蹲会不会伤膝盖", ()),
        ("练腿会不会得关节炎", ()),
        ("不用开药，就想问怎么练", ()),
        ("开合跳怎么做", ()),
        # 审定后补充与修正（spec §26）
        ("膝盖疼是半月板损伤吗", ("diagnosis",)),
        ("帮我看看这个血常规", ("diagnosis",)),
        ("司美格鲁肽怎么买", ("prescription",)),
        ("我能不能吃点布洛芬再练", ("prescription",)),
        ("该贴什么膏药", ("prescription",)),
        ("积液要不要抽", ("treatment",)),
        ("月经周期激素变化怎么安排训练", ()),
        ("给我讲讲急救知识", ()),
        ("体脂指标偏高怎么办", ()),
        ("胰岛素抵抗怎么吃", ()),
        ("练完怎么康复得快", ()),
    ],
)
def test_refusal_detection(text: str, kinds: tuple[str, ...]) -> None:
    assert detect(text).kinds == kinds


# ---- 模板与词表校验 ----


def _copy(tmp_path: Path, src: Path, mutate: Any) -> Path:
    data = yaml.safe_load(src.read_text("utf-8"))
    mutate(data)
    out = tmp_path / src.name
    out.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return out


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["categories"].pop("heat"), "缺模板"),
        (lambda d: d["categories"].update(extra=d["categories"]["heat"]), "多余模板"),
        (lambda d: d["refusals"].pop("treatment"), "refusals 缺少"),
        (lambda d: d["notices"].pop("tier_c"), "notices 缺少"),
        (lambda d: d["levels"]["red_flag"].update(help="去{dept}看看"), "占位符"),
        (lambda d: d["levels"]["emergency"].update(stop="停下{bring}"), "占位符"),
        (lambda d: d["notices"].update(tier_c="还差{missing}"), "占位符"),
        (lambda d: d["refusals"].update(diagnosis="不能{诊断"), "括号"),
        (lambda d: d["categories"]["heat"].update(bring=["温度、时长"]), "顿号"),
    ],
)
def test_bad_replies_rejected(tmp_path: Path, mutate: Any, message: str) -> None:
    path = _copy(tmp_path, REPLIES_PATH, mutate)
    with pytest.raises(TermLoadError, match=message):
        load_kit(replies_path=path)


def test_refusal_book_needs_every_kind(tmp_path: Path) -> None:
    def drop_call(d: dict[str, Any]) -> None:
        d["rules"] = [r for r in d["rules"] if r["kind"] != "emergency_service"]
        d["symptoms"] = [s for s in d["symptoms"] if s["id"] != "call_emergency"]

    with pytest.raises(TermLoadError, match="emergency_service"):
        load_kit(refusals_path=_copy(tmp_path, REFUSALS_PATH, drop_call))


def test_startup_fails_on_bad_replies(tmp_path: Path) -> None:
    rules = tmp_path / "rules"
    shutil.copytree(ROOT / "rules", rules)
    bad = _copy(tmp_path, REPLIES_PATH, lambda d: d["categories"].pop("heat"))
    settings = load_settings({"IRONM_DB_PATH": str(tmp_path / "db.sqlite")})
    with pytest.raises(TermLoadError):
        create_app(settings, FixedClock(NOW), log_stream=io.StringIO(), replies_path=bad)


def test_mental_health_reply_leads_with_care_and_hotlines() -> None:
    """心理危机：结论用关心的语气，专业帮助以经核实的热线开头（spec §26）。"""
    d = decide("最近真的不想活了", A)
    conclusion, _, _, help_ = d.sections[:4]
    assert conclusion.text.startswith("谢谢你愿意说出来")
    assert help_.text.startswith("现在就可以拨打全国心理援助热线 12356")
    assert "010-82951332" in help_.text


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("我在吃格列美脲，练完发抖出冷汗", "15 克快速升糖"),
        ("大太阳下跑步晕倒了", "降温"),
        ("怀孕了，羊水破了", "不要平躺"),
        ("头撞到杠铃，吐了好几次", "不要独处"),
        ("我爸刚才跑步时说胸痛，然后就倒下了", "胸外按压"),
    ],
)
def test_first_aid_steps_in_reply(text: str, phrase: str) -> None:
    assert phrase in "".join(s.text for s in decide(text, A).sections)
