"""动作讲解（spec [2026-10-21] 统一链路）：讲解动作库里的一个动作，附常见错误与替代。

内容全部来自动作库，不另编；找不到动作时如实说明。
"""

import re
from typing import Any

from ironm.contracts.agents import AgentOutput, Citation, Opinion, make_opinion
from ironm.contracts.exercises import Equipment, Restriction
from ironm.knowledge.exercises import Library, default_library
from ironm.skills.loader import SkillContext
from ironm.skills.outputs import no_opinion
from ironm.skills.view import SnapshotView

LIBRARY_REF = "KB-EXERCISES"  # 动作库（knowledge/exercises.yaml）

# 从问题里找动作名：取动作库里最长的、出现在问题中的名称
_MIN_NAME = 2


def _match(lib: Library, text: str) -> tuple[str, ...]:
    """问题里出现的动作名对应的编号，按名称长度从长到短。"""
    hits = [
        (len(e.name), e.id) for e in lib.exercises if len(e.name) >= _MIN_NAME and e.name in text
    ]
    return tuple(eid for _, eid in sorted(hits, reverse=True))


def _environment(view: SnapshotView) -> frozenset[Equipment]:
    env = view.get("profile.environment")
    lib = default_library()
    if isinstance(env, str) and env in ("hotel", "home", "gym"):
        return lib.environment(env)  # type: ignore[arg-type]
    return lib.environment("hotel")  # 不知道环境时按最少器材处理


def _restrictions(view: SnapshotView) -> tuple[Restriction, ...]:
    raw = view.get("profile.restrictions")
    if not isinstance(raw, list | tuple):
        return ()
    known = {r.value for r in Restriction}
    return tuple(Restriction(r) for r in raw if isinstance(r, str) and r in known)


def _opinion(
    context: SkillContext, lib: Library, exercise_id: str, view: SnapshotView
) -> AgentOutput:
    ex = lib.get(exercise_id)
    restrictions = _restrictions(view)
    equipment = _environment(view)
    cautions = [lib.restrictions[r].caution for r in ex.caution_if if r in restrictions]

    kind = "太简单" if "太简单" in str(_last_text(context)) else "做不了"
    if kind == "做不了" and ex.easier is not None:
        alt_id, alt_label = ex.easier, "做不了时先降阶"
    elif kind == "太简单" and ex.harder is not None:
        alt_id, alt_label = ex.harder, "太简单时进阶"
    else:
        alt_id = None
        alt_label = ""
    alternative: list[str] = []
    if alt_id is not None:
        alt = lib.get(alt_id)
        alternative = [f"{alt_label}：{alt.name}。{alt.key_points[0]}"]
    else:
        subs = lib.substitutes(exercise_id, equipment=equipment, restrictions=restrictions)
        if subs:
            alternative = [f"可以换成：{subs[0].name}"]

    citations = [
        Citation(
            kind="knowledge",
            ref=f"{LIBRARY_REF}:{ex.id}",
            label=f"动作库《{ex.name}》（编号 {ex.id}）",
            detail="；".join(ex.key_points),
        )
    ]
    for r in ex.caution_if:
        if r in restrictions:
            citations.append(
                Citation(
                    kind="rule",
                    ref=f"{LIBRARY_REF}:restrictions.{r.value}",
                    label=lib.restrictions[r].name,
                    detail=lib.restrictions[r].caution,
                )
            )
    return make_opinion(
        context,
        Opinion(
            conclusion=f"「{ex.name}」是练{_pattern_word(ex.pattern)}的动作。",
            citations=tuple(citations),
            reasoning=f"按动作库中《{ex.name}》这一条给出要点与常见错误。",
            how="；".join([*ex.key_points, *alternative]),
            caution="；".join([*ex.common_errors, *cautions]) or None,
            uncertain=(
                "我按动作库中的这一条讲解，没有看你的实际动作；"
                "如果想确认做得对不对，可以描述一下你的做法或上传视频。"
            ),
        ),
    )


def _last_text(context: SkillContext) -> Any:
    return getattr(context, "text", "")


def _pattern_word(pattern: Any) -> str:
    return {
        "squat": "下肢蹲",
        "hinge": "髋铰链",
        "lunge": "单腿",
        "push": "上肢推",
        "pull": "上肢拉",
        "core": "核心",
        "cardio": "心肺",
    }.get(str(pattern), str(pattern))


def run(view: SnapshotView, context: SkillContext) -> AgentOutput:
    lib = default_library()
    text = str(getattr(context, "text", "") or "")
    item = str(getattr(context, "context_item", "") or "")
    # 带条目编号进入（如从动作页点进来）：编号形如 exercise:sq_goblet
    if m := re.search(r"exercise:([a-z][a-z0-9_]*)", item):
        try:
            lib.get(m.group(1))
        except KeyError:
            return no_opinion(context, "unknown_exercise")
        return _opinion(context, lib, m.group(1), view)
    matched = _match(lib, text)
    if not matched:
        return no_opinion(context, "unknown_exercise")
    return _opinion(context, lib, matched[0], view)
