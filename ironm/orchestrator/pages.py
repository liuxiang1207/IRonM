"""功能页提问与上下文（spec [2026-10-21] §3、§10）。

- 四个功能页：训练、饮食、恢复、指标及档案。
  功能页提问只调用本功能的专业 Agent 与安全官，不经管家调度。
- 本功能以外的问题（如在训练页问饮食），转入管家对话并带上上下文，不调用其他功能的 Agent。
- 上下文标识只含页面与条目编号（如 today.exercise:3.set:2），不含用户原话；
  它只用来补全含糊的问题（“这个动作”），不替代、也不会降低急症检查与拒答检查。
"""

from typing import Annotated, Literal

from pydantic import Field

from ironm.contracts.common import ContractModel
from ironm.contracts.roles import AgentId
from ironm.orchestrator.intent import Intent

__all__ = ["PAGE_AGENTS", "PAGE_DEFAULT", "PAGE_INTENTS", "Page", "PageContext"]

type Page = Literal["training", "diet", "recovery", "metrics"]

_A = AgentId
_I = Intent

# 每个功能页可以回答的意图 → 本功能的主责 Agent
PAGE_AGENTS: dict[Page, dict[Intent, AgentId]] = {
    "training": {
        _I.EXERCISE_TECHNIQUE: _A.EXERCISE_COACH,
        _I.TRAINING_KNOWLEDGE: _A.TRAINING_PLANNER,
        _I.PLAN: _A.TRAINING_PLANNER,
    },
    "diet": {
        _I.NUTRITION: _A.NUTRITIONIST,
        _I.SUPPLEMENTS_DRUGS: _A.NUTRITIONIST,
    },
    "recovery": {_I.RECOVERY_SLEEP: _A.RECOVERY},
    "metrics": {_I.PERSONAL_DATA: _A.DATA_ANALYST, _I.LOGGING: _A.DATA_ANALYST},
}
PAGE_INTENTS: dict[Page, frozenset[Intent]] = {p: frozenset(m) for p, m in PAGE_AGENTS.items()}

# 问题含糊（识别不出意图）时，按所在页面理解
PAGE_DEFAULT: dict[Page, Intent] = {
    "training": _I.EXERCISE_TECHNIQUE,
    "diet": _I.NUTRITION,
    "recovery": _I.RECOVERY_SLEEP,
    "metrics": _I.PERSONAL_DATA,
}


class PageContext(ContractModel):
    """从功能页带入的上下文：所在页面与条目编号。只含标识，不含用户原话。"""

    page: Page
    item: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.:\-]{0,79}$")] | None = None
