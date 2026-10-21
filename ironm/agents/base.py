"""Agent（spec §28、§29、§32）。

Agent 只是“专业身份 + 权限 + 契约”，具体能力全部来自本次加载的 Skill。

每个加载的 Skill 各出一份输出，各自只能读自己声明的数据；新增能力只加 Skill、不改 Agent。
运行时把关：提案字段必须在该 Skill 声明的产出之内，非处方类 Skill 不能产出提案。
单个 Skill 出错（包括越过上述把关）时只降级为该 Skill“本次无意见”，不影响其他 Skill；
严格模式（测试用）下照常抛出，便于发现问题。
"""

from ironm.contracts.agents import AgentOutput, NoOpinionOutput, ProposalOutput
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.snapshot import Snapshot
from ironm.core.log import get_logger
from ironm.rules.loader import RuleBook
from ironm.skills.loader import LoadedSkill, Selection, SkillContext
from ironm.skills.view import SnapshotView

__all__ = ["NO_SKILL", "Agent", "AgentContractError"]

NO_SKILL = "no_skill@0.0.0"  # 没有加载任何 Skill 时“本次无意见”的来源
log = get_logger("ironm.agents")


class AgentContractError(ValueError):
    pass


class Agent:
    def __init__(self, agent_id: AgentId, *, strict: bool = False) -> None:
        self.agent_id = agent_id
        self.strict = strict

    def execute(
        self,
        snapshot: Snapshot,
        selection: Selection,
        rulebook: RuleBook,
        *,
        text: str = "",
        context_item: str | None = None,
    ) -> tuple[AgentOutput, ...]:
        """逐个运行本次加载的 Skill；没有 Skill 时只返回一份“本次无意见”。"""
        if selection.agent != self.agent_id:
            raise AgentContractError(f"选择结果属于 {selection.agent}，不是 {self.agent_id}")
        if not selection.loaded:
            return (self._no_opinion(NO_SKILL, "no_matching_skill"),)
        return tuple(
            self._run_one(snapshot, skill, rulebook, text, context_item)
            for skill in selection.loaded
        )

    def _run_one(
        self,
        snapshot: Snapshot,
        skill: LoadedSkill,
        rulebook: RuleBook,
        text: str,
        context_item: str | None,
    ) -> AgentOutput:
        m = skill.manifest
        try:
            view = SnapshotView(snapshot, m.requires_data)
            output = skill.run(
                view,
                SkillContext(skill=skill, rulebook=rulebook, text=text, context_item=context_item),
            )
            self._check(output, skill)
            return output
        except Exception as e:
            if self.strict:
                raise
            log.exception("agents.skill_error", e, skill_id=f"{m.id}:{m.version}")
            return self._no_opinion(m.versioned, f"skill_error:{type(e).__name__}")

    def _check(self, output: AgentOutput, skill: LoadedSkill) -> None:
        m = skill.manifest
        if output.agent != self.agent_id:
            raise AgentContractError(f"{self.agent_id} 的输出标成了 {output.agent}")
        if output.skill != m.versioned:
            raise AgentContractError(f"{m.versioned} 的输出标成了别的 Skill 版本：{output.skill}")
        if isinstance(output, ProposalOutput):
            # 契约已保证非处方类的产出为空（下一条检查也会拦下）；这里再兜一层，防止契约规则被改动
            if not m.prescriptive:
                raise AgentContractError(f"非处方类 Skill {m.versioned} 不能产出提案")
            undeclared = {p.field for p in output.items} - set(m.produces)
            if undeclared:
                raise AgentContractError(f"{m.versioned} 产出了未声明的字段：{sorted(undeclared)}")

    def _no_opinion(self, skill: str, reason: str) -> NoOpinionOutput:
        return NoOpinionOutput(
            agent=self.agent_id, skill=skill, kind=OutputKind.NO_OPINION, reason_code=reason
        )
