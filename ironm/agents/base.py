"""Agent（spec §28、§29）。Agent 只是“专业身份 + 权限 + 契约”，具体能力全部来自本次加载的 Skill。

每个加载的 Skill 各出一份输出，各自只能读自己声明的数据；新增能力只加 Skill、不改 Agent。
"""

from ironm.contracts.agents import AgentOutput, NoOpinionOutput
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.snapshot import Snapshot
from ironm.rules.loader import RuleBook
from ironm.skills.loader import Selection, SkillContext
from ironm.skills.view import SnapshotView

__all__ = ["NO_SKILL", "Agent", "AgentContractError"]

NO_SKILL = "no_skill@0.0.0"  # 没有加载任何 Skill 时“本次无意见”的来源


class AgentContractError(ValueError):
    pass


class Agent:
    def __init__(self, agent_id: AgentId) -> None:
        self.agent_id = agent_id

    def execute(
        self, snapshot: Snapshot, selection: Selection, rulebook: RuleBook
    ) -> tuple[AgentOutput, ...]:
        """逐个运行本次加载的 Skill；没有 Skill 时只返回一份“本次无意见”。"""
        if selection.agent != self.agent_id:
            raise AgentContractError(f"选择结果属于 {selection.agent}，不是 {self.agent_id}")
        if not selection.loaded:
            return (
                NoOpinionOutput(
                    agent=self.agent_id,
                    skill=NO_SKILL,
                    kind=OutputKind.NO_OPINION,
                    reason_code="no_matching_skill",
                ),
            )
        outputs: list[AgentOutput] = []
        for skill in selection.loaded:
            view = SnapshotView(snapshot, skill.manifest.requires_data)
            output = skill.run(view, SkillContext(skill=skill, rulebook=rulebook))
            if output.agent != self.agent_id:
                raise AgentContractError(f"{self.agent_id} 的输出标成了 {output.agent}")
            if output.skill != skill.manifest.versioned:
                raise AgentContractError(
                    f"{skill.manifest.versioned} 的输出标成了别的 Skill 版本：{output.skill}"
                )
            outputs.append(output)
        return tuple(outputs)
