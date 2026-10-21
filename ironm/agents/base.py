"""Agent 基类（spec §28）。Agent 只是“专业身份 + 权限 + 契约”，具体能力来自本次加载的 Skill。"""

from abc import ABC, abstractmethod
from typing import ClassVar

from ironm.contracts.agents import AgentOutput, NoOpinionOutput
from ironm.contracts.roles import AgentId, OutputKind
from ironm.contracts.snapshot import Snapshot
from ironm.skills.loader import LoadedSkill, Selection
from ironm.skills.view import SnapshotView

__all__ = ["NO_SKILL", "Agent", "AgentContractError"]

NO_SKILL = "no_skill@0.0.0"  # 没有加载任何 Skill 时“本次无意见”的来源


class AgentContractError(ValueError):
    pass


class Agent(ABC):
    agent_id: ClassVar[AgentId]

    def execute(self, snapshot: Snapshot, selection: Selection) -> AgentOutput:
        """用本次加载的 Skill 运行；没有 Skill 时直接返回“本次无意见”，不调用具体实现。"""
        if selection.agent != self.agent_id:
            raise AgentContractError(f"选择结果属于 {selection.agent}，不是 {self.agent_id}")
        if not selection.loaded:
            return NoOpinionOutput(
                agent=self.agent_id,
                skill=NO_SKILL,
                kind=OutputKind.NO_OPINION,
                reason_code="no_matching_skill",
            )
        view = SnapshotView(snapshot, selection.requires_data)
        output = self.run(view, selection.loaded)
        if output.agent != self.agent_id:
            raise AgentContractError(f"{self.agent_id} 的输出标成了 {output.agent}")
        if output.skill not in selection.versions:
            raise AgentContractError(f"输出来自未加载的 Skill 版本：{output.skill}")
        return output

    @abstractmethod
    def run(self, view: SnapshotView, skills: tuple[LoadedSkill, ...]) -> AgentOutput:
        """具体实现：只能通过 view 读取数据，输出须来自 skills 中的某个版本。"""
