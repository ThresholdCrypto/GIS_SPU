# -*- coding: utf-8 -*-
"""两方执行抽象：WorldConfig / PartyManager（任务书 §16）。

纪律
====
"先抽象 Party → 再实现 2PC → 验证 2PC → 再考虑 3PC / NPC"。
本项目当前只有**两方**执行链路（见 capability.PSI_RUNTIME_WORLD_SIZE=2）。
SPU 枚举里存在 ``PROTOCOL_ECDH_3PC`` 不等于本编译器已具备三方 PSI——
本模块对超过两方的登记**显式拒绝**，不静默接受、不假装支持。

PartyManager 只做三件事：
1. 把 ``PartyInput(party_id=...)`` 绑定的数据源去重成"参与方"；
2. 按**首次出现顺序**给参与方分配 rank（确定性，供审计与结果披露）；
3. 把"哪份输入属于谁"输出成可写进 JSON 的快照（不改变两方装配方式：
   left / right 仍由算子输入顺序决定）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .input_adapter import ResolvedCellInput

#: 本阶段支持的参与方数量（与 capability.PSI_RUNTIME_WORLD_SIZE 一致）
SUPPORTED_WORLD_SIZE = 2


@dataclass(frozen=True)
class WorldConfig:
    """执行世界配置。当前只承认两方；其它数量在构造/校验时显式失败。"""

    world_size: int = SUPPORTED_WORLD_SIZE

    def validate(self) -> None:
        if self.world_size != SUPPORTED_WORLD_SIZE:
            raise ValueError(
                f"当前仅支持 {SUPPORTED_WORLD_SIZE} 方执行（§16）；"
                f"world_size={self.world_size} 未实现——三方 / 多方 PSI 需要"
                "独立设计（含 N 方链路与协议矩阵），不得据此宣称已支持"
            )


@dataclass(frozen=True)
class PartyDescriptor:
    """一个参与方：谁（party_id）、rank、提供了哪些输入。"""

    party_id: str
    rank: int
    inputs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "party_id": self.party_id,
            "rank": self.rank,
            "inputs": list(self.inputs),
        }


class PartyManager:
    """两方绑定的登记处（确定性；不改变 left/right 装配）。"""

    def __init__(self, world: WorldConfig | None = None) -> None:
        self.world = world or WorldConfig()
        self.world.validate()
        self._parties: dict[str, list[str]] = {}

    # ---------------- 登记 ----------------

    def bind(self, input_name: str, party_id: str) -> int:
        """把一份输入登记到某个参与方；返回该参与方的 rank。

        参与方按首次登记顺序获得 rank。超过两方直接拒绝（不静默截断）。
        """

        party_id = str(party_id).strip()
        if not party_id:
            raise ValueError("party_id 不能为空")
        if party_id in self._parties:
            self._parties[party_id].append(input_name)
            rank = self.rank_of(party_id)
            return 0 if rank is None else rank
        if len(self._parties) >= self.world.world_size:
            raise ValueError(
                f"已登记 {len(self._parties)} 方（{tuple(self._parties)}），"
                f"当前仅支持 {self.world.world_size} 方执行（§16）；"
                "三方 / 多方 PSI 未实现，不得据此宣称已支持"
            )
        self._parties[party_id] = [input_name]
        return len(self._parties) - 1

    def rank_of(self, party_id: str) -> int | None:
        keys = list(self._parties)
        return keys.index(party_id) if party_id in self._parties else None

    def descriptors(self) -> tuple[PartyDescriptor, ...]:
        return tuple(
            PartyDescriptor(party_id, rank, tuple(inputs))
            for rank, (party_id, inputs) in enumerate(self._parties.items())
        )

    # ---------------- 快照 ----------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world.world_size,
            "parties": [item.to_dict() for item in self.descriptors()],
        }

    @classmethod
    def from_inputs(
        cls, resolved: Mapping[str, ResolvedCellInput], world: WorldConfig | None = None
    ) -> "PartyManager":
        """从已规范化的输入构建：只登记带 party_id 的输入（未标注的忽略）。"""

        manager = cls(world)
        for name, value in resolved.items():
            if value.party_id:
                manager.bind(str(name), value.party_id)
        return manager


def same_party_both_sides(left: str | None, right: str | None) -> bool:
    """两侧输入是否明确来自同一参与方（用于诚实披露，不是错误判定）。"""

    return bool(left) and left == right
