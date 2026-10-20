"""服务端标识生成。生产用随机 UUID；测试与模拟用固定种子，使标识序列可复现。"""

import random
import uuid
from typing import Protocol


class IdGenerator(Protocol):
    def new_id(self) -> str: ...


class RandomIdGenerator:
    def new_id(self) -> str:
        return str(uuid.uuid4())


class SeededIdGenerator:
    """同一种子产生同一序列。只用于测试与模拟，不具备密码学随机性。"""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def new_id(self) -> str:
        return str(uuid.UUID(int=self._rng.getrandbits(128), version=4))
