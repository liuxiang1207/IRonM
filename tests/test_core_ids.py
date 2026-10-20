"""标识生成（spec [2026-10-20] §2）。"""

import uuid

from ironm.core.ids import RandomIdGenerator, SeededIdGenerator


def test_random_ids_unique() -> None:
    gen = RandomIdGenerator()
    ids = [gen.new_id() for _ in range(10_000)]
    assert len(set(ids)) == len(ids)


def test_ids_are_uuid_strings() -> None:
    for gen in (RandomIdGenerator(), SeededIdGenerator(1)):
        new_id = gen.new_id()
        assert str(uuid.UUID(new_id)) == new_id


def test_same_seed_same_sequence() -> None:
    a, b = SeededIdGenerator(42), SeededIdGenerator(42)
    assert [a.new_id() for _ in range(100)] == [b.new_id() for _ in range(100)]


def test_different_seed_different_sequence() -> None:
    a, b = SeededIdGenerator(1), SeededIdGenerator(2)
    assert [a.new_id() for _ in range(10)] != [b.new_id() for _ in range(10)]


def test_seeded_ids_unique() -> None:
    gen = SeededIdGenerator(7)
    ids = [gen.new_id() for _ in range(10_000)]
    assert len(set(ids)) == len(ids)
