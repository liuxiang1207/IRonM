from collections.abc import Iterator

import pytest

from tests.memory_helpers import close_all


@pytest.fixture(autouse=True)
def _close_memory_dbs() -> Iterator[None]:
    yield
    close_all()
