"""对抗题库（spec [2026-10-20] §7）。问答与判定实现前全部为预期失败。"""

from pathlib import Path

import pytest

from tests.casefiles import (
    ADVERSARIAL_CATEGORIES,
    AdversarialCase,
    AdversarialFile,
    check_unique_ids,
    load_case_file,
    resolve,
)

FILE = load_case_file(Path(__file__).with_name("questions.yaml"), AdversarialFile)


def test_dataset_shape() -> None:
    check_unique_ids(c.id for c in FILE.cases)
    assert len(FILE.cases) == 30
    assert {c.category for c in FILE.cases} == ADVERSARIAL_CATEGORIES


def _judge(case: AdversarialCase, answer: object) -> None:
    raise NotImplementedError("回答判定方式随问答功能定义：必含要点 + 一票否决项")


@pytest.mark.parametrize("case", FILE.cases, ids=lambda c: c.id)
def test_adversarial(case: AdversarialCase, request: pytest.FixtureRequest) -> None:
    if FILE.status == "pending":
        request.applymarker(
            pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="问答未实现")
        )
    answer = resolve(FILE.target, FILE.status)(case.prompt)
    _judge(case, answer)
