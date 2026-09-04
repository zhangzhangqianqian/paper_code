from __future__ import annotations

import pytest

from src.joint_dispatch.formal_v4_gate0 import MANDATORY_CHECK_IDS, evaluate_gate0, run_gate0


class Fixture:
    def __init__(self):
        self.failures: set[str] = set()


@pytest.mark.parametrize("failure", MANDATORY_CHECK_IDS)
def test_gate0_refuses_any_mandatory_failure(failure):
    fixture = Fixture()
    fixture.failures.add(failure)
    result = run_gate0(fixture)
    assert result.authorized_gate1 is False
    assert result.checks[failure].passed is False


def test_gate0_authorizes_clean_fixture():
    result = run_gate0(Fixture())
    assert result.authorized_gate1 is True
    assert result.failures == ()


def test_gate0_rejects_missing_or_unknown_check_ids():
    with pytest.raises(ValueError, match="registry mismatch"):
        evaluate_gate0({"only": lambda: True}, expected_check_ids=("required",))
