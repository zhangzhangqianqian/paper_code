from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.joint_dispatch.losses import weights_for_epoch  # noqa: E402


def test_decision_weight_is_positive_from_first_epoch_and_ramps():
    kwargs = dict(ramp_epochs=20, decision_start=0.05, decision_final=1.0, imitation_start=1.0, imitation_final=0.25)
    first = weights_for_epoch(epoch=0, **kwargs)
    middle = weights_for_epoch(epoch=10, **kwargs)
    final = weights_for_epoch(epoch=20, **kwargs)
    assert 0.0 < first.decision < middle.decision < final.decision
    assert first.imitation > middle.imitation > final.imitation
    assert final.decision == 1.0
    assert final.imitation == 0.25
