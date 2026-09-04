from __future__ import annotations

import numpy as np

from src.joint_dispatch.formal_v4_2_metrics import compute_v42_metrics
from src.joint_dispatch.formal_v4_2_rollout import PhysicalResidualsV42, SettledStepV42


def test_physical_metric_keeps_shortage_separate_from_balance_error():
    residuals = PhysicalResidualsV42(
        balance=np.zeros(3), capacity=np.zeros(21), conversion=np.zeros(5), soc=np.zeros(2),
        ramp=np.zeros(1), exclusivity=np.zeros(1), renewable_accounting=np.zeros(2), finite=np.zeros(1),
    )
    settled = np.zeros(21); settled[17] = 2.0
    outcome = SettledStepV42(
        planned=settled.copy(), settled=settled, shortage=np.array([2.0, 0.0, 0.0]), p_dump=4.0, q_dump=0.0,
        residuals=residuals, next_state=None, executed_plan_index=0, operating_cost=1.0, physical_carbon=2.0,
        penalized_objective=3.0,
    )
    metrics = compute_v42_metrics(outcome)
    assert metrics.shortage_energy.sum() == 2.0
    assert metrics.balance_residual_max == 0.0
    assert metrics.p_dump[0] == 4.0
