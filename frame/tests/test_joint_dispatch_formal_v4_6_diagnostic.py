from __future__ import annotations

from scripts.run_rsc_pf_formal_v4_6_diagnostic import run_formal_v46_diagnostic


def test_diagnostic_never_opens_2019_or_2020(tmp_path):
    receipt = run_formal_v46_diagnostic(
        config="configs/joint_forecast_dispatch_formal_v4_6.json",
        output_root=tmp_path, run_id="v46-test-diagnostic", max_batches=1, epoch_cap=2,
    )
    assert receipt["accessed_years"] == [2015, 2016, 2017, 2018]
    assert receipt["selection_year_accessed"] is False
    assert receipt["evaluation_year_accessed"] is False
    assert receipt["joint_gradient_norms"]["decision_to_risk"] > 0
    assert receipt["decoupled_gradient_norms"]["decision_to_base"] <= 1e-12
    assert receipt["diagnostic_authorized_pilot"] is True
