from __future__ import annotations

from pathlib import Path

from scripts.run_rsc_pf_formal_v4_5_diagnostic import run_formal_v45_diagnostic


def test_diagnostic_never_opens_selection_or_evaluation_year(tmp_path: Path) -> None:
    receipt = run_formal_v45_diagnostic(
        config="configs/joint_forecast_dispatch_formal_v4_5.json",
        output_root=tmp_path, max_batches=2,
    )
    assert receipt["accessed_years"] == [2015, 2016, 2017, 2018]
    assert receipt["pilot_authorized"] is False
    assert receipt["selection_year_accessed"] is False
    assert receipt["evaluation_year_accessed"] is False
