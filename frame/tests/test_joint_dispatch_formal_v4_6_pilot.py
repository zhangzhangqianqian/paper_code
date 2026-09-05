from __future__ import annotations

import json

import pytest

from scripts.run_rsc_pf_formal_v4_6_pilot import run_formal_v46_pilot


def test_pilot_requires_successful_diagnostic(tmp_path):
    diagnostic = tmp_path / "DIAGNOSTIC_RECEIPT.json"
    diagnostic.write_text(json.dumps({"diagnostic_authorized_pilot": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic"):
        run_formal_v46_pilot(
            config="configs/joint_forecast_dispatch_formal_v4_6.json",
            diagnostic_receipt=diagnostic, source_manifest=tmp_path / "manifest.json",
            materialized_root=tmp_path / "cache", train_data=tmp_path / "train.npz",
            benchmark=tmp_path / "benchmark.yaml", capacity_receipt=tmp_path / "capacity.json",
            output_root=tmp_path, run_id="v46-reject",
        )
