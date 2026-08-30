from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_joint_forecast_dispatch import main


def test_dry_run_does_not_require_data_arrays(tmp_path: Path):
    output = tmp_path / "dry-run"
    assert main(["--stage", "dry-run", "--output-dir", str(output)]) == 0
    receipt = json.loads((output / "dry_run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["online_exact_lp_calls"] == 0
    assert receipt["complete"] is False


def test_smoke_requires_limit(tmp_path: Path):
    with pytest.raises(SystemExit, match="smoke stage requires"):
        main(["--stage", "smoke", "--output-dir", str(tmp_path / "smoke")])


def test_test_stage_requires_selection_receipt(tmp_path: Path):
    with pytest.raises(SystemExit, match="requires frozen selection receipt"):
        main(["--stage", "test", "--output-dir", str(tmp_path / "test")])


def test_non_empty_output_is_never_overwritten(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--stage", "dry-run", "--output-dir", str(output)])
