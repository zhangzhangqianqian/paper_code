from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.heat_pump.frontier import frontier_contract_hash, load_frontier_contract


CONTRACT = ROOT / "configs" / "heat_pump_frontier_contract_v1.json"


def test_frontier_contract_is_validation_only():
    contract = load_frontier_contract(CONTRACT)
    assert contract.split == "validation"
    assert contract.test_set_accessed is False
    assert contract.epsilon_cost_tolerances == (0.0, 0.01, 0.03, 0.05)
    assert contract.scenario_generation["test_calibration"] is False


def test_frontier_dry_run_does_not_create_results(tmp_path: Path):
    output = tmp_path / "dry"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_heat_pump_frontier.py"), "--benchmark", "D:/Paper/standard_ies_benchmark_v1.yaml", "--contract", str(CONTRACT), "--output-dir", str(output), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["test_set_accessed"] is False
    assert not output.exists()


def test_frontier_verify_only_checks_manifest_and_contract(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    contract_hash = frontier_contract_hash(CONTRACT)
    (output / "frontier_summary.json").write_text(
        json.dumps({
            "status": "complete",
            "test_set_accessed": False,
            "contract_sha256": contract_hash,
            "decision_status": "failed_no_identifiable_heat_pump_frontier",
            "total_records": 0,
        }),
        encoding="utf-8",
    )
    (output / "frontier_manifest.json").write_text(
        json.dumps({"status": "complete", "test_set_accessed": False, "records": 0}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_heat_pump_frontier.py"),
            "--benchmark", "D:/Paper/standard_ies_benchmark_v1.yaml",
            "--contract", str(CONTRACT),
            "--output-dir", str(output),
            "--verify-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "verified"
    assert payload["checks"]["contract_hash_matches"] is True
