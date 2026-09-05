from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from src.joint_dispatch.formal_v4_4_contract import load_formal_v4_4_contract
from src.joint_dispatch.formal_v4_4_gate0 import run_gate0_v44, validate_gate0_receipt_v44
from src.joint_dispatch.formal_v4_4_artifacts import sha256_file
from src.joint_dispatch.formal_v4_4_provenance import build_source_manifest, write_source_manifest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_4.json"
REPORT = ROOT / "reports" / "joint_forecast_dispatch_formal_v4_2" / "formal_v4_2_20260905_g"
TRAIN = REPORT / "data" / "base_train.npz"
SELECTION = REPORT / "data" / "base_selection.npz"
BENCHMARK = REPORT / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
CAPACITY = REPORT / "gate0" / "CAPACITY_FREEZE.json"


@pytest.mark.timeout(180)
def test_gate0_measures_every_check_and_authorizes_pilot(tmp_path: Path) -> None:
    source = tmp_path / "SOURCE_MANIFEST.json"
    manifest = build_source_manifest(repo_root=ROOT, run_id="gate0_ok", contract_sha256=load_formal_v4_4_contract(CONTRACT).contract_sha256)
    write_source_manifest(source, manifest)
    receipt = run_gate0_v44(contract_path=CONTRACT, source_manifest=source, base_train_data=TRAIN, base_selection_data=SELECTION, benchmark=BENCHMARK, capacity_receipt=CAPACITY, output_root=tmp_path, run_id="gate0_ok")
    assert receipt.authorized_pilot
    assert receipt.checks["physical_residual"].measured_max <= 1.0e-6
    assert receipt.source_manifest_sha256 != "0" * 64
    validated = validate_gate0_receipt_v44(tmp_path / "gate0_ok" / "gate0" / "GATE0_RECEIPT.json", load_formal_v4_4_contract(CONTRACT), {"source_manifest_sha256": sha256_file(source)})
    assert validated.authorized_pilot


def test_gate0_rejects_forbidden_selection_year(tmp_path: Path) -> None:
    bad = tmp_path / "bad_selection.npz"
    import numpy as np
    with np.load(SELECTION, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["timestamps"] = arrays["timestamps"].copy(); arrays["timestamps"][0] = np.datetime64("2020-01-01")
    np.savez_compressed(bad, **arrays)
    source = tmp_path / "SOURCE_MANIFEST.json"
    manifest = build_source_manifest(repo_root=ROOT, run_id="gate0_bad", contract_sha256=load_formal_v4_4_contract(CONTRACT).contract_sha256)
    write_source_manifest(source, manifest)
    with pytest.raises(ValueError, match="2019|2020"):
        run_gate0_v44(contract_path=CONTRACT, source_manifest=source, base_train_data=TRAIN, base_selection_data=bad, benchmark=BENCHMARK, capacity_receipt=CAPACITY, output_root=tmp_path, run_id="gate0_bad")
