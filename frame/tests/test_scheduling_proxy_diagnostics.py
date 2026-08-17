from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")
CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v1.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_smoke_artifact(tmp_path: Path) -> Path:
    from scripts.run_scheduling_proxy_pipeline import run_pipeline

    output = tmp_path / "proxy_smoke"
    result = run_pipeline("smoke", BENCHMARK, CONTRACT, output)
    assert result["status"] == "passed"
    return output


def test_audit_accepts_a_valid_proxy_artifact(tmp_path):
    from scripts.audit_scheduling_proxy import audit_proxy_artifact

    output = _make_smoke_artifact(tmp_path)
    report = audit_proxy_artifact(output, BENCHMARK, CONTRACT)
    assert report["status"] == "passed"
    assert report["mode"] == "smoke"
    assert report["test_split_used_for_selection"] is False
    assert report["test_output_shape"] == [16, 4, 21]
    assert report["fallback_rate"] == 1.0


def test_audit_rejects_prediction_contract_mismatch(tmp_path):
    from scripts.audit_scheduling_proxy import audit_proxy_artifact

    output = _make_smoke_artifact(tmp_path)
    prediction_path = output / "predictions_test.npz"
    import numpy as np

    with np.load(prediction_path, allow_pickle=False) as payload:
        values = {name: payload[name] for name in payload.files}
    values["contract_sha256"] = np.asarray("0" * 64)
    np.savez_compressed(prediction_path, **values)
    with pytest.raises(ValueError, match="contract SHA-256"):
        audit_proxy_artifact(output, BENCHMARK, CONTRACT)


def test_audit_rejects_dataset_contract_mismatch(tmp_path):
    from scripts.audit_scheduling_proxy import audit_proxy_artifact

    output = _make_smoke_artifact(tmp_path)
    manifest_path = output / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contract_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="contract/benchmark hash"):
        audit_proxy_artifact(output, BENCHMARK, CONTRACT)


def test_audit_does_not_modify_input_artifacts(tmp_path):
    from scripts.audit_scheduling_proxy import audit_proxy_artifact
    from src.scheduling.proxy_contract import file_sha256

    output = _make_smoke_artifact(tmp_path)
    before = file_sha256(output / "predictions_test.npz")
    audit_proxy_artifact(output, BENCHMARK, CONTRACT)
    assert file_sha256(output / "predictions_test.npz") == before
