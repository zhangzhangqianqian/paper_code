from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import runpy
import pytest

from src.joint_dispatch.formal_v4_2_artifacts import canonical_sha256, sha256_file
from src.joint_dispatch.formal_v4_2_gate0 import (
    CAPACITY_SCHEMA,
    DIFFOPT_SCHEMA,
    ITRANSFORMER_SCHEMA,
    SOURCE_SCHEMA,
    Gate0PrerequisitePaths,
    Gate0ReceiptError,
    produce_diffopt_receipt,
    produce_itransformer_receipt,
    produce_source_manifest,
    validate_prerequisites,
)


PROBE = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "run_rsc_pf_formal_v4_2_gate0.py"))


def _receipts(tmp_path: Path) -> SimpleNamespace:
    paths = {}
    for name in ("itransformer_receipt", "diffopt_receipt", "capacity_receipt", "source_manifest"):
        path = tmp_path / f"{name}.json"; path.write_text("{}", encoding="utf-8"); paths[name] = path
    return SimpleNamespace(output_root=tmp_path, run_id="formal_v4_2_gate0_test", **paths)


def test_resource_probe_calls_real_components(monkeypatch, tmp_path):
    calls = Counter()
    def counted(name):
        def operation():
            calls[name] += 1
            return {"identity": name}
        return operation
    monkeypatch.setitem(PROBE["run_gate0_probe"].__globals__, "run_rsc_forward", counted("rsc_forward"))
    monkeypatch.setitem(PROBE["run_gate0_probe"].__globals__, "run_rsc_backward", counted("rsc_backward"))
    monkeypatch.setitem(PROBE["run_gate0_probe"].__globals__, "run_highs_lp", counted("highs_lp"))
    monkeypatch.setitem(PROBE["run_gate0_probe"].__globals__, "run_diff_lp", counted("diff_lp"))
    receipt = PROBE["run_gate0_probe"](_receipts(tmp_path))
    assert calls == Counter(rsc_forward=1, rsc_backward=1, highs_lp=1, diff_lp=1)
    assert receipt["synthetic_probe"] is False
    assert receipt["authorized_pilot"] is True


def test_gate0_fails_when_source_or_dependency_receipt_is_missing(tmp_path):
    fixture = _receipts(tmp_path)
    fixture.itransformer_receipt.unlink()
    receipt = PROBE["run_gate0_probe"](fixture)
    assert receipt["authorized_pilot"] is False


def test_gate0_uses_train_years_for_fit_and_never_evaluation(tmp_path):
    receipt = PROBE["run_gate0_probe"](_receipts(tmp_path))
    assert receipt["capacity_fit_years"] == [2015, 2016, 2017, 2018]
    assert receipt["normalization_fit_years"] == [2015, 2016, 2017, 2018]
    assert receipt["materialized_years"] == [2015, 2016, 2017, 2018, 2019]
    assert receipt["evaluation_year_accessed"] is False


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _valid_prerequisites(tmp_path: Path, *, run_id: str, contract_hash: str) -> Gate0PrerequisitePaths:
    source_unsigned = {
        "schema": SOURCE_SCHEMA,
        "run_id": run_id,
        "contract_sha256": contract_hash,
        "git_commit": "deadbeef",
        "entry_count": 1,
        "entries": [{"path": "frame/a.py", "sha256": "c" * 64, "size_bytes": 1}],
        "evaluation_year_accessed": False,
    }
    source = _write_json(
        tmp_path / "SOURCE_MANIFEST.json",
        {**source_unsigned, "identity_sha256": canonical_sha256(source_unsigned)},
    )
    source_hash = sha256_file(source)
    itransformer = _write_json(tmp_path / "ITRANSFORMER_SOURCE_RECEIPT.json", {
        "schema": ITRANSFORMER_SCHEMA,
        "run_id": run_id,
        "contract_sha256": contract_hash,
        "source_manifest_sha256": source_hash,
        "repository": "https://github.com/thuml/iTransformer",
        "commit": "c2426e68ca13f74aaec08045c5c724d8ad328124",
        "backbone_class": "model.iTransformer.Model",
        "reproduction_level": "official_backbone_adaptation",
        "verified": True,
        "imported_file_hashes": {"model/iTransformer.py": "d" * 64},
        "license_sha256": "e" * 64,
        "evaluation_year_accessed": False,
    })
    diffopt = _write_json(tmp_path / "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json", {
        "schema": DIFFOPT_SCHEMA,
        "run_id": run_id,
        "contract_sha256": contract_hash,
        "source_manifest_sha256": source_hash,
        "eligible": True,
        "finite": True,
        "dpp_passed": True,
        "gradient_norm": 1.0,
        "packages": {"cvxpylayers": "0.1.6"},
        "lock_sha256": "f" * 64,
        "evaluation_year_accessed": False,
    })
    capacity = _write_json(tmp_path / "CAPACITY_FREEZE.json", {
        "schema": CAPACITY_SCHEMA,
        "run_id": run_id,
        "contract_sha256": contract_hash,
        "source_manifest_sha256": source_hash,
        "status": "pass",
        "fit_years": [2015, 2016, 2017, 2018],
        "selection_influenced_capacity": False,
        "selected": {"multiplier": 1.0},
        "candidate_multipliers": [1.0],
        "thresholds": {"cooling_shortage_energy_ratio_max": 0.005},
        "capacity_scenario_hash": "1" * 64,
        "evaluation_year_accessed": False,
    })
    return Gate0PrerequisitePaths(source, itransformer, diffopt, capacity)


def test_gate0_receipts_reject_cross_run_and_cross_contract(tmp_path):
    paths = _valid_prerequisites(tmp_path, run_id="formal_v4_2_test", contract_hash="a" * 64)
    with pytest.raises(Gate0ReceiptError, match="run_id"):
        validate_prerequisites(paths, run_id="formal_v4_2_other", contract_sha256="a" * 64)
    with pytest.raises(Gate0ReceiptError, match="contract_sha256"):
        validate_prerequisites(paths, run_id="formal_v4_2_test", contract_sha256="b" * 64)


def test_gate0_receipts_reject_file_existence_without_valid_content(tmp_path):
    paths = Gate0PrerequisitePaths(*(
        _write_json(tmp_path / name, {})
        for name in (
            "SOURCE_MANIFEST.json",
            "ITRANSFORMER_SOURCE_RECEIPT.json",
            "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json",
            "CAPACITY_FREEZE.json",
        )
    ))
    with pytest.raises(Gate0ReceiptError, match="schema"):
        validate_prerequisites(paths, run_id="formal_v4_2_test", contract_sha256="a" * 64)


def test_source_manifest_producer_requires_clean_tracked_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "gate0@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Gate Zero"], check=True)
    (repo / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    closure = repo / "closure.txt"
    closure.write_text("a.py\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "a.py", "closure.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "fixture"], check=True, capture_output=True)
    receipt_path = tmp_path / "source.json"
    receipt = produce_source_manifest(repo, closure, receipt_path, run_id="formal_v4_2_test", contract_sha256="a" * 64)
    assert receipt["entry_count"] == 1
    assert receipt["evaluation_year_accessed"] is False
    (repo / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        produce_source_manifest(repo, closure, tmp_path / "dirty.json", run_id="formal_v4_2_test", contract_sha256="a" * 64)


def test_real_itransformer_and_diffopt_receipts_are_produced(tmp_path):
    frame = Path(__file__).parents[1]
    repo = frame.parent
    source_hash = "b" * 64
    itransformer = produce_itransformer_receipt(
        frame / "third_party" / "iTransformer_source",
        tmp_path / "itransformer.json",
        run_id="formal_v4_2_test",
        contract_sha256="a" * 64,
        source_manifest_sha256=source_hash,
        repo_root=repo,
    )
    diffopt = produce_diffopt_receipt(
        frame / "requirements" / "formal_v4_diffopt.lock",
        tmp_path / "diffopt.json",
        run_id="formal_v4_2_test",
        contract_sha256="a" * 64,
        source_manifest_sha256=source_hash,
    )
    assert itransformer["verified"] is True
    assert diffopt["eligible"] is True
    assert diffopt["gradient_norm"] > 0.0
