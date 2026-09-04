from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import runpy
import pytest

from src.joint_dispatch.formal_v4_2_artifacts import canonical_sha256, sha256_file
from src.joint_dispatch.formal_v4_2_gate0 import (
    CAPACITY_SCHEMA,
    DIFFOPT_SCHEMA,
    ITRANSFORMER_SCHEMA,
    SOURCE_SCHEMA,
    Gate0InputsV42,
    Gate0PrerequisitePaths,
    Gate0ReceiptError,
    produce_diffopt_receipt,
    produce_itransformer_receipt,
    produce_source_manifest,
    validate_prerequisites,
)


PROBE = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "run_rsc_pf_formal_v4_2_gate0.py"))


def test_versioned_gate0_import_does_not_require_legacy_modules():
    frame_root = Path(__file__).parents[1]
    code = r"""
import importlib.abc
import sys

class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"src.joint_dispatch.formal_protocol", "src.joint_dispatch.pto"}:
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockLegacy())
from src.joint_dispatch.formal_v4_2_artifacts import canonical_sha256
assert callable(canonical_sha256)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=frame_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


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


def _orchestrator_inputs(tmp_path: Path) -> Gate0InputsV42:
    frame = Path(__file__).parents[1]
    return Gate0InputsV42(
        contract_path=frame / "configs" / "joint_forecast_dispatch_formal_v4_2.json",
        output_root=tmp_path,
        run_id="formal_v4_2_gate0_orchestrator_test",
        data_dir=tmp_path / "data",
        diffopt_python=Path(__import__("sys").executable),
        itransformer_source=tmp_path / "iTransformer",
    )


def _install_orchestrator_fakes(monkeypatch, tmp_path: Path):
    globals_ = PROBE["run_gate0_orchestrator"].__globals__

    def source(repo_root, closure_path, destination, **kwargs):
        return _write_json(Path(destination), {"schema": SOURCE_SCHEMA, **kwargs, "evaluation_year_accessed": False})

    def itransformer(source_root, destination, **kwargs):
        return _write_json(Path(destination), {"schema": ITRANSFORMER_SCHEMA, **{k: v for k, v in kwargs.items() if k != "repo_root"}, "evaluation_year_accessed": False})

    def diffopt(lock_path, destination, **kwargs):
        return _write_json(Path(destination), {"schema": DIFFOPT_SCHEMA, **kwargs, "evaluation_year_accessed": False})

    def capacity(contract, root, data_dir, source_manifest_sha256):
        payload = {
            "schema": CAPACITY_SCHEMA,
            "run_id": Path(root).name,
            "contract_sha256": contract.contract_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "status": "pass",
            "fit_years": [2015, 2016, 2017, 2018],
            "selection_influenced_capacity": False,
            "evaluation_year_accessed": False,
            "selected": {"multiplier": 2.7},
            "candidate_multipliers": [2.7],
            "thresholds": {"cooling_shortage_energy_ratio_max": 0.005},
            "capacity_scenario_hash": "1" * 64,
        }
        _write_json(Path(root) / "gate0" / "CAPACITY_FREEZE.json", payload)
        return payload

    monkeypatch.setitem(globals_, "produce_source_manifest", source)
    monkeypatch.setitem(globals_, "produce_itransformer_receipt", itransformer)
    monkeypatch.setitem(globals_, "produce_diffopt_receipt", diffopt)
    monkeypatch.setitem(globals_, "build_capacity_evidence", capacity)
    monkeypatch.setitem(globals_, "validate_prerequisites", lambda *args, **kwargs: {
        "source_manifest": {"passed": True},
        "itransformer_receipt": {"passed": True},
        "diffopt_receipt": {"passed": True},
        "capacity_receipt": {"passed": True},
    })
    monkeypatch.setitem(globals_, "_run_real_operations", lambda *args, **kwargs: (
        {
            name: {"timing": {"p95_seconds": 0.01}, "details": {"identity": name}, "real_operation": True}
            for name in ("rsc_forward", "rsc_backward", "highs_lp", "diff_lp")
        },
        {name: {"passed": True, "identity": name} for name in ("rsc_forward", "rsc_backward", "highs_lp", "diff_lp")},
    ))
    monkeypatch.setattr(globals_["shutil"], "disk_usage", lambda path: SimpleNamespace(total=100, used=40, free=60))


def test_gate0_orchestrator_writes_authorization_only_after_checks(tmp_path, monkeypatch):
    _install_orchestrator_fakes(monkeypatch, tmp_path)
    inputs = _orchestrator_inputs(tmp_path)
    receipt = PROBE["run_gate0_orchestrator"](inputs)
    root = tmp_path / inputs.run_id
    assert receipt["authorized_pilot"] is True
    current = json.loads((root / "protocol" / "CURRENT_GATE.json").read_text(encoding="utf-8"))
    assert current["next_gate"] == "pilot"
    assert current["authorized_pilot"] is True
    assert not (root / "pilot").exists()


def test_gate0_orchestrator_writes_failure_and_returns_no_authorization(tmp_path, monkeypatch):
    _install_orchestrator_fakes(monkeypatch, tmp_path)
    inputs = _orchestrator_inputs(tmp_path)
    globals_ = PROBE["run_gate0_orchestrator"].__globals__
    monkeypatch.setitem(globals_, "build_capacity_evidence", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("capacity failed")))
    with pytest.raises(PROBE["Gate0ExecutionError"], match="capacity_receipt"):
        PROBE["run_gate0_orchestrator"](inputs)
    root = tmp_path / inputs.run_id
    failure = json.loads((root / "gate0" / "GATE0_FAILURE.json").read_text(encoding="utf-8"))
    assert failure["authorized_pilot"] is False
    assert not (root / "protocol" / "CURRENT_GATE.json").exists()
    assert not (root / "pilot").exists()
