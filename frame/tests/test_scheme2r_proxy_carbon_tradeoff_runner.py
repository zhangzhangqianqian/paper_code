from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_scheme2r_proxy_carbon_tradeoff as runner  # noqa: E402
from scripts.run_scheme2r_proxy_carbon_tradeoff import build_parser, run  # noqa: E402
from src.scheduling.proxy_integration import load_integration_contract  # noqa: E402
from src.scheduling.proxy_integration_runtime import execution_fingerprint, load_execution_contract, sha256_file  # noqa: E402


EXECUTION = ROOT / "configs" / "scheme2r_proxy_carbon_tradeoff_execution_v1.json"
SCIENTIFIC = ROOT / "configs" / "scheme2r_proxy_carbon_tradeoff_contract_v1.json"


def _install_seal_fixture(tmp_path, monkeypatch):
    formal_root = tmp_path / "formal"
    dataset = formal_root / "dataset"
    dataset.mkdir(parents=True)
    for relative, content in {
        "normalization_stats.npz": b"normalization",
        "best_model.pt": b"checkpoint",
        "history.json": b"{}\n",
        "formal_validation_acceptance.json": b"{}\n",
        "dataset/train.npz": b"train",
        "dataset/manifest.json": b"{}\n",
        "predictions_test.npz": b"predictions",
        "test_constraint_diagnostics.json": b"{}\n",
        "metrics_test.json": json.dumps({
            "inference_exact_lp_calls": 0,
            "fallback_rate": 0.0,
            "provenance": {
                "model_family": "feasible_scheduling_proxy_v2",
                "decoder_schema_version": "horizon-reachable-feasible-v2",
                "inference_exact_lp_calls": 0,
                "fallback_rate": 0.0,
            },
        }).encode("utf-8"),
    }.items():
        path = formal_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    scheduler = (ROOT / "configs" / "scheduling_proxy_contract_v4.json").resolve()
    freeze = {
        "status": "frozen",
        "test_evaluation_started": False,
        "metadata": {
            "runtime": {
                "model_family": "feasible_scheduling_proxy_v2",
                "decoder_schema_version": "horizon-reachable-feasible-v2",
                "inference_exact_lp_calls": 0,
            },
            "checkpoint": {
                "model_family": "feasible_scheduling_proxy_v2",
                "decoder_schema_version": "horizon-reachable-feasible-v2",
                "inference_exact_lp_calls": 0,
            },
        },
        "gate_decision": {
            "conditions": [
                {"name": "fallback_rate", "passed": True, "measured": 0.0},
                {"name": "inference_exact_lp_calls", "passed": True, "measured": 0.0},
            ]
        },
        "artifacts": {"scheduler_contract": {"path": str(scheduler)}},
    }
    freeze_path = formal_root / "formal_freeze.json"
    freeze_path.write_text(json.dumps(freeze) + "\n", encoding="utf-8")
    freeze_hash = sha256_file(freeze_path)
    test_artifacts = {
        name: sha256_file(formal_root / name)
        for name in ("metrics_test.json", "predictions_test.npz", "test_constraint_diagnostics.json")
    }
    manifest = {
        "status": "complete",
        "split": "test",
        "freeze_sha256": freeze_hash,
        "artifacts": {
            name: {"path": name, "sha256": digest}
            for name, digest in test_artifacts.items()
        },
        "artifact_sha256": test_artifacts,
    }
    test_manifest_path = formal_root / "formal_test_manifest.json"
    test_manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (formal_root / "formal_test_receipt.json").write_text(json.dumps({
        "status": "complete", "split": "test", "test_evaluation_count": 1,
        "smoke_only": False, "freeze_sha256": freeze_hash,
        "manifest_sha256": sha256_file(test_manifest_path),
        "artifacts": test_artifacts,
    }) + "\n", encoding="utf-8")

    paired_rows = [
        {
            "weight": weight,
            "scheduler": scheduler,
            "operating_cost": 1.0,
            "slack_penalty": 0.0,
            "physical_carbon_emissions": 2.0,
            "objective": 1.0 + weight * 2.0,
            "slack": 0.0,
            "cost_including_slack": 1.0,
            "objective_gap_to_lp": 0.0,
            "carbon_gap_to_lp": 0.0,
            "distinct_dispatch_rate": 0.0,
            "carbon_nonincreasing_rate": 1.0,
            "feasible_rate": 1.0,
            "inference_exact_lp_calls": 0,
            "offline_lp_calls": 6,
        }
        for scheduler in ("lp", "proxy")
        for weight in (0.0, 0.2, 1.0)
    ]
    for split in ("validation", "test"):
        prefix = f"carbon_tradeoff_{split}"
        report = {
            "status": "complete",
            "rows": paired_rows,
            "metadata": {
                "status": "complete",
                "mode": "formal",
                "split": split,
                "evaluation_levels": [0.0, 0.2, 1.0],
                "semantic_name": "normalized_carbon_penalty_weight",
                "is_market_price": False,
                "proxy_inference_exact_lp_calls": 0,
                "offline_lp_calls": 6,
                "physical_feasible_rate": {"lp": 1.0, "proxy": 1.0},
            },
        }
        report_path = formal_root / f"{prefix}.json"
        csv_path = formal_root / f"{prefix}.csv"
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        csv_fields = tuple(paired_rows[0])
        csv_lines = [",".join(csv_fields)]
        csv_lines.extend(",".join(str(row[field]) for field in csv_fields) for row in paired_rows)
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        report_manifest = {
            "status": "complete",
            "mode": "formal",
            "split": split,
            "evaluation_levels": [0.0, 0.2, 1.0],
            "semantic_name": "normalized_carbon_penalty_weight",
            "is_market_price": False,
            "proxy_inference_exact_lp_calls": 0,
            "offline_lp_calls": 6,
            "physical_feasible_rate": {"lp": 1.0, "proxy": 1.0},
            "objective_identity": "objective = operating_cost + slack_penalty + lambda_c * physical_carbon_emissions",
            "artifacts": {
                report_path.name: {"path": report_path.name, "size": report_path.stat().st_size, "sha256": sha256_file(report_path)},
                csv_path.name: {"path": csv_path.name, "size": csv_path.stat().st_size, "sha256": sha256_file(csv_path)},
            },
        }
        (formal_root / f"{prefix}_manifest.json").write_text(json.dumps(report_manifest) + "\n", encoding="utf-8")

    raw = json.loads(SCIENTIFIC.read_text(encoding="utf-8"))
    raw["proxy"] = copy.deepcopy(raw["proxy"])
    raw["proxy"]["root"] = str(formal_root)
    # The repository contract is already sealed.  A temporary fixture must
    # begin unsealed so this test exercises the first-seal path rather than
    # treating copied hashes as evidence for a different root.
    for field in runner.SEAL_HASH_FIELDS:
        raw["proxy"][field] = ""
    scientific_path = tmp_path / "scheme2r_contract.json"
    scientific_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    scientific = load_integration_contract(scientific_path)
    monkeypatch.setattr(runner, "validate_formal_freeze", lambda *_args, **_kwargs: None)
    return scientific_path, scientific, formal_root


def test_scheme2r_full_run_requires_explicit_formal(tmp_path):
    args = build_parser().parse_args([
        "--execution-contract", str(EXECUTION), "--output-dir", str(tmp_path / "formal"),
    ])
    try:
        run(args)
    except ValueError as exc:
        assert "explicit --formal" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("runner unexpectedly launched formal integration")


def test_scheme2r_carbon_tradeoff_dry_run_plans_three_weight_copies(tmp_path):
    output = tmp_path / "dry"
    args = build_parser().parse_args([
        "--execution-contract", str(EXECUTION), "--output-dir", str(output), "--dry-run",
    ])
    result = run(args)
    assert result["status"] == "dry_run"
    assert result["evaluation_levels"] == [0.0, 0.2, 1.0]
    assert result["base_unit_count"] == 210
    assert result["unit_count"] == 630
    assert all("weight_" in value for value in result["unit_id_examples"])
    assert not output.exists()


def test_execution_fingerprint_binds_all_numeric_source_modules():
    execution = load_execution_contract(EXECUTION)
    hashes = runner._execution_source_hashes(execution)
    required_fragments = (
        "proxy_adapter.py", "proxy_model.py", "proxy_decoder.py", "proxy_physics.py",
        "proxy_dataset.py", "dispatch_schema.py", "dispatch_lp.py", "recourse.py",
        "proxy_training.py", "carbon_tradeoff.py",
    )
    assert all(any(fragment in key for key in hashes) for fragment in required_fragments)
    first = execution_fingerprint(execution, EXECUTION, resolved_sources=hashes)
    mutated = dict(hashes)
    first_key = next(key for key in mutated if "proxy_decoder.py" in key)
    mutated[first_key] = "0" * 64
    second = execution_fingerprint(execution, EXECUTION, resolved_sources=mutated)
    assert first != second


def test_execution_source_closure_rejects_missing_numeric_dependency():
    execution = load_execution_contract(EXECUTION)
    trimmed = replace(
        execution,
        source_files=tuple(
            source for source in execution.source_files
            if Path(str(source)).name != "proxy_dataset.py"
        ),
    )
    with pytest.raises(ValueError, match="proxy_dataset.py"):
        runner._execution_source_hashes(trimmed)


def test_scheme2r_smoke_reports_absent_v4_checkpoint_without_v3_substitution(tmp_path, monkeypatch):
    output = tmp_path / "smoke"
    monkeypatch.setattr(runner, "_checkpoint_path", lambda *args, **kwargs: tmp_path / "missing-v4" / "best_model.pt")
    args = build_parser().parse_args([
        "--execution-contract", str(EXECUTION), "--output-dir", str(output),
        "--work-dir", str(tmp_path / "work"), "--smoke-limit", "8",
    ])
    result = run(args)
    assert result["status"] in {"skipped_missing_v4_checkpoint", "complete"}
    if result["status"] == "skipped_missing_v4_checkpoint":
        assert "v3 checkpoint" in result["reason"]


def test_scheme2r_rejects_existing_empty_final_output_before_work(tmp_path):
    output = tmp_path / "existing-empty"
    output.mkdir()
    args = build_parser().parse_args([
        "--execution-contract", str(EXECUTION), "--output-dir", str(output), "--smoke-limit", "8",
    ])
    with pytest.raises(FileExistsError, match="final output path"):
        run(args)


def test_seal_contract_preview_write_and_idempotency(tmp_path, monkeypatch):
    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path, monkeypatch)
    preview = runner._seal_v4_contract(scientific_path, scientific, dry_run=True)
    assert preview["status"] == "seal_preview"
    assert preview["changed_fields"]
    before = scientific_path.read_bytes()
    sealed = runner._seal_v4_contract(scientific_path, scientific)
    assert sealed["status"] == "sealed"
    evidence_path = formal_root / runner.PAIRED_EVIDENCE_RECEIPT_FILE
    assert evidence_path.exists()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert set(evidence["reports"]) == {"validation", "test"}
    assert all(len(evidence["reports"][split]["artifacts"]) == 3 for split in evidence["reports"])
    anchor_path = formal_root / runner.SEAL_RECEIPT_FILE
    assert anchor_path.exists()
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    assert anchor["schema_version"] == runner.SEAL_RECEIPT_SCHEMA
    assert anchor["schema"] == runner.SEAL_RECEIPT_SCHEMA
    assert anchor["base_freeze_sha256"] == sha256_file(formal_root / "formal_freeze.json")
    assert anchor["formal_test_receipt_sha256"] == sha256_file(formal_root / "formal_test_receipt.json")
    assert anchor["paired_evidence_receipt_sha256"] == sha256_file(evidence_path)
    assert anchor["scientific_contract_sha256"] == sha256_file(scientific_path)
    assert anchor["execution_contract_sha256"] == sha256_file(EXECUTION)
    assert scientific_path.read_bytes() != before
    sealed_bytes = scientific_path.read_bytes()
    anchor_bytes = anchor_path.read_bytes()
    again = runner._seal_v4_contract(scientific_path, scientific)
    assert again["status"] == "already_sealed"
    assert scientific_path.read_bytes() == sealed_bytes
    assert anchor_path.read_bytes() == anchor_bytes


def test_seal_contract_cli_preview_is_explicit_and_side_effect_free(tmp_path, monkeypatch):
    scientific_path, _, _ = _install_seal_fixture(tmp_path, monkeypatch)
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    execution["scientific_contract"] = str(scientific_path)
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(execution) + "\n", encoding="utf-8")
    before = scientific_path.read_bytes()
    args = build_parser().parse_args([
        "--execution-contract", str(execution_path), "--seal-contract", "--dry-run",
    ])
    result = run(args)
    assert result["status"] == "seal_preview"
    assert scientific_path.read_bytes() == before


def test_seal_contract_rejects_missing_receipt_and_tampered_artifact(tmp_path, monkeypatch):
    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path, monkeypatch)
    (formal_root / "formal_test_receipt.json").unlink()
    with pytest.raises(ValueError, match="test receipt"):
        runner._seal_v4_contract(scientific_path, scientific)

    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path / "tampered", monkeypatch)
    runner._seal_v4_contract(scientific_path, scientific)
    (formal_root / "best_model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="conflicting seal hashes"):
        runner._seal_v4_contract(scientific_path, scientific)


def test_seal_contract_rejects_missing_formal_paired_report(tmp_path, monkeypatch):
    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path, monkeypatch)
    (formal_root / "carbon_tradeoff_test.json").unlink()
    with pytest.raises(ValueError, match="test paired report"):
        runner._seal_v4_contract(scientific_path, scientific)


def test_seal_contract_rejects_nonfeasible_paired_report(tmp_path, monkeypatch):
    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path, monkeypatch)
    manifest_path = formal_root / "carbon_tradeoff_test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["physical_feasible_rate"]["proxy"] = 0.5
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="physically feasible"):
        runner._seal_v4_contract(scientific_path, scientific)


def test_sealed_paired_evidence_rejects_deleted_or_tampered_artifacts(tmp_path, monkeypatch):
    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path, monkeypatch)
    runner._seal_v4_contract(scientific_path, scientific)
    freeze_path = formal_root / "formal_freeze.json"
    receipt_path = formal_root / "formal_test_receipt.json"
    evidence_path = formal_root / runner.PAIRED_EVIDENCE_RECEIPT_FILE
    (formal_root / "carbon_tradeoff_validation.csv").unlink()
    with pytest.raises(ValueError, match="validation paired CSV"):
        runner._validate_sealed_paired_evidence(formal_root, freeze_path, receipt_path)

    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path / "tampered-evidence", monkeypatch)
    runner._seal_v4_contract(scientific_path, scientific)
    evidence_path = formal_root / runner.PAIRED_EVIDENCE_RECEIPT_FILE
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["reports"]["test"]["artifacts"]["carbon_tradeoff_test.csv"]["sha256"] = "0" * 64
    evidence_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash/size mismatch|does not match sealed reports"):
        runner._validate_sealed_paired_evidence(
            formal_root,
            formal_root / "formal_freeze.json",
            formal_root / "formal_test_receipt.json",
        )


def test_seal_anchor_rejects_missing_or_tampered_anchor(tmp_path, monkeypatch):
    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path, monkeypatch)
    runner._seal_v4_contract(scientific_path, scientific)
    anchor_path = formal_root / runner.SEAL_RECEIPT_FILE
    anchor_path.unlink()
    with pytest.raises(ValueError, match="seal trust receipt is missing"):
        runner._seal_v4_contract(scientific_path, scientific)
    with pytest.raises(ValueError, match="seal trust receipt"):
        runner._validate_seal_receipt(
            formal_root,
            freeze_path=formal_root / "formal_freeze.json",
            test_receipt_path=formal_root / "formal_test_receipt.json",
            paired_evidence_path=formal_root / runner.PAIRED_EVIDENCE_RECEIPT_FILE,
            scientific_path=scientific_path,
            execution_path=EXECUTION,
        )

    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path / "tampered-anchor", monkeypatch)
    runner._seal_v4_contract(scientific_path, scientific)
    anchor_path = formal_root / runner.SEAL_RECEIPT_FILE
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["paired_evidence_receipt_sha256"] = "0" * 64
    anchor_path.write_text(json.dumps(anchor) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match current sealed artifacts"):
        runner._validate_seal_receipt(
            formal_root,
            freeze_path=formal_root / "formal_freeze.json",
            test_receipt_path=formal_root / "formal_test_receipt.json",
            paired_evidence_path=formal_root / runner.PAIRED_EVIDENCE_RECEIPT_FILE,
            scientific_path=scientific_path,
            execution_path=EXECUTION,
        )


def test_seal_anchor_rejects_rewritten_reports_manifests_and_evidence(tmp_path, monkeypatch):
    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path, monkeypatch)
    runner._seal_v4_contract(scientific_path, scientific)

    # Rewrite the report bytes and a CSV byte while keeping their logical
    # rows valid; update both manifests and rebuild the paired receipt.  The
    # immutable anchor must reject the complete rewritten set.
    report_path = formal_root / "carbon_tradeoff_test.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.write_text(json.dumps(report, indent=4) + "\n", encoding="utf-8")
    csv_path = formal_root / "carbon_tradeoff_test.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    manifest_path = formal_root / "carbon_tradeoff_test_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][report_path.name].update({"size": report_path.stat().st_size, "sha256": sha256_file(report_path)})
    manifest["artifacts"][csv_path.name].update({"size": csv_path.stat().st_size, "sha256": sha256_file(csv_path)})
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    paired_reports = {
        split: runner._validate_formal_paired_report(formal_root, split)
        for split in ("validation", "test")
    }
    evidence = runner._paired_evidence_payload(
        formal_root,
        formal_root / "formal_freeze.json",
        formal_root / "formal_test_receipt.json",
        paired_reports,
    )
    (formal_root / runner.PAIRED_EVIDENCE_RECEIPT_FILE).write_bytes(runner._canonical_json_bytes(evidence))
    with pytest.raises(ValueError, match="does not match current sealed artifacts"):
        runner._validate_seal_receipt(
            formal_root,
            freeze_path=formal_root / "formal_freeze.json",
            test_receipt_path=formal_root / "formal_test_receipt.json",
            paired_evidence_path=formal_root / runner.PAIRED_EVIDENCE_RECEIPT_FILE,
            scientific_path=scientific_path,
            execution_path=EXECUTION,
        )


def test_missing_paired_evidence_recovers_only_from_sealed_anchor(tmp_path, monkeypatch):
    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path, monkeypatch)
    runner._seal_v4_contract(scientific_path, scientific)
    evidence_path = formal_root / runner.PAIRED_EVIDENCE_RECEIPT_FILE
    expected_evidence = evidence_path.read_bytes()
    anchor_path = formal_root / runner.SEAL_RECEIPT_FILE
    expected_anchor = anchor_path.read_bytes()
    evidence_path.unlink()

    result = runner._seal_v4_contract(scientific_path, scientific)
    assert result["status"] == "sealed_evidence_repaired"
    assert evidence_path.read_bytes() == expected_evidence
    assert anchor_path.read_bytes() == expected_anchor


def test_formal_integration_startup_requires_seal_anchor(tmp_path, monkeypatch):
    scientific_path, scientific, formal_root = _install_seal_fixture(tmp_path, monkeypatch)
    runner._seal_v4_contract(scientific_path, scientific)
    (formal_root / runner.SEAL_RECEIPT_FILE).unlink()
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    execution["scientific_contract"] = str(scientific_path)
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(json.dumps(execution) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "verify_contract_sources",
        lambda *args, **kwargs: {"proxy_root": str(formal_root), "source_hashes": {}},
    )
    args = build_parser().parse_args([
        "--execution-contract", str(execution_path), "--output-dir", str(tmp_path / "integration"), "--formal",
    ])
    with pytest.raises(ValueError, match="seal trust receipt"):
        run(args)


def test_published_manifest_requires_hashes_for_every_required_file(tmp_path):
    import pandas as pd

    root = tmp_path / "final"
    files = {"pathway_metrics.csv": pd.DataFrame({"x": [1]})}
    manifest = {"required_output_files": ["pathway_metrics.csv"]}
    runner._publish(root, files, manifest)
    payload = json.loads((root / "carbon_tradeoff_manifest.json").read_text(encoding="utf-8"))
    assert payload["artifacts"]["pathway_metrics.csv"]["size"] == (root / "pathway_metrics.csv").stat().st_size
    assert payload["artifacts"]["pathway_metrics.csv"]["sha256"] == sha256_file(root / "pathway_metrics.csv")


def test_published_manifest_rejects_missing_required_file(tmp_path):
    import pandas as pd

    with pytest.raises(ValueError, match="missing required output files"):
        runner._publish(
            tmp_path / "missing-required",
            {"other.csv": pd.DataFrame({"x": [1]})},
            {"required_output_files": ["pathway_metrics.csv"]},
        )


def test_published_manifest_rejects_hash_tampering(tmp_path):
    import pandas as pd

    root = tmp_path / "tampered-manifest"
    runner._publish(
        root,
        {"pathway_metrics.csv": pd.DataFrame({"x": [1]})},
        {"required_output_files": ["pathway_metrics.csv"]},
    )
    payload = json.loads((root / "carbon_tradeoff_manifest.json").read_text(encoding="utf-8"))
    payload["artifacts"]["pathway_metrics.csv"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash/size mismatch"):
        runner._validate_published_manifest(root, payload)


def test_published_manifest_rejects_existing_empty_final_directory(tmp_path):
    import pandas as pd

    root = tmp_path / "existing-empty"
    root.mkdir()
    with pytest.raises(FileExistsError, match="existing final output"):
        runner._publish(root, {"pathway_metrics.csv": pd.DataFrame({"x": [1]})}, {"required_output_files": ["pathway_metrics.csv"]})


def test_combine_bundles_reports_sample_level_distinct_rate():
    import numpy as np
    import pandas as pd

    dispatch = np.zeros((2, 4, 4, 4, 21), dtype=np.float64)
    changed = dispatch.copy()
    changed[:, 0, 2:, 0, 0] = 1.0
    metric_base = np.zeros((2, 4, 4, 6), dtype=np.float64)
    metric_base[..., 2] = 2.0
    metric_base[..., 4] = 1.0
    metric_changed = metric_base.copy()
    metric_changed[:, 0, 2:, 2] = 3.0
    rows = pd.DataFrame([
        {"seed": seed, "pathway": pathway, "cost_mean": 1.0, "carbon_mean": 2.0, "objective_mean": 1.0, "slack_total_mean": 0.0, "physical_feasible_rate": 1.0}
        for seed in (2026, 2027)
        for pathway in ("A", "B", "C", "D")
    ])
    def bundle(pathway_dispatch, metric_values):
        return {
            "pathway_metrics": rows,
            "effect_decomposition": pd.DataFrame(),
            "forecast_error_by_task_horizon": pd.DataFrame(),
            "dispatch_error_by_variable": pd.DataFrame(),
            "arrays": {"pathway_dispatch": pathway_dispatch, "pathway_metric_values": metric_values},
            "summary": {"seeds": [2026, 2027], "lp_calls": {}, "inference_exact_lp_calls": {}},
        }
    summary, *_ = runner._combine_bundles({0.0: bundle(dispatch, metric_base), 0.2: bundle(changed, metric_changed), 1.0: bundle(changed, metric_changed)}, (0.0, 0.2, 1.0))
    assert summary["distinct_dispatch_rate"]["0.2"]["A"] == pytest.approx(0.5)
    assert summary["carbon_nonincreasing_rate"]["A"] == pytest.approx(0.75)


def test_metric_arrays_flatten_seed_sample_before_combining():
    import numpy as np
    import pandas as pd

    seed_count, pathway_count, sample_count, horizon, label_count = 2, 4, 4, 4, 21
    feature_count = 10
    dispatch = np.zeros((seed_count, pathway_count, sample_count, horizon, label_count), dtype=np.float64)
    changed = dispatch.copy()
    changed[:, 0, 2:, 0, 0] = 1.0
    features = np.zeros((seed_count, pathway_count, sample_count, horizon, feature_count), dtype=np.float64)
    features[..., 6:9] = 1.0
    rows = pd.DataFrame([
        {
            "seed": seed,
            "pathway": pathway,
            "cost_mean": 0.0,
            "carbon_mean": 0.0,
            "objective_mean": 0.0,
            "slack_total_mean": 0.0,
            "physical_feasible_rate": 1.0,
        }
        for seed in (2026, 2027)
        for pathway in ("A", "B", "C", "D")
    ])

    def bundle(pathway_dispatch):
        return {
            "pathway_metrics": rows.copy(),
            "effect_decomposition": pd.DataFrame(),
            "forecast_error_by_task_horizon": pd.DataFrame(),
            "dispatch_error_by_variable": pd.DataFrame(),
            "arrays": {
                "pathway_dispatch": pathway_dispatch,
                "pathway_features": features.copy(),
            },
            "summary": {
                "seeds": [2026, 2027],
                "lp_calls": {},
                "inference_exact_lp_calls": {},
            },
        }

    baseline = bundle(dispatch)
    higher = bundle(changed)
    parameters = {
        "grid_emission_factor": 1.0,
        "gas_emission_factor": 0.0,
        "bess_throughput_cost": 0.0,
        "unserved_penalty": 0.0,
    }
    # No pathway_metric_values are injected: attach must exercise the
    # seed×sample flatten/restore path before combine consumes the arrays.
    runner._attach_v4_metric_arrays(baseline, parameters)
    runner._attach_v4_metric_arrays(higher, parameters)
    assert baseline["arrays"]["pathway_metric_values"].shape == (2, 4, 4, 6)
    summary, *_ = runner._combine_bundles(
        {0.0: baseline, 0.2: higher, 1.0: higher},
        (0.0, 0.2, 1.0),
        parameters,
    )
    assert summary["distinct_dispatch_rate"]["0.2"]["A"] == pytest.approx(0.5)
