from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")
CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v1.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_smoke_pipeline_emits_required_artifacts(tmp_path):
    output = tmp_path / "scheduling_proxy_smoke"
    command = [
        sys.executable, str(ROOT / "scripts" / "run_scheduling_proxy_pipeline.py"),
        "--mode", "smoke", "--benchmark", str(BENCHMARK), "--contract", str(CONTRACT), "--output-dir", str(output),
    ]
    completed = subprocess.run(command, cwd=str(ROOT.parent), capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    required = [
        output / "dataset" / "train.npz", output / "dataset" / "validation.npz", output / "dataset" / "test.npz",
        output / "dataset" / "manifest.json", output / "normalization_stats.npz", output / "best_model.pt",
        output / "history.json", output / "metrics_test.json", output / "predictions_test.npz", output / "smoke_manifest.json",
    ]
    assert all(path.exists() for path in required)
    manifest = json.loads((output / "smoke_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["source_type"] == "pure_simulation"
    assert set(manifest["artifact_sha256"]) == {
        "dataset/train.npz", "dataset/validation.npz", "dataset/test.npz",
        "dataset/manifest.json", "normalization_stats.npz", "best_model.pt",
        "history.json", "metrics_test.json", "predictions_test.npz",
    }
    from src.scheduling.proxy_contract import file_sha256
    for relative, digest in manifest["artifact_sha256"].items():
        assert file_sha256(output / relative) == digest
    with np.load(output / "predictions_test.npz", allow_pickle=False) as payload:
        assert payload["raw_prediction"].shape == (16, 4, 21)
        assert payload["safe_prediction"].shape == (16, 4, 21)
        assert str(payload["source_type"].item()) == "pure_simulation"
        assert int(payload["sample_count"].item()) == 16
        assert payload["scenario_id_digest"].ndim == 0
        assert payload["feature_order"].shape == (10,)
        assert payload["label_order"].shape == (21,)
    metrics = json.loads((output / "metrics_test.json").read_text(encoding="utf-8"))
    assert "raw_proxy" in metrics and "safe_with_exact_fallback" in metrics
    assert "objective_gap_absolute_mean" in metrics["raw_proxy"]
    assert "objective_gap_absolute_mean" in metrics["safe_with_exact_fallback"]

    # Rebind validation IDs to train IDs in a copied dataset and refresh its
    # local digests; the manifest validator must still reject cross-split overlap.
    overlap_root = tmp_path / "overlap"
    shutil.copytree(output / "dataset", overlap_root / "dataset")
    train_path = overlap_root / "dataset" / "train.npz"
    validation_path = overlap_root / "dataset" / "validation.npz"
    with np.load(train_path, allow_pickle=False) as train_payload:
        train_ids = np.asarray(train_payload["scenario_ids"])[:16]
    with np.load(validation_path, allow_pickle=False) as validation_payload:
        validation_values = {name: validation_payload[name] for name in validation_payload.files}
    validation_values["scenario_ids"] = train_ids
    validation_values["scenario_id_digest"] = np.asarray(__import__("hashlib").sha256(train_ids.astype(np.int64).tobytes()).hexdigest())
    np.savez_compressed(validation_path, **validation_values)
    manifest_path = overlap_root / "dataset" / "manifest.json"
    overlap_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from src.scheduling.proxy_contract import file_sha256
    from src.scheduling.proxy_dataset import scenario_id_digest, validate_dataset_manifest
    overlap_manifest["splits"]["validation"]["artifact_sha256"] = file_sha256(validation_path)
    overlap_manifest["splits"]["validation"]["scenario_id_digest"] = scenario_id_digest(train_ids)
    overlap_manifest["splits"]["validation"]["scenario_id_min"] = int(train_ids.min())
    overlap_manifest["splits"]["validation"]["scenario_id_max"] = int(train_ids.max())
    manifest_path.write_text(json.dumps(overlap_manifest), encoding="utf-8")
    from src.scheduling.proxy_contract import load_contract
    with __import__("pytest").raises(ValueError, match="overlap"):
        validate_dataset_manifest(manifest_path, load_contract(CONTRACT, BENCHMARK), BENCHMARK, smoke=True)

    # A mutually consistent model bundle must still be tied to the current
    # train/validation IDs.  Substitute a different synthetic train-ID set,
    # refresh its local artifact metadata, and ensure official evaluation
    # rejects the old bundle despite matching contract/benchmark/seeds.
    substitution_root = tmp_path / "substitution"
    shutil.copytree(output, substitution_root)
    substituted_train_path = substitution_root / "dataset" / "train.npz"
    with np.load(substituted_train_path, allow_pickle=False) as train_payload:
        substituted_values = {name: train_payload[name] for name in train_payload.files}
    substituted_ids = np.asarray(substituted_values["scenario_ids"], dtype=np.int64) + 10_000_000
    substituted_values["scenario_ids"] = substituted_ids
    substituted_values["scenario_id_digest"] = np.asarray(scenario_id_digest(substituted_ids))
    np.savez_compressed(substituted_train_path, **substituted_values)
    substitution_manifest_path = substitution_root / "dataset" / "manifest.json"
    substitution_manifest = json.loads(substitution_manifest_path.read_text(encoding="utf-8"))
    substitution_manifest["splits"]["train"].update({
        "artifact_sha256": file_sha256(substituted_train_path),
        "scenario_id_digest": scenario_id_digest(substituted_ids),
        "scenario_id_min": int(substituted_ids.min()),
        "scenario_id_max": int(substituted_ids.max()),
    })
    substitution_manifest_path.write_text(json.dumps(substitution_manifest), encoding="utf-8")
    from scripts.run_scheduling_proxy_pipeline import evaluate_from_dataset
    with __import__("pytest").raises(ValueError, match="scenario-ID"):
        evaluate_from_dataset(load_contract(CONTRACT, BENCHMARK), BENCHMARK, substitution_root, smoke=True)


def test_default_output_path_is_ignored_smoke_location():
    from scripts.run_scheduling_proxy_pipeline import build_parser

    args = build_parser().parse_args(["--mode", "smoke", "--benchmark", str(BENCHMARK)])
    assert args.output_dir.endswith("frame\\reports\\scheduling_proxy_v1\\smoke") or args.output_dir.endswith("frame/reports/scheduling_proxy_v1/smoke")


def test_prediction_artifact_loader_rejects_missing_provenance(tmp_path):
    from src.scheduling.proxy_evaluation import load_prediction_artifact

    path = tmp_path / "prediction.npz"
    np.savez_compressed(
        path,
        raw_prediction=np.zeros((1, 4, 21)), safe_prediction=np.zeros((1, 4, 21)),
        target=np.zeros((1, 4, 21)), fallback_mask=np.zeros(1, dtype=bool),
        raw_feasible_mask=np.zeros(1, dtype=bool), features=np.zeros((1, 4, 10)),
        scenario_ids=np.asarray([1]), teacher_objective=np.zeros(1),
        source_type=np.asarray("pure_simulation"), generator_version=np.asarray("synthetic-scheduling-v1"),
        split=np.asarray("test"), seed=np.asarray(2028), feature_order=np.asarray([str(i) for i in range(10)]),
        label_order=np.asarray([str(i) for i in range(21)]), benchmark_sha256=np.asarray("0" * 64),
        contract_sha256=np.asarray("0" * 64),
    )
    with __import__("pytest").raises(ValueError):
        load_prediction_artifact(path, expected_split="test")


def test_prediction_artifact_identity_tampering_is_rejected(tmp_path):
    from src.scheduling.proxy_contract import FEATURE_ORDER, LABEL_ORDER
    from src.scheduling.proxy_dataset import scenario_id_digest
    from src.scheduling.proxy_evaluation import load_prediction_artifact

    ids = np.asarray([10, 11], dtype=np.int64)
    payload = {
        "raw_prediction": np.zeros((2, 4, 21)), "safe_prediction": np.zeros((2, 4, 21)),
        "target": np.zeros((2, 4, 21)), "fallback_mask": np.asarray([False, True]),
        "raw_feasible_mask": np.asarray([True, False]), "features": np.zeros((2, 4, 10)),
        "scenario_ids": ids, "teacher_objective": np.zeros(2), "benchmark_sha256": np.asarray("a" * 64),
        "contract_sha256": np.asarray("b" * 64), "source_type": np.asarray("pure_simulation"),
        "generator_version": np.asarray("synthetic-scheduling-v1"), "split": np.asarray("test"),
        "seed": np.asarray(2028, dtype=np.int64), "sample_count": np.asarray(2, dtype=np.int64),
        "scenario_id_digest": np.asarray(scenario_id_digest(ids)), "feature_order": np.asarray(FEATURE_ORDER),
        "label_order": np.asarray(LABEL_ORDER),
    }

    def write(name, **updates):
        values = dict(payload)
        values.update(updates)
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, **values)
        return path

    kwargs = dict(
        expected_split="test", expected_count=2, expected_seed=2028,
        expected_scenario_ids=ids, expected_scenario_id_digest=scenario_id_digest(ids),
    )
    load_prediction_artifact(write("valid"), **kwargs)
    with __import__("pytest").raises(ValueError, match="seed"):
        load_prediction_artifact(write("seed", seed=np.asarray(2027, dtype=np.int64)), **kwargs)
    changed_ids = np.asarray([10, 12], dtype=np.int64)
    with __import__("pytest").raises(ValueError, match="digest"):
        load_prediction_artifact(write("stale_digest", scenario_ids=changed_ids), **kwargs)
    with __import__("pytest").raises(ValueError, match="IDs"):
        load_prediction_artifact(write("wrong_ids", scenario_ids=changed_ids, scenario_id_digest=np.asarray(scenario_id_digest(changed_ids))), **kwargs)
    with __import__("pytest").raises(ValueError, match="integer"):
        load_prediction_artifact(write("float_ids", scenario_ids=ids.astype(float), scenario_id_digest=np.asarray(scenario_id_digest(ids))), **kwargs)
    with __import__("pytest").raises(ValueError, match="sample_count"):
        load_prediction_artifact(write("wrong_count", sample_count=np.asarray(3, dtype=np.int64)), **kwargs)
    with __import__("pytest").raises(ValueError, match="binary"):
        load_prediction_artifact(write("bad_mask", fallback_mask=np.asarray([0, 2], dtype=np.int64)), **kwargs)
