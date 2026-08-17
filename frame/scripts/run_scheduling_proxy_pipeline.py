"""Generate, train, evaluate and smoke-test the pure-simulation proxy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FRAME_ROOT = SCRIPT_DIR.parent
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.scheduling.proxy_contract import (  # noqa: E402
    DEFAULT_CONTRACT_PATH,
    ProxyContract,
    contract_sha256,
    file_sha256,
    load_contract,
)
from src.scheduling.proxy_dataset import (  # noqa: E402
    LabeledProxySplit,
    ProxyNormalizationStats,
    build_labeled_proxy_split,
    load_proxy_split,
    save_proxy_split,
    scenario_id_digest,
    validate_dataset_manifest,
    write_dataset_manifest,
)
from src.scheduling.proxy_evaluation import evaluate_model_on_split, load_prediction_artifact, save_metrics  # noqa: E402
from src.scheduling.proxy_training import load_trained_proxy, train_proxy  # noqa: E402
from src.scheduling.synthetic_scenarios import generate_synthetic_scenarios  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "dataset": output_dir / "dataset",
        "manifest": output_dir / "dataset" / "manifest.json",
        "stats": output_dir / "normalization_stats.npz",
        "checkpoint": output_dir / "best_model.pt",
        "history": output_dir / "history.json",
        "metrics": output_dir / "metrics_test.json",
        "predictions": output_dir / "predictions_test.npz",
        "smoke_manifest": output_dir / "smoke_manifest.json",
    }


def _validate_model_bundle_against_manifest(manifest: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    """Bind a trained bundle to the exact current train/validation splits."""

    metadata = checkpoint.get("metadata") if isinstance(checkpoint, Mapping) else None
    splits = manifest.get("splits") if isinstance(manifest, Mapping) else None
    if not isinstance(metadata, Mapping) or not isinstance(splits, Mapping):
        raise ValueError("model bundle or dataset manifest provenance is missing")
    for split_name, digest_field, seed_field in (
        ("train", "train_scenario_id_digest", "train_seed"),
        ("validation", "validation_scenario_id_digest", "validation_seed"),
    ):
        entry = splits.get(split_name)
        if not isinstance(entry, Mapping) or "scenario_id_digest" not in entry:
            raise ValueError(f"dataset manifest {split_name} scenario-ID provenance is missing")
        if str(metadata.get(digest_field, "")).lower() != str(entry["scenario_id_digest"]).lower():
            raise ValueError(f"model bundle {split_name} scenario-ID digest does not match current dataset manifest")
        try:
            stored_seed = int(metadata.get(seed_field))
            manifest_seed = int(entry.get("seed"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"model bundle {split_name} seed provenance is invalid") from exc
        if stored_seed != manifest_seed:
            raise ValueError(f"model bundle {split_name} seed does not match current dataset manifest")


def generate_dataset(
    contract: ProxyContract,
    benchmark_path: Path,
    output_dir: Path,
    smoke: bool,
) -> tuple[dict[str, LabeledProxySplit], dict[str, Any]]:
    paths = _paths(output_dir)
    paths["dataset"].mkdir(parents=True, exist_ok=True)
    splits: dict[str, LabeledProxySplit] = {}
    entries: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        spec = contract.split(name, smoke=smoke)
        scenarios = generate_synthetic_scenarios(
            benchmark_path,
            name,
            seed=spec.seed,
            n_samples=spec.size,
            generator_version=contract.generator_version,
        )
        labelled = build_labeled_proxy_split(scenarios, benchmark_path, contract=contract)
        save_proxy_split(labelled, paths["dataset"] / f"{name}.npz")
        splits[name] = labelled
        entries[name] = labelled.manifest_entry()
    write_dataset_manifest(splits, paths["manifest"], contract=contract, benchmark_path=benchmark_path, smoke=smoke)
    validate_dataset_manifest(paths["manifest"], contract, benchmark_path, smoke=smoke)
    stats = ProxyNormalizationStats.fit(
        splits["train"],
        benchmark=benchmark_path,
        epsilon=float(contract.normalization["epsilon"]),
        scale_floor=float(contract.normalization["scale_floor"]),
    )
    stats.save(paths["stats"])
    return splits, entries


def train_from_dataset(
    contract: ProxyContract,
    benchmark_path: Path,
    output_dir: Path,
    smoke: bool,
) -> Any:
    paths = _paths(output_dir)
    manifest = validate_dataset_manifest(paths["manifest"], contract, benchmark_path, smoke=smoke)
    train_spec = contract.split("train", smoke=smoke)
    validation_spec = contract.split("validation", smoke=smoke)
    expected_hash = file_sha256(benchmark_path)
    train_split = load_proxy_split(
        paths["dataset"] / "train.npz", expected_split="train", expected_seed=train_spec.seed,
        expected_count=train_spec.size, expected_contract=contract,
        expected_benchmark_sha256=expected_hash, expected_smoke=smoke,
    )
    validation_split = load_proxy_split(
        paths["dataset"] / "validation.npz", expected_split="validation", expected_seed=validation_spec.seed,
        expected_count=validation_spec.size, expected_contract=contract,
        expected_benchmark_sha256=expected_hash, expected_smoke=smoke,
    )
    stats = ProxyNormalizationStats.load(
        paths["stats"], expected_benchmark_sha256=expected_hash,
        expected_contract_sha256=contract_sha256(contract), expected_fit_split="train",
        expected_train_seed=train_spec.seed, expected_source_type=contract.source_type,
        expected_generator_version=contract.generator_version,
    ) if paths["stats"].exists() else ProxyNormalizationStats.fit(train_split, benchmark=benchmark_path)
    # Smoke runs intentionally use a bounded CPU epoch budget.  Formal mode
    # follows the frozen contract defaults.
    overrides = {"max_epochs": 3, "patience": 2} if smoke else {}
    result = train_proxy(
        train_split,
        validation_split,
        output_dir,
        contract=contract,
        benchmark=benchmark_path,
        stats=stats,
        **overrides,
    )
    _validate_model_bundle_against_manifest(manifest, {"metadata": result.metadata})
    return result


def evaluate_from_dataset(contract: ProxyContract, benchmark_path: Path, output_dir: Path, smoke: bool = False) -> dict[str, Any]:
    paths = _paths(output_dir)
    manifest = validate_dataset_manifest(paths["manifest"], contract, benchmark_path, smoke=smoke)
    test_spec = contract.split("test", smoke=smoke)
    expected_hash = file_sha256(benchmark_path)
    test_split = load_proxy_split(
        paths["dataset"] / "test.npz", expected_split="test", expected_seed=test_spec.seed,
        expected_count=test_spec.size, expected_contract=contract,
        expected_benchmark_sha256=expected_hash, expected_smoke=smoke,
    )
    model, stats, checkpoint = load_trained_proxy(
        output_dir, contract=contract, benchmark_path=benchmark_path,
        expected_train_seed=contract.split("train", smoke=smoke).seed,
    )
    _validate_model_bundle_against_manifest(manifest, checkpoint)
    from src.scheduling.synthetic_scenarios import load_benchmark
    parameters = load_benchmark(benchmark_path)["values"]
    metrics = evaluate_model_on_split(
        model,
        test_split,
        stats,
        parameters,
        allow_exact_fallback=bool(contract.safety["allow_exact_fallback"]),
        feasibility_tolerance=float(contract.safety["feasibility_tolerance"]),
    )
    save_metrics(metrics, paths["metrics"])
    # Store raw predictions and fallback-assisted predictions separately; the
    # raw array is never overwritten by safety fallback.
    from src.scheduling.proxy_adapter import Scheme2RProxyAdapter
    loads = np.concatenate([test_split.demand, test_split.inputs[:, :, 3:4]], axis=-1)
    renewable = np.stack([test_split.pv_available, test_split.wt_available], axis=-1)
    result = Scheme2RProxyAdapter(stats, benchmark_path, contract).infer(
        model, loads, renewable, prices=test_split.prices, initial_soc=test_split.initial_soc,
        use_gas_prior=True, allow_exact_fallback=bool(contract.safety["allow_exact_fallback"]),
        feasibility_tolerance=float(contract.safety["feasibility_tolerance"]),
    )
    prediction_payload = {
        "prediction": result.raw_dispatch,
        "raw_prediction": result.raw_dispatch,
        "safe_prediction": result.safe_dispatch,
        "target": test_split.dispatch,
        "fallback_mask": result.fallback_mask,
        "raw_feasible_mask": result.raw_feasible_mask,
        "features": result.features,
        "scenario_ids": test_split.scenario_ids,
        "teacher_objective": test_split.teacher_objective,
        "benchmark_sha256": np.asarray(test_split.benchmark_sha256),
        "contract_sha256": np.asarray(test_split.contract_sha256),
        "source_type": np.asarray(test_split.source_type),
        "generator_version": np.asarray(test_split.generator_version),
        "split": np.asarray(test_split.split),
        "seed": np.asarray(test_split.seed, dtype=np.int64),
        "sample_count": np.asarray(test_split.n_samples, dtype=np.int64),
        "scenario_id_digest": np.asarray(scenario_id_digest(test_split.scenario_ids)),
        "feature_order": np.asarray(contract.feature_order),
        "label_order": np.asarray(contract.label_order),
    }
    temporary = paths["predictions"].with_suffix(paths["predictions"].suffix + ".tmp")
    np.savez_compressed(temporary, **prediction_payload)
    os.replace(str(temporary) if temporary.exists() else str(temporary) + ".npz", str(paths["predictions"]))
    load_prediction_artifact(
        paths["predictions"], expected_split=test_split.split, expected_count=test_split.n_samples,
        expected_seed=test_split.seed, expected_scenario_ids=test_split.scenario_ids,
        expected_scenario_id_digest=scenario_id_digest(test_split.scenario_ids),
        expected_benchmark_sha256=test_split.benchmark_sha256,
        expected_contract_sha256=test_split.contract_sha256,
        expected_source_type=test_split.source_type,
        expected_generator_version=test_split.generator_version,
    )
    return metrics


def run_pipeline(mode: str, benchmark: str | Path, contract_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    benchmark_path = Path(benchmark)
    output_path = Path(output_dir)
    contract = load_contract(contract_path, benchmark_path=benchmark_path)
    smoke = mode == "smoke"
    started = time.perf_counter()
    dataset_entries: dict[str, Any] = {}
    if mode in {"generate", "smoke"}:
        _, dataset_entries = generate_dataset(contract, benchmark_path, output_path, smoke=smoke)
    if mode == "train":
        train_from_dataset(contract, benchmark_path, output_path, smoke=False)
    elif mode == "smoke":
        train_from_dataset(contract, benchmark_path, output_path, smoke=True)
    if mode in {"evaluate", "smoke"}:
        metrics = evaluate_from_dataset(contract, benchmark_path, output_path, smoke=smoke)
    else:
        metrics = {}
    elapsed = time.perf_counter() - started
    if mode == "smoke":
        prediction_path = _paths(output_path)["predictions"]
        with np.load(prediction_path, allow_pickle=False) as payload:
            shape = list(payload["raw_prediction"].shape)
        dataset_manifest = json.loads(_paths(output_path)["manifest"].read_text(encoding="utf-8"))
        artifact_names = [
            "dataset/train.npz", "dataset/validation.npz", "dataset/test.npz",
            "dataset/manifest.json", "normalization_stats.npz", "best_model.pt",
            "history.json", "metrics_test.json", "predictions_test.npz",
        ]
        artifact_hashes = {name: file_sha256(output_path / name) for name in artifact_names}
        smoke_manifest = {
            "status": "passed",
            "mode": "smoke",
            "benchmark": str(benchmark_path),
            "benchmark_sha256": file_sha256(benchmark_path),
            "contract_sha256": contract_sha256(contract),
            "generator_version": contract.generator_version,
            "source_type": contract.source_type,
            "selection_split": "validation",
            "test_split_used_for_selection": False,
            "split_sizes": {name: int(contract.split(name, smoke=True).size) for name in ("train", "validation", "test")},
            "split_seeds": {name: int(contract.split(name, smoke=True).seed) for name in ("train", "validation", "test")},
            "output_shape": shape,
            "feature_order": list(contract.feature_order),
            "label_order": list(contract.label_order),
            "dataset_splits": dataset_manifest["splits"],
            "artifact_sha256": artifact_hashes,
            "rejected_solve_counts": {name: int(entry["rejected_solve_count"]) for name, entry in dataset_entries.items()},
            "artifacts": [
                *artifact_names, "smoke_manifest.json",
            ],
            "elapsed_seconds": elapsed,
            "metrics": _jsonable(metrics),
            "formal_paper_result": False,
        }
        _paths(output_path)["smoke_manifest"].parent.mkdir(parents=True, exist_ok=True)
        _paths(output_path)["smoke_manifest"].write_text(json.dumps(smoke_manifest, indent=2, sort_keys=True), encoding="utf-8")
        return smoke_manifest
    return {"status": "passed", "mode": mode, "metrics": _jsonable(metrics), "elapsed_seconds": elapsed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "train", "evaluate", "smoke"), required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--output-dir", default=str(FRAME_ROOT / "reports" / "scheduling_proxy_v1" / "smoke"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_pipeline(args.mode, args.benchmark, args.contract, args.output_dir)
    print(json.dumps(_jsonable(manifest), indent=2, sort_keys=True))
    return 0 if manifest.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
