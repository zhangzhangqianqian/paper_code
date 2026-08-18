"""Read-only identity audit for a completed scheduling-proxy artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.scheduling.proxy_contract import contract_sha256, file_sha256, load_contract  # noqa: E402
from src.scheduling.proxy_adapter import evaluate_raw_feasibility  # noqa: E402
from src.scheduling.proxy_dataset import load_proxy_split, validate_dataset_manifest  # noqa: E402
from src.scheduling.proxy_diagnostics import diagnose_dispatch, diagnostics_npz_payload  # noqa: E402
from src.scheduling.proxy_evaluation import dispatch_metrics, load_prediction_artifact  # noqa: E402
from src.scheduling.proxy_training import load_trained_proxy  # noqa: E402
from src.scheduling.synthetic_scenarios import load_benchmark  # noqa: E402


FORMAL_ARTIFACTS = (
    "dataset/train.npz",
    "dataset/validation.npz",
    "dataset/test.npz",
    "dataset/manifest.json",
    "normalization_stats.npz",
    "best_model.pt",
    "history.json",
    "metrics_test.json",
    "predictions_test.npz",
)


def _evaluation_split(contract: Any, smoke: bool) -> str:
    return "validation" if bool(smoke) and bool(contract.is_v2) else "test"


def _artifact_names(contract: Any, smoke: bool) -> tuple[str, ...]:
    evaluation_split = _evaluation_split(contract, smoke)
    return (
        "dataset/train.npz",
        "dataset/validation.npz",
        "dataset/test.npz",
        "dataset/manifest.json",
        "normalization_stats.npz",
        "best_model.pt",
        "history.json",
        f"metrics_{evaluation_split}.json",
        f"predictions_{evaluation_split}.npz",
    )


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _require_files(output_dir: Path, artifact_names: tuple[str, ...]) -> None:
    missing = [relative for relative in artifact_names if not (output_dir / relative).is_file()]
    if missing:
        raise ValueError(f"proxy artifact is missing required files: {missing}")


def _close(left: Any, right: Any, *, label: str) -> None:
    if not np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12):
        raise ValueError(f"{label} is inconsistent: {left!r} != {right!r}")


def audit_proxy_artifact(
    output_dir: str | Path,
    benchmark_path: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Audit a completed proxy artifact without training or changing files."""

    output = Path(output_dir)
    benchmark = Path(benchmark_path)
    contract_file = Path(contract_path)
    contract = load_contract(contract_file, benchmark)
    contract_digest = contract_sha256(contract)
    dataset_manifest_path = output / "dataset" / "manifest.json"
    dataset_manifest = validate_dataset_manifest(
        dataset_manifest_path,
        contract,
        benchmark,
        smoke=None,
    )
    smoke = bool(dataset_manifest["smoke"])
    smoke_manifest = None
    if contract.is_v2 and smoke:
        smoke_manifest = _read_json(output / "smoke_manifest.json", "smoke_manifest.json")
    evaluation_split = _evaluation_split(contract, smoke)
    artifact_names = _artifact_names(contract, smoke)
    _require_files(output, artifact_names)
    if contract.is_v2 and smoke:
        forbidden = [name for name in ("metrics_test.json", "predictions_test.npz") if (output / name).exists()]
        if forbidden:
            raise ValueError(f"v2 validation-only smoke artifact contains forbidden test outputs: {forbidden}")
    evaluation_spec = contract.split(evaluation_split, smoke=smoke)
    evaluation_data = load_proxy_split(
        output / "dataset" / f"{evaluation_split}.npz",
        expected_split=evaluation_split,
        expected_seed=evaluation_spec.seed,
        expected_count=evaluation_spec.size,
        expected_contract=contract,
        expected_benchmark_sha256=file_sha256(benchmark),
        expected_smoke=smoke,
    )
    _, stats, checkpoint = load_trained_proxy(
        output,
        contract=contract,
        benchmark_path=benchmark,
        expected_train_seed=contract.split("train", smoke=smoke).seed,
    )
    checkpoint_metadata = checkpoint.get("metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError("checkpoint metadata is missing")
    checkpoint_digest = file_sha256(output / "best_model.pt")
    if smoke_manifest is not None:
        if str(smoke_manifest.get("checkpoint_sha256", "")).lower() != checkpoint_digest.lower():
            raise ValueError("smoke manifest checkpoint SHA-256 does not match current checkpoint")
        if smoke_manifest.get("evaluation_split") != evaluation_split:
            raise ValueError("smoke manifest evaluation split mismatch")
    prediction_path = output / f"predictions_{evaluation_split}.npz"
    predictions = load_prediction_artifact(
        prediction_path,
        expected_split=evaluation_split,
        expected_count=evaluation_spec.size,
        expected_seed=evaluation_spec.seed,
        expected_scenario_ids=evaluation_data.scenario_ids,
        expected_scenario_id_digest=str(dataset_manifest["splits"][evaluation_split]["scenario_id_digest"]),
        expected_benchmark_sha256=file_sha256(benchmark),
        expected_contract_sha256=contract_digest,
        expected_source_type=contract.source_type,
        expected_generator_version=contract.generator_version,
        expected_model_family="feasible_scheduling_proxy_v2" if contract.is_v2 else None,
        expected_decoder_schema_version=contract.decoder_schema_version if contract.is_v2 else None,
        expected_inference_exact_lp_calls=0 if contract.is_v2 else None,
        expected_checkpoint_sha256=checkpoint_digest if contract.is_v2 else None,
        expected_train_seed=int(checkpoint_metadata["train_seed"]) if contract.is_v2 else None,
        expected_validation_seed=int(checkpoint_metadata["validation_seed"]) if contract.is_v2 else None,
        expected_train_scenario_id_digest=str(checkpoint_metadata["train_scenario_id_digest"]) if contract.is_v2 else None,
        expected_validation_scenario_id_digest=str(checkpoint_metadata["validation_scenario_id_digest"]) if contract.is_v2 else None,
    )
    metrics_path = output / f"metrics_{evaluation_split}.json"
    metrics = _read_json(metrics_path, metrics_path.name)
    provenance = metrics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{metrics_path.name} is missing provenance")
    try:
        provenance_seed = int(provenance.get("seed", -1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("metrics evaluation seed provenance is invalid") from exc
    if provenance.get("split") != evaluation_split or provenance_seed != int(evaluation_data.seed):
        raise ValueError("metrics evaluation split provenance mismatch")
    if bool(provenance.get("test_split_used_for_selection", True)):
        raise ValueError("metrics provenance indicates test-set selection")
    if str(provenance.get("contract_sha256", "")).lower() != contract_digest.lower():
        raise ValueError("metrics contract hash does not match the frozen contract")
    if str(provenance.get("benchmark_sha256", "")).lower() != file_sha256(benchmark).lower():
        raise ValueError("metrics benchmark hash does not match the benchmark")
    if contract.is_v2:
        if provenance.get("model_family") != "feasible_scheduling_proxy_v2":
            raise ValueError("v2 metrics provenance is missing model family")
        if provenance.get("decoder_schema_version") != "horizon-reachable-feasible-v2":
            raise ValueError("v2 metrics decoder schema mismatch")
        if int(provenance.get("inference_exact_lp_calls", -1)) != 0:
            raise ValueError("v2 metrics inference_exact_lp_calls must be zero")
        for field in (
            "decoder_dtype", "control_temperature", "decision_dim", "decision_groups",
            "checkpoint_sha256",
            "train_split", "train_seed", "train_scenario_id_digest",
            "validation_split", "validation_seed", "validation_scenario_id_digest",
        ):
            if field not in provenance:
                raise ValueError(f"v2 metrics provenance is missing {field}")
            expected_value = checkpoint_digest if field == "checkpoint_sha256" else checkpoint_metadata.get(field)
            if provenance[field] != expected_value:
                raise ValueError(f"v2 metrics provenance {field} does not match checkpoint")
    raw_proxy = metrics.get("raw_proxy")
    safe_proxy = metrics.get("safe_with_exact_fallback")
    if not isinstance(raw_proxy, Mapping) or not isinstance(safe_proxy, Mapping):
        raise ValueError("metrics must contain raw_proxy and safe_with_exact_fallback blocks")
    fallback_rate = float(np.mean(np.asarray(predictions["fallback_mask"], dtype=bool)))
    raw_feasible_rate = float(np.mean(np.asarray(predictions["raw_feasible_mask"], dtype=bool)))
    _close(metrics.get("fallback_rate"), fallback_rate, label="fallback_rate")
    _close(raw_proxy.get("raw_feasible_rate"), raw_feasible_rate, label="raw_feasible_rate")
    if tuple(predictions["raw_prediction"].shape) != (evaluation_spec.size, 4, contract.output_dim):
        raise ValueError("raw prediction shape does not match the proxy contract")
    if tuple(predictions["safe_prediction"].shape) != tuple(predictions["raw_prediction"].shape):
        raise ValueError("safe prediction shape does not match raw prediction shape")
    expected_features = np.asarray(evaluation_data.inputs, dtype=np.float64).copy()
    if contract.is_v2:
        expected_features[..., 3] = 0.0
    if not np.array_equal(np.asarray(predictions["features"], dtype=np.float64), expected_features):
        raise ValueError("prediction features do not match the primary physical input contract")
    if not np.array_equal(np.asarray(predictions["target"], dtype=np.float64), np.asarray(evaluation_data.dispatch, dtype=np.float64)):
        raise ValueError("prediction target does not match the evaluated split")
    if not np.array_equal(np.asarray(predictions["teacher_objective"], dtype=np.float64), np.asarray(evaluation_data.teacher_objective, dtype=np.float64)):
        raise ValueError("prediction teacher objective does not match the evaluated split")
    if contract.is_v2:
        if fallback_rate != 0.0 or raw_feasible_rate != 1.0:
            raise ValueError("v2 artifact requires zero fallback and 100% raw feasibility")
        if not np.array_equal(predictions["raw_prediction"], predictions["safe_prediction"]):
            raise ValueError("v2 raw and safe dispatch arrays must be identical")
        if int(metrics.get("inference_exact_lp_calls", -1)) != 0:
            raise ValueError("v2 metrics inference_exact_lp_calls must be zero")
        if str(metrics.get("decoder_schema_version", "")) != "horizon-reachable-feasible-v2":
            raise ValueError("v2 metrics decoder schema mismatch")
    parameters = load_benchmark(benchmark)["values"]
    expected_raw_metrics = dispatch_metrics(
        predictions["raw_prediction"],
        predictions["target"],
        predictions["features"],
        evaluation_data.teacher_cost,
        evaluation_data.teacher_carbon,
        parameters,
        stats=stats,
        feasibility_tolerance=float(contract.safety["feasibility_tolerance"]),
        teacher_objective=evaluation_data.teacher_objective,
        include_charge_discharge_overlap=contract.is_v2,
    )
    expected_safe_metrics = dispatch_metrics(
        predictions["safe_prediction"],
        predictions["target"],
        predictions["features"],
        evaluation_data.teacher_cost,
        evaluation_data.teacher_carbon,
        parameters,
        stats=stats,
        feasibility_tolerance=float(contract.safety["feasibility_tolerance"]),
        teacher_objective=evaluation_data.teacher_objective,
        include_charge_discharge_overlap=contract.is_v2,
    )
    for block_name, expected_block in (("raw_proxy", expected_raw_metrics), ("safe_with_exact_fallback", expected_safe_metrics)):
        stored_block = metrics[block_name]
        for key, expected_value in expected_block.items():
            if key not in stored_block:
                if contract.is_v2:
                    raise ValueError(f"{metrics_path.name} {block_name} is missing {key}")
                continue
            if isinstance(expected_value, (int, float, np.integer, np.floating)):
                _close(stored_block[key], expected_value, label=f"{metrics_path.name} {block_name}.{key}")
    if "safe_success_rate" in metrics:
        _close(metrics["safe_success_rate"], expected_safe_metrics["raw_feasible_rate"], label="safe_success_rate")
    if "fallback_count" in metrics and int(metrics.get("fallback_count", -1)) != int(np.asarray(predictions["fallback_mask"], dtype=bool).sum()):
        raise ValueError("fallback_count is inconsistent with prediction artifact")
    if "inference_exact_lp_calls" in metrics and int(metrics.get("inference_exact_lp_calls", -1)) != int(predictions.get("inference_exact_lp_calls", np.asarray(-1)).item()):
        raise ValueError("inference_exact_lp_calls is inconsistent with prediction artifact")
    diagnostics = diagnose_dispatch(
        predictions["raw_prediction"],
        predictions["features"],
        parameters,
        tolerance=float(contract.safety["feasibility_tolerance"]),
        scenario_ids=predictions["scenario_ids"],
    )
    raw_feasibility = evaluate_raw_feasibility(
        predictions["raw_prediction"], predictions["features"], parameters,
        tolerance=float(contract.safety["feasibility_tolerance"]),
    )
    diagnostic_raw_mask = np.asarray(diagnostics["per_sample"]["raw_feasible_mask"], dtype=bool)
    stored_raw_mask = np.asarray(predictions["raw_feasible_mask"], dtype=bool)
    if not np.array_equal(diagnostic_raw_mask, stored_raw_mask):
        raise ValueError("constraint diagnostic raw-feasible mask does not match prediction artifact")
    _close(
        diagnostics["aggregate"]["raw_feasible_rate"],
        raw_feasible_rate,
        label="diagnostic raw_feasible_rate",
    )
    if contract.is_v2:
        overlap = float(np.max(np.asarray(raw_feasibility["charge_discharge_overlap"])))
        if overlap > 1.0e-6:
            raise ValueError(f"v2 charge/discharge overlap exceeds 1e-6: {overlap}")
        for family in diagnostics["family_order"]:
            maximum = float(diagnostics["aggregate"][family]["max"])
            if maximum > 1.0e-6:
                raise ValueError(f"v2 {family} residual exceeds 1e-6: {maximum}")
    artifact_hashes = {relative: file_sha256(output / relative) for relative in artifact_names}
    report = {
        "schema_version": "scheduling-proxy-audit-v2" if contract.is_v2 else "scheduling-proxy-baseline-audit-v1",
        "status": "passed",
        "output_dir": str(output.resolve()),
        "mode": "smoke" if smoke else "formal",
        "evaluation_split": evaluation_split,
        "source_type": contract.source_type,
        "generator_version": contract.generator_version,
        "contract_sha256": contract_digest,
        "benchmark_sha256": file_sha256(benchmark),
        "checkpoint_sha256": checkpoint_digest,
        "split_sizes": dict(dataset_manifest["split_sizes"]),
        "split_seeds": dict(dataset_manifest["split_seeds"]),
        "test_split_used_for_selection": False,
        "evaluation_output_shape": list(predictions["raw_prediction"].shape),
        "raw_feasible_rate": raw_feasible_rate,
        "fallback_rate": fallback_rate,
        "model_family": "feasible_scheduling_proxy_v2" if contract.is_v2 else "scheduling_proxy_v1",
        "decoder_schema_version": contract.decoder_schema_version,
        "inference_exact_lp_calls": int(metrics.get("inference_exact_lp_calls", 0)),
        "safe_success_rate": float(metrics.get("safe_success_rate", 0.0)),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "best_validation_loss": float(checkpoint.get("validation_loss", float("nan"))),
        "normalization_fit_split": stats.fit_split,
        "constraint_diagnostics": {
            "schema_version": diagnostics["schema_version"],
            "value_space": diagnostics["value_space"],
            "family_order": diagnostics["family_order"],
            "aggregate": diagnostics["aggregate"],
        },
        "artifact_sha256": artifact_hashes,
    }
    # Validation-only V2 smoke artifacts must not imply that test data was
    # evaluated.  Preserve the historical key for V1 and formal test audits.
    if not (contract.is_v2 and smoke):
        report["test_output_shape"] = list(predictions["raw_prediction"].shape)
    if not np.isfinite(report["best_validation_loss"]):
        raise ValueError("checkpoint validation loss is not finite")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--report-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_proxy_artifact(args.output_dir, args.benchmark, args.contract)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output_dir)
    benchmark = Path(args.benchmark)
    contract_file = Path(args.contract)
    contract = load_contract(contract_file, benchmark)
    manifest = validate_dataset_manifest(output / "dataset" / "manifest.json", contract, benchmark, smoke=None)
    smoke = bool(manifest["smoke"])
    evaluation_split = _evaluation_split(contract, smoke)
    evaluation_spec = contract.split(evaluation_split, smoke=smoke)
    evaluation_data = load_proxy_split(
        output / "dataset" / f"{evaluation_split}.npz",
        expected_split=evaluation_split,
        expected_seed=evaluation_spec.seed,
        expected_count=evaluation_spec.size,
        expected_contract=contract,
        expected_benchmark_sha256=file_sha256(benchmark),
        expected_smoke=smoke,
    )
    _, _, checkpoint = load_trained_proxy(
        output,
        contract=contract,
        benchmark_path=benchmark,
        expected_train_seed=contract.split("train", smoke=smoke).seed,
    )
    checkpoint_metadata = checkpoint.get("metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError("checkpoint metadata is missing")
    checkpoint_digest = file_sha256(output / "best_model.pt")
    prediction = load_prediction_artifact(
        output / f"predictions_{evaluation_split}.npz",
        expected_split=evaluation_split,
        expected_count=evaluation_spec.size,
        expected_seed=evaluation_spec.seed,
        expected_scenario_ids=evaluation_data.scenario_ids,
        expected_scenario_id_digest=str(manifest["splits"][evaluation_split]["scenario_id_digest"]),
        expected_benchmark_sha256=file_sha256(benchmark),
        expected_contract_sha256=contract_sha256(contract),
        expected_source_type=contract.source_type,
        expected_generator_version=contract.generator_version,
        expected_model_family="feasible_scheduling_proxy_v2" if contract.is_v2 else None,
        expected_decoder_schema_version=contract.decoder_schema_version if contract.is_v2 else None,
        expected_inference_exact_lp_calls=0 if contract.is_v2 else None,
        expected_checkpoint_sha256=checkpoint_digest if contract.is_v2 else None,
        expected_train_seed=int(checkpoint_metadata["train_seed"]) if contract.is_v2 else None,
        expected_validation_seed=int(checkpoint_metadata["validation_seed"]) if contract.is_v2 else None,
        expected_train_scenario_id_digest=str(checkpoint_metadata["train_scenario_id_digest"]) if contract.is_v2 else None,
        expected_validation_scenario_id_digest=str(checkpoint_metadata["validation_scenario_id_digest"]) if contract.is_v2 else None,
    )
    diagnostics = diagnose_dispatch(
        prediction["raw_prediction"],
        prediction["features"],
        load_benchmark(benchmark)["values"],
        tolerance=float(contract.safety["feasibility_tolerance"]),
        scenario_ids=prediction["scenario_ids"],
    )
    diagnostic_npz = report_dir / "proxy_constraint_diagnostics.npz"
    np.savez_compressed(diagnostic_npz, **diagnostics_npz_payload(diagnostics))
    diagnostic_json = report_dir / "proxy_constraint_diagnostics.json"
    diagnostic_summary = {
        "schema_version": diagnostics["schema_version"],
        "value_space": diagnostics["value_space"],
        "sample_count": diagnostics["sample_count"],
        "horizon": diagnostics["horizon"],
        "family_order": diagnostics["family_order"],
        "aggregate": diagnostics["aggregate"],
        "npz": str(diagnostic_npz.resolve()),
        "npz_sha256": file_sha256(diagnostic_npz),
    }
    diagnostic_json.write_text(json.dumps(diagnostic_summary, indent=2, sort_keys=True), encoding="utf-8")
    report["constraint_diagnostics"]["npz"] = str(diagnostic_npz.resolve())
    report["constraint_diagnostics"]["npz_sha256"] = file_sha256(diagnostic_npz)
    report["constraint_diagnostics"]["summary_json"] = str(diagnostic_json.resolve())
    report_path = report_dir / "proxy_baseline_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
