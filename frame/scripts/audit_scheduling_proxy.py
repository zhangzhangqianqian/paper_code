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
from src.scheduling.proxy_dataset import load_proxy_split, validate_dataset_manifest  # noqa: E402
from src.scheduling.proxy_diagnostics import diagnose_dispatch, diagnostics_npz_payload  # noqa: E402
from src.scheduling.proxy_evaluation import load_prediction_artifact  # noqa: E402
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


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _require_files(output_dir: Path) -> None:
    missing = [relative for relative in FORMAL_ARTIFACTS if not (output_dir / relative).is_file()]
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
    _require_files(output)
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
    test_spec = contract.split("test", smoke=smoke)
    test_split = load_proxy_split(
        output / "dataset" / "test.npz",
        expected_split="test",
        expected_seed=test_spec.seed,
        expected_count=test_spec.size,
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
    prediction_path = output / "predictions_test.npz"
    predictions = load_prediction_artifact(
        prediction_path,
        expected_split="test",
        expected_count=test_spec.size,
        expected_seed=test_spec.seed,
        expected_scenario_ids=test_split.scenario_ids,
        expected_scenario_id_digest=str(dataset_manifest["splits"]["test"]["scenario_id_digest"]),
        expected_benchmark_sha256=file_sha256(benchmark),
        expected_contract_sha256=contract_digest,
        expected_source_type=contract.source_type,
        expected_generator_version=contract.generator_version,
    )
    metrics = _read_json(output / "metrics_test.json", "metrics_test.json")
    provenance = metrics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("metrics_test.json is missing provenance")
    if bool(provenance.get("test_split_used_for_selection", True)):
        raise ValueError("metrics provenance indicates test-set selection")
    if str(provenance.get("contract_sha256", "")).lower() != contract_digest.lower():
        raise ValueError("metrics contract hash does not match the frozen contract")
    if str(provenance.get("benchmark_sha256", "")).lower() != file_sha256(benchmark).lower():
        raise ValueError("metrics benchmark hash does not match the benchmark")
    raw_proxy = metrics.get("raw_proxy")
    safe_proxy = metrics.get("safe_with_exact_fallback")
    if not isinstance(raw_proxy, Mapping) or not isinstance(safe_proxy, Mapping):
        raise ValueError("metrics must contain raw_proxy and safe_with_exact_fallback blocks")
    fallback_rate = float(np.mean(np.asarray(predictions["fallback_mask"], dtype=bool)))
    raw_feasible_rate = float(np.mean(np.asarray(predictions["raw_feasible_mask"], dtype=bool)))
    _close(metrics.get("fallback_rate"), fallback_rate, label="fallback_rate")
    _close(raw_proxy.get("raw_feasible_rate"), raw_feasible_rate, label="raw_feasible_rate")
    if tuple(predictions["raw_prediction"].shape) != (test_spec.size, 4, contract.output_dim):
        raise ValueError("raw prediction shape does not match the proxy contract")
    if tuple(predictions["safe_prediction"].shape) != tuple(predictions["raw_prediction"].shape):
        raise ValueError("safe prediction shape does not match raw prediction shape")
    parameters = load_benchmark(benchmark)["values"]
    diagnostics = diagnose_dispatch(
        predictions["raw_prediction"],
        predictions["features"],
        parameters,
        tolerance=float(contract.safety["feasibility_tolerance"]),
        scenario_ids=predictions["scenario_ids"],
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
    artifact_hashes = {relative: file_sha256(output / relative) for relative in FORMAL_ARTIFACTS}
    report = {
        "schema_version": "scheduling-proxy-baseline-audit-v1",
        "status": "passed",
        "output_dir": str(output.resolve()),
        "mode": "smoke" if smoke else "formal",
        "source_type": contract.source_type,
        "generator_version": contract.generator_version,
        "contract_sha256": contract_digest,
        "benchmark_sha256": file_sha256(benchmark),
        "split_sizes": dict(dataset_manifest["split_sizes"]),
        "split_seeds": dict(dataset_manifest["split_seeds"]),
        "test_split_used_for_selection": False,
        "test_output_shape": list(predictions["raw_prediction"].shape),
        "raw_feasible_rate": raw_feasible_rate,
        "fallback_rate": fallback_rate,
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
    test_spec = contract.split("test", smoke=smoke)
    test_split = load_proxy_split(
        output / "dataset" / "test.npz",
        expected_split="test",
        expected_seed=test_spec.seed,
        expected_count=test_spec.size,
        expected_contract=contract,
        expected_benchmark_sha256=file_sha256(benchmark),
        expected_smoke=smoke,
    )
    prediction = load_prediction_artifact(
        output / "predictions_test.npz",
        expected_split="test",
        expected_count=test_spec.size,
        expected_seed=test_spec.seed,
        expected_scenario_ids=test_split.scenario_ids,
        expected_scenario_id_digest=str(manifest["splits"]["test"]["scenario_id_digest"]),
        expected_benchmark_sha256=file_sha256(benchmark),
        expected_contract_sha256=contract_sha256(contract),
        expected_source_type=contract.source_type,
        expected_generator_version=contract.generator_version,
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
