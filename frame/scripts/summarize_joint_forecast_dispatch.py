"""Build compact, machine-readable tables from joint forecast--dispatch receipts.

The runner stores the complete raw time series in JSON receipts so that the
selection and audit protocol remain reproducible.  This utility creates small
CSV tables for manuscript review without recomputing or changing any model.
It deliberately reports ablation names as a protocol registry only; an
ablation row is never treated as an observed result unless a receipt exists.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.contract import TASK_ORDER, load_joint_training_contract
from src.joint_dispatch.evaluation import paired_moving_block_bootstrap


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _receipts(root: Path, contract) -> dict[tuple[str, int], dict[str, object]]:
    result: dict[tuple[str, int], dict[str, object]] = {}
    validation = root / "validation"
    for variant in ("frozen_pto", "joint_from_scratch", "warm_started_joint"):
        for seed in contract.seeds:
            path = validation / variant / f"seed_{seed}" / "validation_receipt.json"
            if path.exists():
                result[(variant, int(seed))] = _read(path)
    test_path = root / "test" / "test_receipt.json"
    if test_path.exists():
        result[("sealed_test", int(_read(test_path).get("selected_seed", 0)))] = _read(test_path)
    return result


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _variant_name(receipt: dict[str, object], fallback: str) -> str:
    return str(receipt.get("variant", fallback))


def build_tables(root: Path, contract_path: Path) -> dict[str, object]:
    contract = load_joint_training_contract(contract_path)
    receipts = _receipts(root, contract)
    out = root / "evaluation_tables"
    forecast_rows: list[dict[str, object]] = []
    dispatch_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    by_variant: dict[str, list[dict[str, object]]] = {}
    for (variant, seed), receipt in sorted(receipts.items()):
        metrics = receipt.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        name = _variant_name(receipt, variant)
        by_variant.setdefault(name, []).append(receipt)
        maes = np.asarray(metrics.get("forecast_mae", []), dtype=float)
        rmses = np.asarray(metrics.get("forecast_rmse", []), dtype=float)
        wapes = np.asarray(metrics.get("forecast_wape", []), dtype=float)
        if maes.ndim == 2 and rmses.shape == maes.shape and wapes.shape == maes.shape:
            for horizon in range(maes.shape[0]):
                for task_index, task in enumerate(TASK_ORDER):
                    forecast_rows.append({
                        "variant": name,
                        "seed": seed,
                        "horizon": horizon + 1,
                        "task": task,
                        "mae": float(maes[horizon, task_index]),
                        "rmse": float(rmses[horizon, task_index]),
                        "wape": float(wapes[horizon, task_index]),
                    })
        summary = metrics.get("closed_loop_summary", {})
        if isinstance(summary, dict):
            dispatch_rows.append({
                "variant": name,
                "seed": seed,
                "steps": metrics.get("steps"),
                "operating_cost": summary.get("operating_cost"),
                "physical_carbon": summary.get("physical_carbon"),
                "shortage": summary.get("shortage"),
                "penalized_objective": summary.get("penalized_objective"),
                "feasibility_rate": summary.get("feasibility_rate"),
                "decision_regret": metrics.get("decision_regret"),
                "online_exact_lp_calls": metrics.get("online_exact_lp_calls"),
            })
        audit = receipt.get("gradient_audit", {})
        rollin = receipt.get("rollin_refresh", {})
        gradient_rows.append({
            "variant": name,
            "seed": seed,
            "forecaster_norm_from_decision_only": audit.get("forecaster_norm_from_decision_only") if isinstance(audit, dict) else None,
            "scheduler_norm_from_total": audit.get("scheduler_norm_from_total") if isinstance(audit, dict) else None,
            "rollin_implemented": rollin.get("implemented") if isinstance(rollin, dict) else None,
            "rollin_executed_steps": rollin.get("executed_steps") if isinstance(rollin, dict) else None,
        })
        state_rows.append({
            "variant": name,
            "seed": seed,
            "complete": receipt.get("complete"),
            "formal": receipt.get("formal"),
            "test_set_accessed": receipt.get("test_set_accessed"),
            "forecast_finite": metrics.get("forecast_finite"),
            "state_finite": metrics.get("state_finite"),
            "online_exact_lp_calls": metrics.get("online_exact_lp_calls"),
            "steps": metrics.get("steps"),
        })

    _write_csv(out / "forecast_by_task_horizon.csv", ("variant", "seed", "horizon", "task", "mae", "rmse", "wape"), forecast_rows)
    _write_csv(out / "dispatch_operational_metrics.csv", (
        "variant", "seed", "steps", "operating_cost", "physical_carbon", "shortage",
        "penalized_objective", "feasibility_rate", "decision_regret", "online_exact_lp_calls",
    ), dispatch_rows)
    _write_csv(out / "gradient_coupling.csv", (
        "variant", "seed", "forecaster_norm_from_decision_only", "scheduler_norm_from_total",
        "rollin_implemented", "rollin_executed_steps",
    ), gradient_rows)
    _write_csv(out / "closed_loop_state_diagnostics.csv", (
        "variant", "seed", "complete", "formal", "test_set_accessed", "forecast_finite",
        "state_finite", "online_exact_lp_calls", "steps",
    ), state_rows)

    comparison_rows: list[dict[str, object]] = []
    pairs = (("joint_from_scratch", "frozen_pto"), ("warm_started_joint", "joint_from_scratch"), ("warm_started_joint", "frozen_pto"))
    for method_a, method_b in pairs:
        if not all((method_a, int(seed)) in receipts and (method_b, int(seed)) in receipts for seed in contract.seeds):
            continue
        series_a = np.asarray([receipts[(method_a, int(seed))]["metrics"]["decision_regret_series"] for seed in contract.seeds], dtype=float)
        series_b = np.asarray([receipts[(method_b, int(seed))]["metrics"]["decision_regret_series"] for seed in contract.seeds], dtype=float)
        # The frozen diagnostic includes the final three non-executable origins,
        # whereas joint roll-in reports executable origins only.  Pair the
        # common causal prefix rather than padding or silently dropping a seed.
        common_hours = min(series_a.shape[1], series_b.shape[1])
        series_a, series_b = series_a[:, :common_hours], series_b[:, :common_hours]
        result = paired_moving_block_bootstrap(series_a, series_b, block_hours=168, replicates=2000, seed=2026)
        comparison_rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "metric": "decision_regret_a_minus_b",
            **result,
        })
    _write_csv(out / "paired_model_comparisons.csv", (
        "method_a", "method_b", "metric", "observed_difference", "ci_lower", "ci_upper",
        "alpha", "replicates", "block_hours", "dependence_model",
    ), comparison_rows)

    ablation_dir = root / "ablations"
    ablation_rows = []
    for name, role in (
        ("No-Device-State", "remove historical device-state stream"),
        ("No-Decision-Loss", "remove realized decision loss from joint objective"),
        ("Frozen-Forecaster", "freeze Scheme2R during scheduler optimization"),
        ("No-Gas-Prior", "remove auxiliary station-side gas-prior target"),
        ("Direct-Dispatch-Diagnostic-Only", "bypass feasible decoder; diagnostic only"),
    ):
        ablation_rows.append({"name": name, "role": role, "status": "not_run", "evidence": "no receipt; not used in claims"})
    _write_csv(ablation_dir / "ablation_registry.csv", ("name", "role", "status", "evidence"), ablation_rows)
    manifest = {
        "schema_version": contract.schema_version,
        "complete": True,
        "source_receipts": len(receipts),
        "tables": sorted(str(path.relative_to(root)) for path in out.glob("*.csv")),
        "ablation_registry": str((ablation_dir / "ablation_registry.csv").relative_to(root)),
        "ablation_results_claimed": False,
        "selection_receipt": str((root / "selection" / "selection_receipt.json").relative_to(root)),
        "test_receipt": str((root / "test" / "test_receipt.json").relative_to(root)),
    }
    (out / "evaluation_tables_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=Path("reports/joint_forecast_dispatch_v1"))
    parser.add_argument("--contract", type=Path, default=Path("configs/joint_forecast_dispatch_contract_v1.json"))
    args = parser.parse_args()
    manifest = build_tables(args.report_root, args.contract)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
