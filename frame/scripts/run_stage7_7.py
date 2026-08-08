"""Stage 7.7 addendum: formal same-track multi-task baseline experiments.

The addendum does not modify the Stage 6.6 freeze or any existing Stage 7-R
directory.  It adds Hard-Share-H2, Dynamic-Symmetric-H1 and a preregistered
two-level PLE-lite under the same two protocols and five formal seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_stage7_3 import (  # noqa: E402
    FORMAL_PROTOCOLS,
    _aggregate_rows,
    _read_json,
    _resolve,
    _run_one,
    _standardize_protocol,
    _summary_row,
    _write_csv,
)
from src.data_pipeline import save_json  # noqa: E402
from src.kitakyushu_pipeline import (  # noqa: E402
    KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_TASKS,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)
from src.stage7_contract import EXPECTED_SEEDS  # noqa: E402


EXPECTED_CANDIDATES: Dict[str, str] = {
    "hard_share": "H2",
    "dynamic_symmetric": "H1",
    "ple-lite": "ple_lite_fixed_v1",
}
EXPECTED_MODELS: Tuple[str, ...] = tuple(EXPECTED_CANDIDATES)
EXPECTED_FORMAL_RUNS = 30
EXPECTED_SMOKE_RUNS = 6
REQUIRED_RUN_FILES: Tuple[str, ...] = (
    "best_model.pt",
    "normalization_stats.npz",
    "history.json",
    "metrics_validation.json",
    "metrics_test.json",
    "predictions_validation.npz",
    "predictions_test.npz",
    "run_manifest.json",
)


def _model_dir_name(model: str) -> str:
    return model.replace("-", "_")


def build_stage7_7_run_plan(
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> Tuple[Dict[str, object], ...]:
    frozen_seeds = tuple(int(seed) for seed in seeds)
    if frozen_seeds != tuple(EXPECTED_SEEDS):
        raise ValueError(
            f"Stage 7.7 requires frozen seeds {EXPECTED_SEEDS}, got {frozen_seeds}"
        )
    rows = tuple(
        {
            "protocol": protocol,
            "model": model,
            "candidate_id": candidate_id,
            "seed": seed,
            "run_id": (
                f"{protocol}/{_model_dir_name(model)}/{candidate_id}/seed_{seed}"
            ),
        }
        for protocol in FORMAL_PROTOCOLS
        for model, candidate_id in EXPECTED_CANDIDATES.items()
        for seed in frozen_seeds
    )
    if len(rows) != EXPECTED_FORMAL_RUNS:
        raise RuntimeError("Stage 7.7 formal run matrix must contain 30 runs")
    if len({str(row["run_id"]) for row in rows}) != len(rows):
        raise RuntimeError("Stage 7.7 run IDs must be unique")
    return rows


def _model_contract_map(
    contract: Mapping[str, object],
) -> Dict[str, Mapping[str, object]]:
    models = contract.get("models")
    if not isinstance(models, list):
        raise ValueError("Stage 7.7 contract models must be a list")
    result: Dict[str, Mapping[str, object]] = {}
    for item in models:
        if not isinstance(item, Mapping):
            raise ValueError("each Stage 7.7 model contract must be an object")
        model = str(item.get("model", ""))
        if not model or model in result:
            raise ValueError("Stage 7.7 contract model names must be unique")
        result[model] = item
    return result


def validate_stage7_7_contract(contract: Mapping[str, object]) -> None:
    if contract.get("contract_version") != "stage7.7-addendum-v1":
        raise ValueError("unexpected Stage 7.7 contract version")
    if contract.get("contract_status") != "frozen_before_stage7_7_test_evaluation":
        raise ValueError("Stage 7.7 contract is not frozen")
    if contract.get("dataset") != "kitakyushu_energy_station":
        raise ValueError("Stage 7.7 is frozen for Kitakyushu")
    if tuple(contract.get("tasks", ())) != KITAKYUSHU_TASKS:
        raise ValueError("Stage 7.7 task order is invalid")
    if tuple(contract.get("protocols", ())) != FORMAL_PROTOCOLS:
        raise ValueError("Stage 7.7 protocols are invalid")
    if tuple(contract.get("seeds", ())) != tuple(EXPECTED_SEEDS):
        raise ValueError("Stage 7.7 seeds are invalid")
    if int(contract.get("expected_formal_run_count", -1)) != EXPECTED_FORMAL_RUNS:
        raise ValueError("Stage 7.7 expected run count must be 30")

    io_contract = contract.get("input_output")
    if not isinstance(io_contract, Mapping):
        raise ValueError("Stage 7.7 input_output contract is missing")
    if io_contract.get("lookback") != 24 or io_contract.get("horizon") != 4:
        raise ValueError("Stage 7.7 requires the frozen 24-to-4 window")
    if io_contract.get("input_mode") != "loads_and_exog":
        raise ValueError("Stage 7.7 joint baselines require historical exogenous input")
    if io_contract.get("future_exogenous_allowed") is not False:
        raise ValueError("future exogenous variables are forbidden")

    policy = contract.get("training_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("Stage 7.7 training policy is missing")
    expected_policy = {
        "device": "cpu",
        "loss": "SmoothL1Loss",
        "optimizer": "AdamW",
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "threads": 8,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(
                f"Stage 7.7 training policy mismatch for {key}: {policy.get(key)!r}"
            )
    protocol_policies = policy.get("protocols")
    if not isinstance(protocol_policies, Mapping):
        raise ValueError("Stage 7.7 protocol-specific training policies are missing")
    expected_protocol_policies = {
        "full": {"batch_size": 256, "max_epochs": 100, "early_stopping_patience": 12},
        "small_sample": {"batch_size": 32, "max_epochs": 200, "early_stopping_patience": 20},
    }
    for protocol, expected_values in expected_protocol_policies.items():
        actual = protocol_policies.get(protocol)
        if not isinstance(actual, Mapping):
            raise ValueError(f"missing training policy for {protocol}")
        for key, expected in expected_values.items():
            if actual.get(key) != expected:
                raise ValueError(
                    f"Stage 7.7 {protocol} training policy mismatch for {key}: "
                    f"{actual.get(key)!r}"
                )

    model_map = _model_contract_map(contract)
    if set(model_map) != set(EXPECTED_MODELS):
        raise ValueError("Stage 7.7 model set is invalid")
    for model, candidate_id in EXPECTED_CANDIDATES.items():
        item = model_map[model]
        if item.get("candidate_id") != candidate_id:
            raise ValueError(f"wrong Stage 7.7 candidate for {model}")
        if not isinstance(item.get("hyperparameters"), Mapping):
            raise ValueError(f"missing Stage 7.7 hyperparameters for {model}")

    test_policy = contract.get("test_policy")
    if not isinstance(test_policy, Mapping):
        raise ValueError("Stage 7.7 test policy is missing")
    if test_policy.get("test_used_for_selection") is not False:
        raise ValueError("Stage 7.7 test set cannot be used for selection")
    if test_policy.get("test_may_not_change_model_or_hyperparameters") is not True:
        raise ValueError("Stage 7.7 test results must not change configurations")


def _validate_against_stage6_freeze(
    contract: Mapping[str, object],
    freeze: Mapping[str, object],
) -> None:
    if freeze.get("freeze_version") != "stage6.6":
        raise ValueError("Stage 7.7 requires the Stage 6.6 freeze")
    ranking = freeze.get("selection_ranking")
    if not isinstance(ranking, list):
        raise ValueError("Stage 6.6 selection ranking is missing")
    best_by_model: Dict[str, str] = {}
    for row in ranking:
        if not isinstance(row, Mapping):
            continue
        model = str(row.get("model", ""))
        if model in {"hard_share", "dynamic_symmetric"} and model not in best_by_model:
            best_by_model[model] = str(row.get("candidate_id", ""))
    expected = {
        "hard_share": EXPECTED_CANDIDATES["hard_share"],
        "dynamic_symmetric": EXPECTED_CANDIDATES["dynamic_symmetric"],
    }
    if best_by_model != expected:
        raise ValueError(
            f"Stage 7.7 selected candidates do not match Stage 6: {best_by_model}"
        )
    contract_models = _model_contract_map(contract)
    if contract_models["ple-lite"].get("selection_source") != (
        "preregistered_fixed_configuration"
    ):
        raise ValueError("PLE-lite must use the preregistered fixed configuration")


def _limit_windows(
    windows: Mapping[str, np.ndarray], limit: int,
) -> Dict[str, np.ndarray]:
    if limit <= 0:
        raise ValueError("smoke sample limit must be positive")
    return {key: value[:limit] for key, value in windows.items()}


def _run_directory(output_dir: Path, run: Mapping[str, object]) -> Path:
    return (
        output_dir
        / str(run["protocol"])
        / _model_dir_name(str(run["model"]))
        / str(run["candidate_id"])
        / f"seed_{run['seed']}"
    )


def _strict_passed_run(
    run_dir: Path,
    run: Mapping[str, object],
) -> Dict[str, object] | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    expected = {
        "stage": "7.7",
        "status": "passed",
        "protocol": run["protocol"],
        "model": run["model"],
        "candidate_id": run["candidate_id"],
        "seed": run["seed"],
        "test_used_for_selection": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if not all((run_dir / name).is_file() for name in REQUIRED_RUN_FILES):
        return None
    with np.load(run_dir / "predictions_test.npz", allow_pickle=False) as values:
        prediction = np.asarray(values["prediction"])
        target = np.asarray(values["target"])
    if prediction.shape != target.shape or prediction.ndim != 3:
        return None
    if tuple(prediction.shape[1:]) != (4, 4):
        return None
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        return None
    manifest["run_dir"] = str(run_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 7.7 joint multi-task baseline addendum"
    )
    parser.add_argument(
        "--kitakyushu-data-dir", default="D:/Paper/Kitakyushu dataset"
    )
    parser.add_argument(
        "--contract",
        default="frame/configs/stage7_7_joint_baselines_contract.json",
    )
    parser.add_argument(
        "--freeze-config",
        default="frame/reports/stage6r_6_kitakyushu/stage6_selected_config.json",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path = _resolve(args.contract)
    freeze_path = _resolve(args.freeze_config)
    contract = _read_json(contract_path)
    freeze = _read_json(freeze_path)
    validate_stage7_7_contract(contract)
    _validate_against_stage6_freeze(contract, freeze)

    formal_plan = build_stage7_7_run_plan()
    plan = (
        tuple(row for row in formal_plan if int(row["seed"]) == EXPECTED_SEEDS[0])
        if args.smoke
        else formal_plan
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "7.7",
                    "status": "dry_run",
                    "mode": "smoke" if args.smoke else "formal",
                    "run_count": len(plan),
                    "models": list(EXPECTED_MODELS),
                    "candidates": EXPECTED_CANDIDATES,
                    "protocols": list(FORMAL_PROTOCOLS),
                    "seeds": sorted({int(row["seed"]) for row in plan}),
                    "test_used_for_selection": False,
                    "runs": list(plan),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    default_output = (
        "frame/reports/stage7r_7_joint_baselines_smoke"
        if args.smoke
        else "frame/reports/stage7r_7_joint_baselines_formal"
    )
    output_dir = _resolve(args.output_dir or default_output)
    manifest_path = output_dir / "stage7_7_manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"Stage 7.7 output exists; use --resume: {manifest_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    read_started = time.perf_counter()
    raw_frame, source_metadata = read_kitakyushu_canonical(
        _resolve(args.kitakyushu_data_dir), years=tuple(range(2015, 2022))
    )
    frame, cleaning_report = clean_kitakyushu_dataframe(raw_frame)
    read_seconds = time.perf_counter() - read_started
    specs = {
        "full": KITAKYUSHU_SPLIT,
        "small_sample": KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    }
    protocol_windows: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    protocol_stats = {}
    for protocol, spec in specs.items():
        windows, stats = _standardize_protocol(frame, spec)
        if args.smoke:
            windows = {
                "train": _limit_windows(windows["train"], 64),
                "validation": _limit_windows(windows["validation"], 32),
                "test": _limit_windows(windows["test"], 32),
            }
        protocol_windows[protocol] = windows
        protocol_stats[protocol] = stats
        stats.save(output_dir / protocol / "normalization_stats.npz")

    model_contracts = _model_contract_map(contract)
    hyperparameters = {
        (model, str(item["candidate_id"])): dict(item["hyperparameters"])
        for model, item in model_contracts.items()
    }
    base_policy = dict(contract["training_policy"])
    protocol_policy_overrides = base_policy.pop("protocols")
    policies = {
        protocol: {**base_policy, **dict(protocol_policy_overrides[protocol])}
        for protocol in FORMAL_PROTOCOLS
    }
    if args.smoke:
        for policy in policies.values():
            policy.update({
                "batch_size": 16,
                "max_epochs": 1,
                "early_stopping_patience": 1,
                "threads": 2,
            })

    completed: List[Dict[str, object]] = []
    failed: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    for run in plan:
        protocol = str(run["protocol"])
        run_dir = _run_directory(output_dir, run)
        if args.resume:
            existing = _strict_passed_run(run_dir, run)
            if existing is not None:
                completed.append(existing)
                rows.append(_summary_row(existing))
                print(
                    json.dumps(
                        {
                            "stage": "7.7",
                            "status": "run_reused",
                            "run_id": run["run_id"],
                            "completed": len(completed),
                            "total": len(plan),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            if (run_dir / "run_manifest.json").exists():
                raise ValueError(f"cannot resume invalid Stage 7.7 run: {run_dir}")
        try:
            result = _run_one(
                run,
                run_dir,
                protocol_windows[protocol],
                protocol_stats[protocol],
                hyperparameters,
                policies[protocol],
                stage_label="7.7",
                stage_role="joint_multitask_baseline",
            )
            result.update(
                {
                    "test_used_for_selection": False,
                    "selection_source": model_contracts[str(run["model"])][
                        "selection_source"
                    ],
                    "formal_training": not args.smoke,
                    "smoke": bool(args.smoke),
                    "run_dir": str(run_dir),
                }
            )
            save_json(result, run_dir / "run_manifest.json")
            completed.append(result)
            rows.append(_summary_row(result))
            print(
                json.dumps(
                    {
                        "stage": "7.7",
                        "status": "run_completed",
                        "run_id": run["run_id"],
                        "completed": len(completed),
                        "total": len(plan),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            error = {
                "stage": "7.7",
                "status": "failed",
                "run_id": run["run_id"],
                "protocol": protocol,
                "model": run["model"],
                "candidate_id": run["candidate_id"],
                "seed": run["seed"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            save_json(error, run_dir / "error.json")
            failed.append(error)
            print(json.dumps(error, ensure_ascii=False), flush=True)

    if rows:
        _write_csv(rows, output_dir / "metrics_by_run.csv")
        _write_csv(
            _aggregate_rows(rows), output_dir / "overall_comparison_mean_std.csv"
        )
    status = "passed" if not failed and len(completed) == len(plan) else "failed"
    manifest = {
        "stage": "7.7",
        "status": status,
        "mode": "smoke" if args.smoke else "formal",
        "contract": str(contract_path),
        "freeze_config": str(freeze_path),
        "dataset": "kitakyushu_energy_station",
        "data_source": source_metadata,
        "cleaning_report": cleaning_report,
        "read_seconds": float(read_seconds),
        "models": list(EXPECTED_MODELS),
        "candidates": EXPECTED_CANDIDATES,
        "protocols": list(FORMAL_PROTOCOLS),
        "seeds": sorted({int(row["seed"]) for row in plan}),
        "candidate_run_count": len(plan),
        "run_count_expected": EXPECTED_SMOKE_RUNS if args.smoke else EXPECTED_FORMAL_RUNS,
        "run_count_completed": len(completed),
        "run_count_failed": len(failed),
        "formal_training": not args.smoke,
        "test_set_accessed": True,
        "test_used_for_selection": False,
        "future_exogenous_used": False,
        "completed_runs": completed,
        "failed_runs": failed,
        "summary_files": [
            "metrics_by_run.csv",
            "overall_comparison_mean_std.csv",
        ],
    }
    save_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
