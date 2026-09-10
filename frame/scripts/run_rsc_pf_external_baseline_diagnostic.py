"""Run a bounded, real-data diagnostic for one formal external baseline.

This command is intentionally not a paper-result runner.  It trains on the
materialized 2015--2018 windows, rolls out on a bounded prefix of 2019, and
marks every artifact as ``diagnostic``/``paper_result=false``.  The purpose is
to prove that the external-baseline training and causal LP evaluation path is
real before spending the full formal budget.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from scripts.run_rsc_pf_formal_v4_2_pilot import _scaled_parameters  # noqa: E402
from src.joint_dispatch.formal_v4_2_data import fit_train_normalization  # noqa: E402
from src.joint_dispatch.formal_v4_2_gate2_training import (  # noqa: E402
    Gate2DataBundle,
    _manifest_hash,
    train_direct_policy,
    train_differentiable_lp,
    train_official_itransformer_pto,
    train_scheme2r_pto,
)
from src.joint_dispatch.formal_v4_2_training import StageBudgetV42  # noqa: E402
from src.joint_dispatch.formal_v4_data import FormalV4WindowSplit  # noqa: E402
from src.joint_dispatch.formal_v4_2_gate2_execution import (  # noqa: E402
    Gate2RowKey,
    _perfect_information_method,
    execute_gate2_row,
    _registered_from_artifact,
)
from src.joint_dispatch.formal_v4_2_methods import build_v42_method  # noqa: E402
from src.joint_dispatch.formal_v4_method_adapter import build_formal_v4_method_adapter  # noqa: E402
from src.joint_dispatch.formal_v4_diffopt import DifferentiableIESLayer  # noqa: E402


def _load_capacity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("selected"), dict):
        raise ValueError("capacity receipt lacks selected capacity")
    return payload


def _build_data(materialized_root: Path, max_eval_origins: int, max_train_origins: int | None = None) -> Gate2DataBundle:
    data_root = materialized_root / "data"
    def load_materialized(path: Path, split: str) -> FormalV4WindowSplit:
        with np.load(path, allow_pickle=False) as payload:
            names = (
                "load_history", "exog_history", "renewable_history", "device_history",
                "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
                "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
                "target_times", "trajectory_ids", "state_hashes",
            )
            return FormalV4WindowSplit(
                **{name: payload[name] for name in names},
                split=split,
            )
    train = load_materialized(data_root / "normalization_source.npz", "train")
    selection = load_materialized(data_root / "selection_full.npz", "selection")
    if max_train_origins is not None:
        if max_train_origins <= 0 or max_train_origins > len(train):
            raise ValueError("max_train_origins must be within the 2015-2018 training split")
        train = replace(train, **{
            name: getattr(train, name)[:max_train_origins]
            for name in (
                "load_history", "exog_history", "renewable_history", "device_history",
                "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
                "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
                "target_times", "trajectory_ids", "state_hashes",
            )
        })
    if train.split != "train" or selection.split != "selection":
        raise ValueError("materialized diagnostic splits have unexpected split labels")
    if max_eval_origins <= 0 or max_eval_origins > len(selection):
        raise ValueError("max_eval_origins must be within the 2019 selection split")
    evaluation = replace(selection, **{
        name: getattr(selection, name)[:max_eval_origins]
        for name in (
            "load_history", "exog_history", "renewable_history", "device_history",
            "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
            "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
            "target_times", "trajectory_ids", "state_hashes",
        )
    })
    normalization = fit_train_normalization(train)
    return Gate2DataBundle(
        train=train,
        calibration=evaluation,
        evaluation=evaluation,
        normalization=normalization,
        train_manifest_sha256=_manifest_hash(train, "diagnostic_train_2015_2018"),
        calibration_manifest_sha256=_manifest_hash(evaluation, "diagnostic_selection_prefix"),
        evaluation_manifest_sha256=_manifest_hash(evaluation, "diagnostic_selection_prefix"),
        normalization_sha256=normalization.receipt_sha256,
    )


def run_diagnostic(
    *,
    materialized_root: Path,
    benchmark: Path,
    capacity_receipt: Path,
    output_root: Path,
    run_id: str,
    seed: int,
    method_id: str,
    epochs: int,
    max_eval_origins: int,
    max_train_origins: int | None = None,
    itransformer_source: Path | None = None,
    itransformer_receipt: Path | None = None,
    teacher_dispatch: Path | None = None,
    difflp_receipt: Path | None = None,
) -> dict[str, Any]:
    data = _build_data(materialized_root, max_eval_origins, max_train_origins)
    parameters = _scaled_parameters(
        yaml.safe_load(benchmark.read_text(encoding="utf-8")),
        _load_capacity(capacity_receipt),
    )
    budget = StageBudgetV42(
        max_epochs=epochs,
        minimum_epochs=epochs,
        ramp_epochs=max(1, epochs),
    )
    trainable = {"Scheme2R-PTO", "Official iTransformer-PTO", "Direct-Policy", "Differentiable-LP"}
    references = {"Seasonal-Naive-PTO", "Perfect-Information-MPC"}
    if method_id not in trainable | references:
        raise ValueError("unsupported bounded diagnostic method")
    seed_label = f"seed_{seed}" if method_id in trainable else "deterministic"
    root = output_root.resolve() / run_id / method_id.replace(" ", "_") / seed_label
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    freeze = {
        "contract_sha256": "0" * 64,
        "source_manifest_sha256": "0" * 64,
        "selected_candidate_value": 1.0,
    }
    artifact = None
    diff_layer = None
    if method_id == "Scheme2R-PTO":
        artifact = train_scheme2r_pto(seed, data, freeze, root, budget=budget)
        method = _registered_from_artifact(artifact, data, parameters)
        row_seed: int | None = seed
    elif method_id == "Official iTransformer-PTO":
        if itransformer_source is None or itransformer_receipt is None:
            raise ValueError("Official iTransformer diagnostic requires source and receipt")
        source_receipt = json.loads(itransformer_receipt.read_text(encoding="utf-8"))
        artifact = train_official_itransformer_pto(
            seed,
            data,
            freeze,
            source_receipt,
            root,
            source_root=itransformer_source,
            receipt_path=itransformer_receipt,
            budget=budget,
        )
        method = _registered_from_artifact(artifact, data, parameters)
        row_seed = seed
    elif method_id == "Direct-Policy":
        if teacher_dispatch is None:
            raise ValueError("Direct-Policy diagnostic requires --teacher-dispatch")
        with np.load(teacher_dispatch, allow_pickle=False) as payload:
            teacher = np.asarray(payload["dispatch"], dtype=np.float64)
        if teacher.shape != (len(data.train), 4, 21):
            raise ValueError("teacher dispatch must match the diagnostic training prefix")
        artifact = train_direct_policy(seed, data, freeze, parameters, root, budget=budget, teacher_dispatch=teacher)
        method = _registered_from_artifact(artifact, data, parameters)
        row_seed = seed
    elif method_id == "Differentiable-LP":
        if difflp_receipt is None:
            raise ValueError("Differentiable-LP diagnostic requires --difflp-receipt")
        receipt = json.loads(difflp_receipt.read_text(encoding="utf-8"))
        diff_layer = DifferentiableIESLayer(parameters)
        artifact = train_differentiable_lp(
            seed, data, freeze, receipt, parameters, root,
            budget=budget, layer=diff_layer, micro_batch_size=8,
        )
        method = _registered_from_artifact(artifact, data, parameters, diff_layer=diff_layer)
        row_seed = seed
    elif method_id == "Seasonal-Naive-PTO":
        adapter = build_formal_v4_method_adapter(
            method_id,
            parameters,
            task_mean=data.normalization.field_mean["load"],
            task_scale=data.normalization.field_scale["load"],
            normalization=data.normalization,
        )
        method = build_v42_method(method_id, parameters=parameters, adapter=adapter)
        row_seed = None
    else:
        method = _perfect_information_method(parameters)
        row_seed = None
    lineage = {
        "contract_sha256": "0" * 64,
        "source_manifest_sha256": "0" * 64,
        "train_manifest_sha256": data.train_manifest_sha256,
        "calibration_manifest_sha256": data.calibration_manifest_sha256,
        "evaluation_manifest_sha256": data.evaluation_manifest_sha256,
        "normalization_sha256": data.normalization_sha256,
    }
    row = execute_gate2_row(
        Gate2RowKey(method_id, row_seed),
        method,
        data,
        parameters,
        root,
        lineage,
    )
    receipt = {
        "schema": "rsc-pf-external-baseline-diagnostic-v1",
        "run_id": run_id,
        "method_id": method_id,
        "seed": row_seed,
        "epochs": int(epochs),
        "train_years": [2015, 2016, 2017, 2018],
        "selection_year": 2019,
        "evaluation_origins": int(max_eval_origins),
        "diagnostic": True,
        "paper_result": False,
        "evaluation_year_accessed": False,
        "test_set_accessed": False,
        "training_receipt": None if artifact is None else str(root / "TRAINING_RECEIPT.json"),
        "row_receipt": str(root / "ROW_RECEIPT.json"),
        "row": row,
    }
    (root / "DIAGNOSTIC_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--capacity-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", choices=("Scheme2R-PTO", "Official iTransformer-PTO", "Direct-Policy", "Differentiable-LP", "Seasonal-Naive-PTO", "Perfect-Information-MPC"), default="Scheme2R-PTO")
    parser.add_argument("--itransformer-source", type=Path)
    parser.add_argument("--itransformer-receipt", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-eval-origins", type=int, default=128)
    parser.add_argument("--max-train-origins", type=int)
    parser.add_argument("--teacher-dispatch", type=Path)
    parser.add_argument("--difflp-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.epochs > 30:
        raise ValueError("epochs must be in [1,30]")
    receipt = run_diagnostic(
        materialized_root=args.materialized_root,
        benchmark=args.benchmark,
        capacity_receipt=args.capacity_receipt,
        output_root=args.output_root,
        run_id=args.run_id,
        seed=args.seed,
        method_id=args.method,
        epochs=args.epochs,
        max_eval_origins=args.max_eval_origins,
        max_train_origins=args.max_train_origins,
        itransformer_source=args.itransformer_source,
        itransformer_receipt=args.itransformer_receipt,
        teacher_dispatch=args.teacher_dispatch,
        difflp_receipt=args.difflp_receipt,
    )
    print(json.dumps({
        "run_id": receipt["run_id"],
        "method_id": receipt["method_id"],
        "evaluation_origins": receipt["evaluation_origins"],
        "diagnostic": receipt["diagnostic"],
        "paper_result": receipt["paper_result"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
