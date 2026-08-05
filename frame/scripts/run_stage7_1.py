"""Run Stage 7.1 A0--A4 interface validation without reading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import count_trainable_parameters  # noqa: E402
from src.stage7_ablations import (  # noqa: E402
    ABLATION_NAMES,
    build_stage7_ablation_model,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Stage 7.1 A0-A4 model interfaces with random tensors"
    )
    parser.add_argument(
        "--contract",
        default="frame/configs/stage7r_contract.json",
        help="Stage 7.0 contract; no data files are read",
    )
    parser.add_argument(
        "--output-dir",
        default="frame/reports/stage7_1_kitakyushu",
    )
    parser.add_argument("--exog-dim", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path = _resolve(args.contract)
    if not contract_path.exists():
        raise FileNotFoundError(f"Stage 7.0 contract not found: {contract_path}")
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("contract_version") != "stage7.0":
        raise ValueError("Stage 7.1 requires a stage7.0 contract")
    expected = tuple(contract.get("stage7_scope", {}).get("ablations", ()))
    if expected != ABLATION_NAMES:
        raise ValueError(f"contract ablations {expected!r} != {ABLATION_NAMES!r}")

    output_dir = _resolve(args.output_dir)
    report_path = output_dir / "stage7_1_manifest.json"
    if report_path.exists() and not args.force:
        raise FileExistsError(f"report exists; use --force to overwrite: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    lookback = int(contract["data_protocol"]["lookback"])
    horizon = int(contract["data_protocol"]["horizon"])
    task_count = len(contract["tasks"])
    loads = torch.randn(args.batch_size, lookback, task_count)
    exog = torch.randn(args.batch_size, lookback, args.exog_dim)
    runs = []
    for ablation_id in ABLATION_NAMES:
        torch.manual_seed(args.seed)
        model = build_stage7_ablation_model(
            ablation_id,
            exog_dim=args.exog_dim,
            task_count=task_count,
            lookback=lookback,
            horizon=horizon,
        )
        model.eval()
        with torch.no_grad():
            prediction, details = model.forward_with_details(loads, exog)
        if tuple(prediction.shape) != (args.batch_size, horizon, task_count):
            raise AssertionError(
                f"{ablation_id} output shape is {tuple(prediction.shape)}"
            )
        if not torch.isfinite(prediction).all():
            raise AssertionError(f"{ablation_id} produced non-finite output")
        gates = details["gates"]
        runs.append(
            {
                "ablation_id": ablation_id,
                "description": model.ablation_description,
                "parameter_count": count_trainable_parameters(model),
                "input_shape": list(loads.shape),
                "exogenous_shape": list(exog.shape),
                "output_shape": list(prediction.shape),
                "gate_shape": list(gates.shape),
                "finite_output": True,
            }
        )

    report = {
        "stage": "7.1",
        "status": "passed",
        "contract": str(contract_path),
        "dataset": contract["dataset"],
        "task_order": contract["tasks"],
        "seed": args.seed,
        "data_read": False,
        "test_set_accessed": False,
        "formal_training_started": False,
        "ablations": runs,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
