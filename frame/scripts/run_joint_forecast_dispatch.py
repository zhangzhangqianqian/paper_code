"""Fail-closed runner for the joint forecast--dispatch experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Sequence

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.contract import load_joint_training_contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("configs/joint_forecast_dispatch_contract_v1.json"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=("dry-run", "smoke", "validation", "select", "test"), required=True)
    parser.add_argument("--variant", choices=("joint_from_scratch", "warm_started_joint", "frozen_pto"), default="joint_from_scratch")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def _artifact_exists(root: Path, names: Sequence[str]) -> list[str]:
    return [name for name in names if not (root / name).exists()]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = load_joint_training_contract(args.contract)
    data_root = Path(args.data_root) if args.data_root is not None else contract.path("output_root") / "data"
    output_dir = Path(args.output_dir) if args.output_dir is not None else contract.path("output_root") / args.stage
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": contract.schema_version,
        "stage": args.stage,
        "variant": args.variant,
        "seed": int(args.seed),
        "formal": False,
        "complete": False,
        "online_exact_lp_calls": 0,
        "python_version": platform.python_version(),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
    }
    if args.stage == "dry-run":
        receipt["path_checks"] = {
            "contract": str(args.contract.resolve()),
            "data_root_exists": data_root.exists(),
            "output_root_exists": output_dir.exists(),
        }
        (output_dir / "dry_run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "smoke":
        if args.smoke_limit is None or args.smoke_limit <= 0:
            raise SystemExit("smoke stage requires a positive --smoke-limit")
        receipt["smoke_limit"] = int(args.smoke_limit)
        missing = _artifact_exists(data_root, ("train.npz", "validation.npz"))
        if missing:
            raise SystemExit(f"smoke data artifacts are missing: {missing}; run the gated data builder first")
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise SystemExit("smoke stage requires optional PyTorch; no experiment was started") from exc
        raise SystemExit("smoke execution is intentionally gated until the resource receipt is present")
    if args.stage == "validation":
        if (data_root / "test.npz").exists():
            raise SystemExit("validation stage must not read the sealed test split")
        missing = _artifact_exists(data_root, ("train.npz", "validation.npz"))
        if missing:
            raise SystemExit(f"validation data artifacts are missing: {missing}")
        raise SystemExit("validation execution is gated until smoke and resource receipts pass")
    selection = output_dir.parent / "selection" / "selection_receipt.json"
    if args.stage == "select":
        missing = _artifact_exists(output_dir.parent / "validation", ("validation_receipt.json",))
        if missing:
            raise SystemExit(f"selection requires complete validation receipt: {missing}")
        raise SystemExit("selection execution is gated until validation candidates are complete")
    if args.stage == "test":
        if not selection.exists():
            raise SystemExit(f"test stage requires frozen selection receipt: {selection}")
        raise SystemExit("test execution is sealed behind the immutable selection receipt")
    raise AssertionError("unreachable stage")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
