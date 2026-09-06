"""Command-line entry point for one complete-formal stochastic training row.

The command requires a serialized train-only data bundle supplied by the caller.
It never discovers data implicitly and therefore cannot silently switch to a
sealed evaluation year.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract, MethodSeedKey
from src.joint_dispatch.complete_formal_training import train_row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="configs/rsc_pf_complete_formal_v1.json")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data", required=True, help="train-only torch bundle")
    parser.add_argument("--output", default="reports/rsc_pf_complete_formal/gate1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    contract = CompleteFormalContract.from_path(args.contract)
    bundle = torch.load(Path(args.data), map_location="cpu", weights_only=False)
    artifact = train_row(
        contract,
        MethodSeedKey(args.method, args.seed),
        bundle,
        args.output,
        resume=args.resume,
    )
    print({
        "method_id": artifact.key.method_id,
        "seed": artifact.key.seed,
        "checkpoint": str(artifact.checkpoint_path),
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "training_exposures": artifact.training_exposures,
        "complete": artifact.complete,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())

