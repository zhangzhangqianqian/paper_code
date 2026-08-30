"""Build or audit the causal data artifacts for joint forecast--dispatch training.

The formal generator is intentionally fail-closed.  ``--dry-run`` is safe to
run before the large Kitakyushu archives are opened; actual LP generation is
enabled only when an hourly renewable/context artifact is supplied by the
experiment manifest.
"""

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
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=None)
    parser.add_argument("--benchmark-solves", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def _check_paths(contract) -> dict[str, object]:
    checks = {}
    for name, path in contract.paths.items():
        checks[name] = {"path": str(path), "exists": bool(Path(path).exists())}
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = load_joint_training_contract(args.contract)
    output_dir = Path(args.output_dir) if args.output_dir is not None else contract.path("output_root")
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = _check_paths(contract)
    missing = [name for name, value in checks.items() if not value["exists"]]
    receipt = {
        "schema_version": contract.schema_version,
        "split": args.split,
        "formal": False,
        "complete": False,
        "dry_run": bool(args.dry_run),
        "smoke_limit": args.smoke_limit,
        "benchmark_requested": args.benchmark_solves,
        "resume": bool(args.resume),
        "python_version": platform.python_version(),
        "path_checks": checks,
        "missing_paths": missing,
    }
    if args.dry_run:
        (output_dir / f"dry_run_{args.split}.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0 if not missing else 2
    if args.benchmark_solves is not None:
        required = int(contract.resource_gate["benchmark_solves"])
        if args.benchmark_solves != required:
            raise SystemExit(
                "resource benchmark is fail-closed: pass exactly "
                f"{required} solves"
            )
        import numpy as np
        import yaml
        from src.kitakyushu_pipeline import clean_kitakyushu_dataframe, read_kitakyushu_canonical
        from src.scheduling.dispatch_lp import DispatchInputs
        from src.joint_dispatch.data import benchmark_lp_generation

        years = (2015, 2016, 2017, 2018, 2019, 2020)
        raw, _ = read_kitakyushu_canonical(contract.path("kitakyushu_data_dir"), years=years)
        frame, _ = clean_kitakyushu_dataframe(raw)
        benchmark_payload = yaml.safe_load(contract.path("benchmark_path").read_text(encoding="utf-8"))
        parameters = dict(benchmark_payload["values"])
        if len(frame) < 28:
            raise SystemExit("not enough canonical hours for the 500-solve benchmark")
        valid_starts = np.arange(0, len(frame) - 4, dtype=int)
        selected = np.linspace(0, len(valid_starts) - 1, required, dtype=int)
        cases = []
        for start_index in valid_starts[selected]:
            demand = frame.loc[int(start_index) : int(start_index) + 3, ["electricity", "cooling", "heating"]].to_numpy(dtype=float)
            cases.append(DispatchInputs(demand=demand, pv_available=np.zeros(4), wt_available=np.zeros(4), parameters=parameters, initial_soc=0.5))
        receipt_obj = benchmark_lp_generation(
            cases,
            projected_total_solves=max(100_000, len(frame)),
            required_solves=required,
            max_projected_hours=float(contract.resource_gate["max_projected_p95_hours"]),
        )
        receipt.update({"formal": False, "complete": True, "resource_gate": receipt_obj.__dict__, "sample_count": required})
        (output_dir / "resource_benchmark_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    raise SystemExit(
        "Formal generation is gated: provide the prepared hourly renewable/context "
        "artifact and run the reviewed generator entry point. No LP solve was run."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
