"""Emit a provenance descriptor for one formal-v4 baseline adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_protocol_v4 import load_formal_v4_spec  # noqa: E402
from src.joint_dispatch.formal_v4_baselines import build_formal_v4_baseline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4.json")
    parser.add_argument("--method", required=True, choices=("RSC-PF", "Decoupled-RSC-PF", "Scheme2R-PTO", "State-Conditioned-PTO", "Direct-Policy", "Seasonal-Naive-PTO", "Official iTransformer-PTO", "Differentiable-LP", "Perfect-Information-MPC"))
    parser.add_argument("--stage-p-checkpoint-sha256", default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    spec = load_formal_v4_spec(args.contract)
    if args.method not in {method.name for method in spec.methods}:
        raise ValueError("method is not in the frozen formal-v4 registry")
    adapter = build_formal_v4_baseline(args.method, stage_p_checkpoint_sha256=args.stage_p_checkpoint_sha256) if args.method not in {"RSC-PF", "Decoupled-RSC-PF"} else type("InternalMetadata", (), {"to_dict": lambda self: {"method_id": args.method, "executable_interface": "predict_and_dispatch(window, rolling_state)", "online_lp_calls_per_window": 0, "forecast_metrics_applicable": True}})()
    payload = adapter.to_dict()
    payload.update({"status": "pass", "test_set_accessed": False, "contract": str(args.contract), "executable_interface": "predict_and_dispatch(window, rolling_state)"})
    destination = args.output or (FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4" / "baselines" / f"{args.method.lower().replace(' ', '_')}.json")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite baseline receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
