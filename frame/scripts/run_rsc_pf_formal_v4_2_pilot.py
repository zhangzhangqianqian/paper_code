"""Run the bounded train-only engineering pilot for formal-v4.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_2_artifacts import write_once_json  # noqa: E402


PILOT_REQUIREMENTS = (
    "stage_p_loss_decreased", "stage_s_loss_decreased", "stage_j_loss_finite",
    "persistent_optimizer_steps", "stage_s_clone_identical", "joint_forecast_decision_gradient_positive",
    "decoupled_forecast_decision_gradient_zero", "first_step_state_carry",
)


def authorize_pilot(receipt: Mapping[str, Any], *, shortage_rate_max: float = 0.80) -> bool:
    """Return true only if every structural and numerical pilot check passes."""

    limit = float(shortage_rate_max)
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError("shortage_rate_max must be finite and non-negative")
    if any(receipt.get(name) is not True for name in PILOT_REQUIREMENTS):
        return False
    shortage = float(receipt.get("shortage_rate", np.inf))
    return bool(np.isfinite(shortage) and 0.0 <= shortage <= limit)


def run_pilot(fixture: Any) -> dict[str, Any]:
    """Persist a non-paper pilot receipt from measured train-only evidence."""

    source = fixture if isinstance(fixture, Mapping) else vars(fixture)
    output_root = Path(source.get("output_root", Path.cwd())).resolve()
    run_id = str(source.get("run_id", "formal_v4_2_pilot"))
    root = output_root / run_id
    pilot_dir = root / "pilot"
    if pilot_dir.exists():
        raise FileExistsError(pilot_dir)
    pilot_dir.mkdir(parents=True)
    receipt = {str(key): value for key, value in source.items() if key not in {"output_root", "run_id"}}
    receipt.setdefault("stage_p_loss_decreased", False)
    receipt.setdefault("stage_s_loss_decreased", False)
    receipt.setdefault("stage_j_loss_finite", False)
    receipt.setdefault("persistent_optimizer_steps", False)
    receipt.setdefault("stage_s_clone_identical", False)
    receipt.setdefault("joint_forecast_decision_gradient_positive", False)
    receipt.setdefault("decoupled_forecast_decision_gradient_zero", False)
    receipt.setdefault("first_step_state_carry", False)
    receipt.setdefault("shortage_rate", np.inf)
    receipt.update({"schema": "formal-v4.2-pilot-receipt-v1", "paper_eligible": False, "evaluation_year_accessed": False, "ranking_generated": False, "train_years": [2015, 2016, 2017, 2018], "selection_year_used": False, "authorized_gate1": authorize_pilot(receipt, shortage_rate_max=float(source.get("shortage_rate_max", 0.80)))})
    write_once_json(pilot_dir / "PILOT_RECEIPT.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default="formal_v4_2_pilot")
    args = parser.parse_args(argv)
    receipt = run_pilot({"output_root": args.output_root, "run_id": args.run_id})
    print(json.dumps({"authorized_gate1": receipt["authorized_gate1"], "paper_eligible": receipt["paper_eligible"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["authorize_pilot", "run_pilot", "main"]
