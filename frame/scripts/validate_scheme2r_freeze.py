"""Validate the frozen Scheme2R-v1 contract against the live model interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import Scheme2RModel  # noqa: E402


def _shape(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def validate(contract_path: Path, batch_size: int = 2) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    dataset = contract["dataset"]
    encoder = contract["encoder"]
    routing = contract["context_and_routing"]
    messages = contract["message_and_prediction"]

    torch.manual_seed(2026)
    model = Scheme2RModel(
        exog_dim=dataset["exog_dim"],
        task_count=len(dataset["tasks"]),
        hidden_dim=encoder["hidden_dim"],
        lookback=dataset["lookback"],
        kernel_size=encoder["kernel_size"],
        dilations=tuple(encoder["dilations"]),
        dropout=encoder["dropout"],
        horizon=dataset["horizon"],
        state_dim=routing["state_dim"],
        state_hidden_dim=32,
        gate_hidden_dim=routing["gate_hidden_dim"],
        task_embedding_dim=routing["task_role_embedding_dim"],
        step_embedding_dim=routing["forecast_step_embedding_dim"],
        rank=messages["low_rank"],
        head_hidden_dim=messages["head_hidden_dim"],
    )
    model.eval()
    loads = torch.randn(
        batch_size, dataset["lookback"], len(dataset["tasks"])
    )
    exog = torch.randn(batch_size, dataset["lookback"], dataset["exog_dim"])

    with torch.no_grad():
        prediction, details = model.forward_with_details(loads, exog)

    expected_output = [batch_size, dataset["horizon"], len(dataset["tasks"])]
    expected_repr = [batch_size, len(dataset["tasks"]), encoder["hidden_dim"]]
    expected_rho = [batch_size, dataset["horizon"], len(dataset["tasks"])]
    expected_gate = [
        batch_size,
        dataset["horizon"],
        len(dataset["tasks"]),
        len(dataset["tasks"]),
    ]

    checks = {
        "prediction_shape": _shape(prediction),
        "representation_shape": _shape(details["representations"]),
        "state_shape": _shape(details["state"]),
        "rho_shape": _shape(details["rho"]),
        "pi_shape": _shape(details["pi"]),
        "gate_shape": _shape(details["gates"]),
        "pair_message_shape": _shape(details["pair_messages"]),
        "fused_shape": _shape(details["fused_representations"]),
        "expected_prediction_shape": expected_output,
        "expected_representation_shape": expected_repr,
        "expected_rho_shape": expected_rho,
        "expected_gate_shape": expected_gate,
        "finite_outputs": bool(
            torch.isfinite(prediction).all()
            and torch.isfinite(details["gates"]).all()
        ),
        "self_route_max_abs": float(
            details["gates"].diagonal(dim1=-2, dim2=-1).abs().max().item()
        ),
    }
    checks["shapes_match"] = (
        checks["prediction_shape"] == expected_output
        and checks["representation_shape"] == expected_repr
        and checks["rho_shape"] == expected_rho
        and checks["gate_shape"] == expected_gate
    )
    checks["self_route_is_zero"] = checks["self_route_max_abs"] == 0.0
    checks["passed"] = bool(
        checks["shapes_match"]
        and checks["finite_outputs"]
        and checks["self_route_is_zero"]
    )
    return {
        "freeze_id": contract["freeze_id"],
        "model": "Scheme2RModel",
        "batch_size": batch_size,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "scheme2r_v1_freeze.json",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.contract, batch_size=args.batch_size)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["checks"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
