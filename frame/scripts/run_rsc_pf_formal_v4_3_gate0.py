"""Run the train/selection-only formal-v4.3 preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_3_artifacts import sha256_file, write_once_json  # noqa: E402
from src.joint_dispatch.formal_v4_3_contract import assert_gate_transition, load_formal_v4_3_contract  # noqa: E402
from src.joint_dispatch.formal_v4_3_pilot import PILOT_METHODS  # noqa: E402
from src.joint_dispatch.formal_v4_models import RegimeAwareRSCPFModel  # noqa: E402


CHECKS = (
    "contract", "split_firewall", "source_regime", "model_shape",
    "joint_gradient", "decoupled_gradient", "physical_residual", "pilot_entrypoint",
)


def authorize_v43_pilot(evidence: Mapping[str, Any]) -> dict[str, Any]:
    checks = evidence.get("checks", {})
    passed = all(checks.get(name) is True for name in CHECKS)
    return {
        "authorized_pilot": bool(passed),
        "evaluation_year_accessed": False,
        "paper_eligible": False,
        "checks": {name: bool(checks.get(name) is True) for name in CHECKS},
    }


def _batch() -> dict[str, torch.Tensor]:
    context = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.5]).reshape(1, 1, 6).expand(2, 4, 6).clone()
    return {
        "load_history": torch.randn(2, 24, 4),
        "exog_history": torch.randn(2, 24, 12),
        "device_history": torch.randn(2, 24, 17),
        "activity_history": torch.zeros(2, 24, 6),
        "scheduler_context": context,
        "previous_chp": torch.zeros(2, 1),
    }


def run_gate0(*, contract_path: Path, output_root: Path, run_id: str) -> dict[str, Any]:
    contract = load_formal_v4_3_contract(contract_path)
    assert_gate_transition(contract, "gate0", "pilot")
    root = output_root.resolve() / str(run_id)
    gate0_dir = root / "gate0"
    if gate0_dir.exists():
        raise FileExistsError(gate0_dir)
    gate0_dir.mkdir(parents=True)
    checks: dict[str, bool] = {
        "contract": True,
        "split_firewall": contract.evaluation_year == 2020 and contract.payload.get("allow_evaluation_access_before_gate2") is False,
        "source_regime": tuple(contract.payload["thermal_regime"]["classes"]) == ("off", "cooling", "heating"),
        "model_shape": False,
        "joint_gradient": False,
        "decoupled_gradient": False,
        "physical_residual": False,
        "pilot_entrypoint": tuple(PILOT_METHODS) == ("stage_p_regime", "rsc_pf_joint", "decoupled_rsc_pf", "continuous_thermal_head"),
    }
    model = RegimeAwareRSCPFModel.for_test()
    batch = _batch()
    output = model(**batch)
    checks["model_shape"] = bool(output.forecast_physical.shape == (2, 4, 4) and output.controls.shape == (2, 4, 15) and output.dispatch.shape == (2, 4, 21))
    output.dispatch[..., 0].sum().backward()
    regime_gradient = model.regime_head.output.weight.grad
    checks["joint_gradient"] = bool(regime_gradient is not None and torch.isfinite(regime_gradient).all() and torch.any(regime_gradient.abs() > 0.0))
    decoupled = RegimeAwareRSCPFModel.for_test()
    decoupled_output = decoupled(detach_forecast_for_dispatch=True, **batch)
    decoupled_output.dispatch[..., 0].sum().backward()
    decoupled_gradient = decoupled.regime_head.output.weight.grad
    checks["decoupled_gradient"] = bool(decoupled_gradient is None or torch.allclose(decoupled_gradient, torch.zeros_like(decoupled_gradient)))
    checks["physical_residual"] = bool(torch.isfinite(output.dispatch).all() and torch.isfinite(output.forecast_physical).all())
    evidence = {
        "schema": "formal-v4.3-gate0-evidence-v1",
        "run_id": str(run_id),
        "contract_sha256": contract.contract_sha256,
        "checks": checks,
        "evaluation_year_accessed": False,
        "source_manifest_sha256": None,
    }
    evidence.update(authorize_v43_pilot(evidence))
    evidence_path = write_once_json(gate0_dir / "GATE0_EVIDENCE.json", evidence)
    transition = {
        "schema": "formal-v4.3-gate0-transition-v1",
        "run_id": str(run_id),
        "contract_sha256": contract.contract_sha256,
        "gate_evidence_sha256": sha256_file(evidence_path),
        "gate": "gate0",
        "next_gate": "pilot",
        "authorized_pilot": evidence["authorized_pilot"],
        "evaluation_year_accessed": False,
    }
    write_once_json(root / "GATE0_TRANSITION.json", transition)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        evidence = run_gate0(contract_path=args.contract, output_root=args.output_root, run_id=args.run_id)
    except Exception as exc:
        print(json.dumps({"authorized_pilot": False, "run_id": args.run_id, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"authorized_pilot": evidence["authorized_pilot"], "run_id": args.run_id}, ensure_ascii=False))
    return 0 if evidence["authorized_pilot"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
