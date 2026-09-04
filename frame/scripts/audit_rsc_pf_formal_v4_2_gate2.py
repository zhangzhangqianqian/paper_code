"""Independent artifact-recomputation audit for formal-v4.2 Gate 2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
for entry in (str(FRAME_ROOT), str(SCRIPT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_rsc_pf_formal_v4_2_gate2 import authorize_gate2  # type: ignore  # noqa: E402
from src.joint_dispatch.formal_v4_2_artifacts import sha256_file, write_once_json  # noqa: E402
from src.joint_dispatch.formal_v4_2_contract import load_formal_v4_2_contract  # noqa: E402
from src.joint_dispatch.formal_v4_2_gate2_artifacts import validate_gate2_row  # noqa: E402
from src.joint_dispatch.formal_v4_2_metrics import compute_v42_metrics_from_arrays  # noqa: E402
from src.joint_dispatch.formal_v4_2_methods import registered_method_rows  # noqa: E402


@dataclass(frozen=True)
class Gate2AuditV42:
    authorized_gate3: bool
    failed_checks: tuple[str, ...]
    recomputed_row_count: int
    seed_extension_authorized: bool
    decision_agreement: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "formal-v4.2-gate2-independent-audit-v1",
            "authorized_gate3": self.authorized_gate3,
            "failed_checks": list(self.failed_checks),
            "recomputed_row_count": self.recomputed_row_count,
            "seed_extension_authorized": self.seed_extension_authorized,
            "decision_agreement": self.decision_agreement,
        }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _scan_access(root: Path) -> list[str]:
    failures: list[str] = []
    for path in root.rglob("*.json"):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if payload.get("evaluation_year_accessed") is True or payload.get("test_set_accessed") is True:
            failures.append("evaluation_access")
        years = payload.get("accessed_years", payload.get("years_used", []))
        if isinstance(years, (list, tuple, set)) and any(int(year) in {2020, 2021} for year in years):
            failures.append("evaluation_access")
    return sorted(set(failures))


def _row_directory(root: Path, method_id: str, seed: int | None) -> Path:
    return root / "gate2" / "rows" / method_id / ("deterministic" if seed is None else str(seed))


def _close(left: Any, right: Any) -> bool:
    try:
        a = np.asarray(left, dtype=np.float64); b = np.asarray(right, dtype=np.float64)
    except (TypeError, ValueError):
        return left == right
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1.0e-10, atol=1.0e-10, equal_nan=True))


def audit_gate2(run_root: str | Path) -> Gate2AuditV42:
    root = Path(run_root).resolve()
    failed: list[str] = []
    contract = load_formal_v4_2_contract(root / "protocol" / "formal_v4_2_contract.json")
    rows_payload = _load_json(root / "gate2" / "rows.json")
    recomputed: list[Mapping[str, Any]] = []
    stage_parents: dict[int, dict[str, str]] = {}
    for expected in registered_method_rows(contract, gate="gate2"):
        label = expected.method_id if expected.seed is None else f"{expected.method_id}__seed_{expected.seed}"
        row = rows_payload.get(label)
        if not isinstance(row, Mapping):
            failed.append(f"{label}:missing"); continue
        directory = _row_directory(root, expected.method_id, expected.seed)
        try:
            validated = validate_gate2_row(directory, {
                "method_id": expected.method_id, "seed": expected.seed,
                "contract_sha256": contract.contract_sha256,
            })
            with np.load(directory / "ROLLOUT.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files if name not in {"schema", "lineage_json"}}
            metrics = compute_v42_metrics_from_arrays(arrays, method_id=expected.method_id)
            saved = _load_json(directory / "METRICS.json")
            comparisons = {
                "penalized_objective": metrics.penalized_objective,
                "operating_cost": metrics.operating_cost,
                "physical_carbon": metrics.physical_carbon,
                "shortage_energy": metrics.shortage_energy.tolist(),
                "shortage_rate": metrics.shortage_rate.tolist(),
                "balance_residual_max": metrics.balance_residual_max,
                "capacity_violation_max": metrics.capacity_violation_max,
            }
            for name, value in comparisons.items():
                if not _close(saved.get(name), value):
                    failed.append(f"{label}:metrics_recompute:{name}")
            online = expected.method_id in {
                "Scheme2R-PTO", "State-Conditioned-PTO", "Official iTransformer-PTO",
                "Differentiable-LP", "Perfect-Information-MPC", "Seasonal-Naive-PTO",
            }
            if int(validated.get("optimizer_calls", -1)) != (int(validated["settled_hours"]) if online else 0):
                failed.append(f"{label}:optimizer_calls")
            if expected.seed is not None:
                training = _load_json(directory / "TRAINING_RECEIPT.json")
                if int(training.get("optimizer_steps", 0)) <= 0:
                    failed.append(f"{label}:optimizer_steps")
                if expected.method_id in {"RSC-PF", "Decoupled-RSC-PF"}:
                    training_parent = str(training.get("stage_s_parent_sha256", ""))
                    if validated.get("stage_s_parent_sha256") != training_parent:
                        failed.append(f"{label}:stage_s_parent_receipt")
                    stage_parents.setdefault(expected.seed, {})[expected.method_id] = training_parent
                    gradient = float(training.get("decision_forecaster_gradient_norm", 0.0))
                    if expected.method_id == "RSC-PF" and gradient <= 0.0:
                        failed.append(f"{label}:decision_gradient")
                    if expected.method_id == "Decoupled-RSC-PF" and gradient != 0.0:
                        failed.append(f"{label}:decision_gradient_boundary")
                if expected.method_id == "Direct-Policy" and training.get("forecast_loss_applicable") is not False:
                    failed.append(f"{label}:forecast_identity")
                if expected.method_id == "Official iTransformer-PTO" and training.get("upstream_commit") != "c2426e68ca13f74aaec08045c5c724d8ad328124":
                    failed.append(f"{label}:upstream_identity")
                if expected.method_id == "Differentiable-LP":
                    if int(training.get("sample_exposures", -1)) != int(training.get("expected_sample_exposures", -2)):
                        failed.append(f"{label}:sample_exposures")
                    if int(training.get("failed_solves", -1)) != 0 or float(training.get("gradient_norm", 0.0)) <= 0.0:
                        failed.append(f"{label}:diffopt_training")
            recomputed.append(validated)
        except Exception as exc:
            failed.append(f"{label}:artifact:{type(exc).__name__}:{exc}")
    for seed, parents in stage_parents.items():
        if set(parents) != {"RSC-PF", "Decoupled-RSC-PF"} or len(set(parents.values())) != 1:
            failed.append(f"seed_{seed}:stage_s_parent")
    failed.extend(_scan_access(root))
    recomputed_decision = authorize_gate2(recomputed, contract)
    failed.extend(recomputed_decision.missing_rows); failed.extend(recomputed_decision.failed_checks)
    runner = _load_json(root / "gate2" / "GATE2_DECISION.json")
    agreement = (
        bool(runner.get("authorized_gate3")) == recomputed_decision.authorized_gate3
        and int(runner.get("row_count", -1)) == recomputed_decision.row_count
        and sorted(runner.get("failed_checks", [])) == sorted(recomputed_decision.failed_checks)
    )
    if not agreement:
        failed.append("gate2_decision_disagreement")
    failed = sorted(set(failed))
    authorized = not failed and recomputed_decision.authorized_gate3 and agreement
    audit = Gate2AuditV42(authorized, tuple(failed), len(recomputed), authorized, agreement)
    write_once_json(root / "gate2" / "GATE2_AUDIT.json", audit.to_dict())
    if authorized:
        write_once_json(root / "protocol" / "GATE2_TRANSITION.json", {
            "schema": "formal-v4.2-gate2-transition-v1", "gate": "gate2",
            "next_gate": "gate2_audit_complete", "authorized_seed_extension": True,
            "allow_evaluation_year": False,
            "gate2_decision_sha256": sha256_file(root / "gate2" / "GATE2_DECISION.json"),
            "gate2_audit_sha256": sha256_file(root / "gate2" / "GATE2_AUDIT.json"),
            "contract_sha256": contract.contract_sha256,
        })
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = audit_gate2(args.run_root)
    print(json.dumps(audit.to_dict(), ensure_ascii=False))
    return 0 if audit.authorized_gate3 else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Gate2AuditV42", "audit_gate2", "build_parser", "main"]
