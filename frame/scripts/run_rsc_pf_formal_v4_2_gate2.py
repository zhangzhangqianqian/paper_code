"""Fixed Gate 2 matrix enumeration and fail-closed authorization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_2_artifacts import write_once_json  # noqa: E402
from src.joint_dispatch.formal_v4_2_contract import FormalV42Contract  # noqa: E402
from src.joint_dispatch.formal_v4_2_methods import registered_method_rows  # noqa: E402


EXPECTED_CALLS = {"RSC-PF": 0, "Decoupled-RSC-PF": 0, "Direct-Policy": 0, "Scheme2R-PTO": 1, "State-Conditioned-PTO": 1, "Official iTransformer-PTO": 1, "Differentiable-LP": 1, "Perfect-Information-MPC": 1, "Seasonal-Naive-PTO": 1}


@dataclass(frozen=True)
class Gate2DecisionV42:
    authorized_gate3: bool
    missing_rows: tuple[str, ...]
    failed_checks: tuple[str, ...]
    row_count: int
    favorable_seed_directions: int = 0

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "formal-v4.2-gate2-decision-v1", "authorized_gate3": self.authorized_gate3, "missing_rows": list(self.missing_rows), "failed_checks": list(self.failed_checks), "row_count": self.row_count, "favorable_seed_directions": self.favorable_seed_directions}


def build_gate2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gate1-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def _row_label(method_id: str, seed: int | None) -> str:
    return f"{method_id}/{seed if seed is not None else 'deterministic'}"


def _find_row(rows: Any, method_id: str, seed: int | None) -> Mapping[str, Any] | None:
    if isinstance(rows, Mapping):
        for key in ((method_id, seed), _row_label(method_id, seed), method_id if seed is None else None):
            if key is not None and key in rows and isinstance(rows[key], Mapping):
                return rows[key]
        for key, value in rows.items():
            if isinstance(value, Mapping) and str(value.get("method_id", "")) == method_id and value.get("seed") == seed:
                return value
    else:
        for value in rows:
            if isinstance(value, Mapping) and str(value.get("method_id", "")) == method_id and value.get("seed") == seed:
                return value
            if getattr(value, "method_id", None) == method_id and getattr(value, "seed", None) == seed:
                return vars(value)
    return None


def _numeric(row: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, Mapping):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        return number
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping):
        for name in names:
            try:
                return float(metrics[name])
            except (KeyError, TypeError, ValueError):
                pass
    return None


def authorize_gate2(rows: Any, contract: FormalV42Contract | Mapping[str, Any]) -> Gate2DecisionV42:
    expected = registered_method_rows(contract, gate="gate2")
    missing: list[str] = []; failed: list[str] = []
    resolved: dict[tuple[str, int | None], Mapping[str, Any]] = {}
    for expected_row in expected:
        method_id, seed = expected_row.method_id, expected_row.seed
        row = _find_row(rows, method_id, seed)
        label = _row_label(method_id, seed)
        if row is None:
            missing.append(label); continue
        resolved[(method_id, seed)] = row
        if row.get("method_id", method_id) != method_id or row.get("seed", seed) != seed:
            failed.append(f"{label}:identity")
        status = row.get("status", "complete")
        if status != "complete": failed.append(f"{label}:status")
        if row.get("test_set_accessed", row.get("evaluation_year_accessed", False)) is True:
            failed.append(f"{label}:evaluation_access")
        calls = _numeric(row, "optimizer_calls", "online_optimizer_calls")
        if calls is not None and int(calls) != EXPECTED_CALLS[method_id]:
            failed.append(f"{label}:optimizer_calls")
        for field in ("penalized_objective", "shortage_energy", "balance_residual_max", "capacity_violation_max"):
            value = _numeric(row, field)
            if value is None or not np.isfinite(value):
                failed.append(f"{label}:{field}")
        physical = _numeric(row, "physical_residual_max", "constraint_violation_max")
        if physical is not None and (not np.isfinite(physical) or physical > 1.0e-5):
            failed.append(f"{label}:physical_residual")
    favorable = 0
    for seed in (2026, 2027, 2028):
        rsc = resolved.get(("RSC-PF", seed)); dec = resolved.get(("Decoupled-RSC-PF", seed))
        if rsc is not None and dec is not None:
            rsc_obj = _numeric(rsc, "penalized_objective"); dec_obj = _numeric(dec, "penalized_objective")
            if rsc_obj is not None and dec_obj is not None and rsc_obj < dec_obj:
                favorable += 1
    if favorable < 2:
        failed.append("primary_direction_reliability")
    for seed in (2026, 2027, 2028):
        rsc = resolved.get(("RSC-PF", seed)); state = resolved.get(("State-Conditioned-PTO", seed))
        if rsc is None or state is None: continue
        a = _numeric(rsc, "shortage_energy"); b = _numeric(state, "shortage_energy")
        if a is not None and b is not None and a > 1.05 * max(b, 1.0e-12):
            failed.append(f"RSC-PF/{seed}:shortage_ratio")
    authorized = not missing and not failed
    return Gate2DecisionV42(authorized, tuple(sorted(set(missing))), tuple(sorted(set(failed))), len(resolved), favorable)


def write_gate2_decision(path: str | Path, decision: Gate2DecisionV42, *, contract_sha256: str, gate1_sha256: str = "") -> Path:
    payload = decision.to_dict(); payload.update({"contract_sha256": contract_sha256, "gate1_sha256": gate1_sha256, "paper_eligible": False})
    write_once_json(path, payload)
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    args = build_gate2_parser().parse_args(argv)
    payload = json.loads(Path(args.run_root).joinpath("gate2_rows.json").read_text(encoding="utf-8"))
    decision = authorize_gate2(payload, {"gate2_seeds": [2026, 2027, 2028]})
    write_gate2_decision(args.output, decision, contract_sha256="" * 64)
    print(json.dumps(decision.to_dict(), ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Gate2DecisionV42", "authorize_gate2", "build_gate2_parser", "main", "write_gate2_decision"]
