"""Fixed Gate 2 matrix enumeration and fail-closed authorization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

import yaml

from src.joint_dispatch.formal_v4_2_artifacts import sha256_file, write_failure_receipt, write_once_json  # noqa: E402
from src.joint_dispatch.formal_v4_2_contract import FormalV42Contract, load_formal_v4_2_contract  # noqa: E402
from src.joint_dispatch.formal_v4_2_gate2_execution import execute_gate2_matrix  # noqa: E402
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
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json")
    parser.add_argument("--output-root", type=Path, default=FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4_2")
    parser.add_argument("--run-id", required=True)
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
        settled_hours = _numeric(row, "settled_hours")
        expected_calls = 0 if EXPECTED_CALLS[method_id] == 0 else int(settled_hours if settled_hours is not None else 1)
        if calls is not None and int(calls) != expected_calls:
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


def _parameters(root: Path) -> dict[str, Any]:
    benchmark_path = root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
    capacity_path = root / "gate0" / "CAPACITY_FREEZE.json"
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    values = dict(benchmark["values"])
    multiplier = float(capacity["selected"]["multiplier"])
    for name in ("electric_chiller_capacity", "absorption_chiller_capacity"):
        values[name] = float(values[name]) * multiplier
    values.setdefault("surplus_penalty", 0.1)
    values["carbon_price"] = values.get("carbon_price_default", 0.0)
    return values


def run_gate2(contract_path: str | Path, output_root: str | Path, run_id: str) -> Gate2DecisionV42:
    contract = load_formal_v4_2_contract(contract_path)
    root = Path(output_root).resolve() / str(run_id)
    protocol = root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    contract_copy = protocol / "formal_v4_2_contract.json"
    if contract_copy.exists():
        if contract_copy.read_bytes() != Path(contract_path).resolve().read_bytes():
            raise FileExistsError("run-root contract copy differs from the requested contract")
    else:
        shutil.copyfile(Path(contract_path).resolve(), contract_copy)
    try:
        rows = execute_gate2_matrix(contract=contract, run_root=root, parameters=_parameters(root))
        decision = authorize_gate2(rows, contract)
        write_gate2_decision(
            root / "gate2" / "GATE2_DECISION.json", decision,
            contract_sha256=contract.contract_sha256,
            gate1_sha256=sha256_file(root / "gate1" / "GATE1_FREEZE.json"),
        )
        return decision
    except Exception as exc:
        write_failure_receipt(
            root / "gate2" / "GATE2_FAILURE.json", stage="gate2", exception=exc,
            lineage={"contract_sha256": contract.contract_sha256},
        )
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_gate2_parser().parse_args(argv)
    decision = run_gate2(args.contract, args.output_root, args.run_id)
    print(json.dumps(decision.to_dict(), ensure_ascii=False))
    return 0 if decision.authorized_gate3 else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Gate2DecisionV42", "authorize_gate2", "build_gate2_parser", "main", "run_gate2", "write_gate2_decision"]
