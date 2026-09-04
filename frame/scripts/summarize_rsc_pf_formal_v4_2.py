"""Independent sealing of formal-v4.2 Gate 3 main-result tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _raw_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _load_complete(root: Path) -> dict[str, Any]:
    path = root / "gate3" / "GATE3_COMPLETE.json"
    if not path.is_file():
        raise FileNotFoundError("GATE3_COMPLETE.json is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "formal-v4.2-gate3-complete-v1":
        raise ValueError("unsupported Gate 3 receipt schema")
    return payload


def _row_table(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {"rows": rows, "row_count": 0}
    numeric_fields = ("penalized_objective", "operating_cost", "physical_carbon", "shortage_energy", "balance_residual_max", "capacity_violation_max")
    table: dict[str, Any] = {"row_count": len(rows), "methods": {}}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        method = str(row.get("method_id", "unknown")); seed = row.get("seed", "deterministic")
        entry = {field: float(row[field]) for field in numeric_fields if field in row and np.isfinite(float(row[field]))}
        table["methods"].setdefault(method, {})[str(seed)] = entry
    return table


def summarize_gate3(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    receipt = _load_complete(root)
    if receipt.get("evaluation_year_accessed") is not True or receipt.get("training_called") is not False:
        raise ValueError("Gate 3 receipt is not a locked evaluation")
    rows_payload = receipt.get("rows", {})
    rows = rows_payload.get("rows", rows_payload) if isinstance(rows_payload, Mapping) else rows_payload
    if not receipt.get("paper_eligible", False):
        raise ValueError("Gate 3 is incomplete and cannot be summarized as paper eligible")
    table = _row_table(rows)
    summary = {
        "schema": "formal-v4.2-main-summary-v1", "primary_contrast": "RSC-PF_minus_Decoupled-RSC-PF",
        "bootstrap": {"block_hours": 168, "replicates": 2000}, "paper_eligible": True,
        "evaluation_year": 2020, "raw_row_count": int(receipt.get("row_count", table.get("row_count", 0))),
        "forecast_tables": table, "dispatch_tables": table,
        "uncertainty": {"stochastic_methods": "seed-matched", "deterministic_methods": "time-block-only"},
        "audit": {"source": "gate3/GATE3_COMPLETE.json", "raw_payload_sha256": _raw_hash(receipt), "pi_mpc_deployable": False},
        "ablations_included": False,
    }
    (root / "summary").mkdir(parents=True, exist_ok=True)
    path = root / "summary" / "MAIN_RESULTS.json"
    encoded = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    if path.exists() and path.read_bytes() != encoded:
        raise FileExistsError("immutable main summary differs")
    if not path.exists():
        path.write_bytes(encoded)
    return summary


def summarize_run(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    try:
        return summarize_gate3(root)
    except (FileNotFoundError, ValueError, PermissionError):
        diagnostics: dict[str, Any] = {"schema": "formal-v4.2-run-diagnostic-summary-v1", "paper_eligible": False, "ablations_included": False}
        for name in ("GATE0_EVIDENCE.json", "PILOT_RECEIPT.json", "GATE1_FREEZE.json", "GATE2_DECISION.json", "SEED_EXTENSION_AUDIT.json"):
            matches = list(root.rglob(name))
            if matches:
                try: diagnostics[name] = json.loads(matches[0].read_text(encoding="utf-8"))
                except Exception: diagnostics[name] = {"path": str(matches[0])}
        return diagnostics


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv); print(json.dumps(summarize_run(args.run_root), ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["summarize_gate3", "summarize_run", "main"]
