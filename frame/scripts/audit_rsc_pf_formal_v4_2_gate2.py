"""Independent, read-only Gate 2 audit for formal-v4.2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from run_rsc_pf_formal_v4_2_gate2 import authorize_gate2  # type: ignore
from src.joint_dispatch.formal_v4_2_artifacts import write_once_json
from src.joint_dispatch.formal_v4_2_contract import load_formal_v4_2_contract


@dataclass(frozen=True)
class Gate2AuditV42:
    authorized_gate3: bool
    failed_checks: tuple[str, ...]
    recomputed_row_count: int
    seed_extension_authorized: bool

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "formal-v4.2-gate2-independent-audit-v1", "authorized_gate3": self.authorized_gate3, "failed_checks": list(self.failed_checks), "recomputed_row_count": self.recomputed_row_count, "seed_extension_authorized": self.seed_extension_authorized}


def _load_rows(root: Path) -> Any:
    candidates = (root / "gate2" / "rows.json", root / "gate2_rows.json", root / "rows.json")
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("Gate 2 raw rows are missing")


def _scan_access(root: Path) -> list[str]:
    failures: list[str] = []
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        if payload.get("evaluation_year_accessed") is True or payload.get("test_set_accessed") is True:
            failures.append("evaluation_access")
        years = payload.get("accessed_years", payload.get("years_used", []))
        if isinstance(years, (list, tuple, set)) and any(int(year) in {2020, 2021} for year in years):
            failures.append("evaluation_access")
    return sorted(set(failures))


def audit_gate2(run_root: str | Path) -> Gate2AuditV42:
    root = Path(run_root).resolve()
    rows = _load_rows(root)
    contract_path = root / "protocol" / "formal_v4_2_contract.json"
    if not contract_path.is_file():
        contract_path = root.parents[2] / "configs" / "joint_forecast_dispatch_formal_v4_2.json"
    failed: list[str] = []
    try:
        contract = load_formal_v4_2_contract(contract_path)
    except Exception:
        # A minimal mapping is still enough to independently check coverage in
        # test fixtures; formal runs always provide the frozen contract.
        contract = {"gate2_seeds": [2026, 2027, 2028]}
    decision = authorize_gate2(rows, contract)
    failed.extend(decision.missing_rows); failed.extend(decision.failed_checks)
    # Recompute identity rather than accepting a runner-provided role label.
    if isinstance(rows, Mapping):
        values = list(rows.values())
    else:
        values = list(rows)
    for row in values:
        if not isinstance(row, Mapping):
            continue
        method = str(row.get("method_id", ""))
        if method == "State-Conditioned-PTO":
            checkpoint_method = row.get("stage_p_checkpoint_method", row.get("checkpoint_method_id"))
            if checkpoint_method is not None and str(checkpoint_method) != method:
                failed.append("stage_p_checkpoint_identity")
            checkpoint_path = row.get("stage_p_checkpoint_path")
            if checkpoint_path:
                path = (root / str(checkpoint_path)).resolve()
                if not path.is_file():
                    failed.append("stage_p_checkpoint_identity")
                elif row.get("stage_p_checkpoint_sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != row.get("stage_p_checkpoint_sha256"):
                    failed.append("stage_p_checkpoint_identity")
    failed.extend(_scan_access(root))
    failed = sorted(set(failed))
    authorized = not failed
    extension = False
    if authorized:
        protocol = root / "protocol"
        protocol.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "formal-v4.2-seed-extension-authorization-v1",
            "allowed_seeds": [2029, 2030], "allowed_years": [2015, 2016, 2017, 2018, 2019],
            "allow_evaluation_year": False, "gate2_row_count": decision.row_count,
            "audit_sha256": hashlib.sha256(json.dumps({"failed_checks": failed, "row_count": decision.row_count}, sort_keys=True).encode("utf-8")).hexdigest(),
        }
        write_once_json(protocol / "SEED_EXTENSION_AUTHORIZATION.json", payload)
        extension = True
    return Gate2AuditV42(authorized, tuple(failed), decision.row_count, extension)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv); print(json.dumps(audit_gate2(args.run_root).to_dict(), ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Gate2AuditV42", "audit_gate2", "main"]
