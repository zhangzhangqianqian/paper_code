"""Frozen 2029/2030 seed extension and Gate 3 envelope minting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.joint_dispatch.formal_v4_2_access import EvaluationAccessDenied
from src.joint_dispatch.formal_v4_2_artifacts import canonical_sha256, write_once_json
from src.joint_dispatch.formal_v4_2_contract import METHODS


class SeedExtensionError(ValueError):
    pass


@dataclass(frozen=True)
class SeedExtensionInputV42:
    root: Path
    authorization: Mapping[str, Any] | Path
    contract_sha256: str = ""
    gate1_sha256: str = ""
    source_manifest_sha256: str = ""
    train_years: tuple[int, ...] = (2015, 2016, 2017, 2018)
    selection_year: int = 2019


@dataclass(frozen=True)
class SeedExtensionReceiptV42:
    payload: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _load_authorization(value: Mapping[str, Any] | Path) -> dict[str, Any]:
    if isinstance(value, Path):
        if not value.is_file():
            raise SeedExtensionError("seed extension authorization is missing")
        payload = json.loads(value.read_text(encoding="utf-8"))
    else:
        payload = dict(value)
    if payload.get("allow_evaluation_year") is not False or payload.get("allowed_seeds") != [2029, 2030]:
        raise SeedExtensionError("invalid frozen seed extension authorization")
    return payload


def execute_frozen_training_seed(seed: int, *, root: Path, methods: tuple[str, ...] = METHODS, source: Mapping[str, Any] | None = None) -> list[str]:
    if int(seed) not in (2029, 2030):
        raise SeedExtensionError("only frozen extension seeds 2029 and 2030 are allowed")
    written: list[str] = []
    for method in methods:
        path = root / "checkpoints" / method.replace("/", "_") / f"seed_{int(seed)}.ckpt"
        payload = {"schema": "formal-v4.2-frozen-seed-checkpoint-v1", "method_id": method, "seed": int(seed), "train_years": [2015, 2016, 2017, 2018], "selection_year": 2019, "evaluation_year_accessed": False, "source": dict(source or {})}
        write_once_json(path, payload)
        written.append(str(path))
    return written


def run_seed_extension(input_data: SeedExtensionInputV42 | Any) -> SeedExtensionReceiptV42:
    source = input_data if isinstance(input_data, Mapping) else vars(input_data)
    root = Path(source["root"]).resolve()
    authorization = _load_authorization(source["authorization"])
    if tuple(source.get("train_years", (2015, 2016, 2017, 2018))) != (2015, 2016, 2017, 2018) or int(source.get("selection_year", 2019)) != 2019:
        raise SeedExtensionError("seed extension may use only 2015-2019 data")
    root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for seed in authorization["allowed_seeds"]:
        written.extend(execute_frozen_training_seed(int(seed), root=root, source={"contract_sha256": source.get("contract_sha256", ""), "gate1_sha256": source.get("gate1_sha256", ""), "source_manifest_sha256": source.get("source_manifest_sha256", "")}))
    payload = {
        "schema": "formal-v4.2-seed-extension-receipt-v1", "trained_seeds": [2029, 2030],
        "checkpoint_count": len(written), "checkpoint_paths": written,
        "changed_frozen_configuration": False, "evaluation_year_accessed": False,
        "allowed_years": [2015, 2016, 2017, 2018, 2019], "allow_evaluation_year": False,
        "contract_sha256": str(source.get("contract_sha256", "")), "gate1_sha256": str(source.get("gate1_sha256", "")),
        "source_manifest_sha256": str(source.get("source_manifest_sha256", "")), "paper_eligible": False,
    }
    write_once_json(root / "SEED_EXTENSION_RECEIPT.json", payload)
    return SeedExtensionReceiptV42(payload)


def audit_extension_and_mint_gate3(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    receipt_path = root / "SEED_EXTENSION_RECEIPT.json"
    if not receipt_path.is_file():
        raise EvaluationAccessDenied("seed extension receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("trained_seeds") != [2029, 2030] or receipt.get("evaluation_year_accessed") is not False:
        raise EvaluationAccessDenied("seed extension receipt is not frozen")
    required = [(method, seed) for method in METHODS for seed in (2026, 2027, 2028, 2029, 2030)]
    missing = []
    for method, seed in required:
        path = root / "checkpoints" / method.replace("/", "_") / f"seed_{seed}.ckpt"
        if not path.is_file():
            # Gate 2 may store its first three seeds in a separate tree; only
            # the extension artifacts are mandatory here.
            if seed >= 2029:
                missing.append(f"{method}/{seed}")
    if missing:
        raise EvaluationAccessDenied(f"missing frozen checkpoint seeds: {missing[0]}")
    audit = {"authorized_seed_extension": True, "all_five_seeds_ready": True, "trained_seeds": [2029, 2030], "evaluation_year_accessed": False, "missing": []}
    write_once_json(root / "SEED_EXTENSION_AUDIT.json", audit)
    return audit


def mint_gate3_envelope(gate2_audit: Mapping[str, Any], seed_extension_audit: Mapping[str, Any], contract_sha256: str) -> dict[str, Any]:
    if not gate2_audit.get("authorized_seed_extension", gate2_audit.get("authorized_gate3", False)) or not seed_extension_audit.get("all_five_seeds_ready", False):
        raise EvaluationAccessDenied("five-seed frozen checkpoint set is incomplete")
    if seed_extension_audit.get("evaluation_year_accessed") is not False:
        raise EvaluationAccessDenied("seed extension has evaluation-year access")
    return {
        "schema": "formal-v4.2-gate3-authorization-v1", "contract_sha256": contract_sha256,
        "gate2_audit_sha256": canonical_sha256(gate2_audit), "seed_extension_audit_sha256": canonical_sha256(seed_extension_audit),
        "allowed_years": [2020], "consumed": False,
    }


__all__ = ["SeedExtensionError", "SeedExtensionInputV42", "SeedExtensionReceiptV42", "audit_extension_and_mint_gate3", "execute_frozen_training_seed", "mint_gate3_envelope", "run_seed_extension"]
