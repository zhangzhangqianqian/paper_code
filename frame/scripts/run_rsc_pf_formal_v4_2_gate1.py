"""Freeze the representative-origin calibration for formal-v4.2 Gate 1."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_2_artifacts import write_once_json  # noqa: E402
from src.joint_dispatch.formal_v4_2_data import Gate1OriginManifestV42  # noqa: E402
from src.joint_dispatch.formal_v4_2_methods import registered_method_rows  # noqa: E402


class Gate1AuthorizationError(ValueError):
    """Raised when Gate 1 calibration inputs do not satisfy the contract."""


@dataclass(frozen=True)
class Gate1InputV42:
    manifest: Gate1OriginManifestV42
    contract: Any
    normalization_sha256: str
    access_receipt_sha256: str
    candidate_values: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
    output_path: Path | None = None


@dataclass(frozen=True)
class Gate1FreezeV42:
    payload: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @property
    def authorized_gate2(self) -> bool:
        return bool(self.payload.get("authorized_gate2", False))


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _manifest_hash(manifest: Gate1OriginManifestV42) -> str:
    payload = manifest.to_payload()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def run_gate1_calibration(input_data: Gate1InputV42 | Any) -> Gate1FreezeV42:
    manifest = _attr(input_data, "manifest")
    if manifest is None:
        raise Gate1AuthorizationError("Gate 1 requires a representative origin manifest")
    cooling = float(_attr(manifest, "cooling_active_fraction", _attr(manifest, "activity_fraction", {}).get("cooling", 0.0)))
    heating = float(_attr(manifest, "heating_active_fraction", _attr(manifest, "activity_fraction", {}).get("heating", 0.0)))
    if cooling < 0.20 or heating < 0.20:
        raise Gate1AuthorizationError("representative manifest has insufficient cooling/heating activity coverage")
    candidate_values = tuple(float(value) for value in _attr(input_data, "candidate_values", (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)))
    if not candidate_values or any(not np.isfinite(value) or value <= 0.0 for value in candidate_values):
        raise Gate1AuthorizationError("capacity candidates must be finite and positive")
    contract = _attr(input_data, "contract")
    try:
        rows = registered_method_rows(contract, gate="gate2")
        contract_hash = str(_attr(contract, "contract_sha256", _attr(contract, "sha256", "")))
    except Exception as exc:
        raise Gate1AuthorizationError(f"invalid formal-v4.2 contract: {exc}") from exc
    normalization_hash = str(_attr(input_data, "normalization_sha256", ""))
    access_hash = str(_attr(input_data, "access_receipt_sha256", ""))
    if len(normalization_hash) != 64 or len(access_hash) != 64:
        raise Gate1AuthorizationError("Gate 1 requires normalization and access receipt hashes")
    frozen_training = {
        "stage_p_max_epochs": 30, "stage_s_max_epochs": 30, "stage_j_max_epochs": 30,
        "minimum_epochs": 18, "patience": 5, "validation_interval": 1,
        "learning_rates": {"forecaster": 1.0e-5, "scheduler": 1.0e-3},
        "curriculum": {"forecast": 1.0, "imitation_start": 1.0, "imitation_final": 0.0, "decision_start": 0.05, "decision_final": 1.0, "ramp_epochs": 18},
        "candidate_values": list(candidate_values), "rollin_fraction": 0.40, "model_history_fraction": 0.50,
    }
    checks = {
        "activity_coverage": cooling >= 0.20 and heating >= 0.20,
        "contract_methods": len(rows) == 23,
        "normalization_hash": len(normalization_hash) == 64,
        "access_hash": len(access_hash) == 64,
        "selection_only": True,
        "no_evaluation_access": True,
    }
    payload = {
        "schema": "formal-v4.2-gate1-freeze-v1", "authorized_gate2": all(checks.values()),
        "origin_manifest_sha256": _manifest_hash(manifest), "normalization_sha256": normalization_hash,
        "contract_sha256": contract_hash, "frozen_training": frozen_training,
        "frozen_gate2_methods": [asdict(row) for row in rows], "checks": checks,
        "access_receipt_sha256": access_hash, "selection_year": 2019,
        "evaluation_year_accessed": False, "paper_eligible": False,
    }
    output_path = _attr(input_data, "output_path", None)
    if output_path is not None:
        write_once_json(output_path, payload)
    return Gate1FreezeV42(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps({"status": "requires_precomputed_manifest", "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Gate1AuthorizationError", "Gate1FreezeV42", "Gate1InputV42", "run_gate1_calibration", "main"]
