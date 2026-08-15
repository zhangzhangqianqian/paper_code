"""Stage 10.7: freeze the transparent simulated IES benchmark parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.benchmark_parameters import derive_benchmark_parameters  # noqa: E402
from src.scheduling.contracts import load_scheduling_contract  # noqa: E402
from src.scheduling.data import build_scheduling_frame  # noqa: E402
from src.scheduling.parameter_audit import read_parameter_ledger  # noqa: E402
from src.scheduling.renewables import pv_available, wt_available  # noqa: E402


def _ledger_values(path: Path) -> dict[str, float]:
    return {record.parameter_id: record.value for record in read_parameter_ledger(path).records}


def _profile_parameters(values: dict[str, float]) -> dict[str, float]:
    return {
        "pv_rated_capacity": 1.0,
        "pv_reference_irradiance": values["pv_reference_irradiance"],
        "pv_conversion_efficiency": values["pv_conversion_efficiency"],
        "pv_reference_temperature": values["pv_reference_temperature"],
        "pv_temperature_coefficient": values["pv_temperature_coefficient"],
        "wt_rated_capacity": 1.0,
        "wt_cut_in_speed": values["wt_cut_in_speed"],
        "wt_rated_speed": values["wt_rated_speed"],
        "wt_cut_out_speed": values["wt_cut_out_speed"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default="frame/configs/scheduling_dual_track_contract_v2.yaml", type=Path)
    parser.add_argument("--ledger", default="frame/configs/scheduling_parameter_ledger_v2.csv", type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    contract = load_scheduling_contract(args.contract)
    values = _ledger_values(args.ledger)
    frame = build_scheduling_frame(args.data_dir, years=contract.train_years)
    data = frame.data.copy()
    profile_parameters = _profile_parameters(values)
    data["pv_profile"] = pv_available(data, profile_parameters)
    data["wt_profile"] = wt_available(data, profile_parameters)
    benchmark = derive_benchmark_parameters(data, values, source_years=contract.train_years)
    payload = {
        "schema_version": "standard-ies-benchmark-v1",
        "dataset": contract.dataset,
        "track": "simulated_dispatch",
        "data_origin": "simulated topology calibrated only from training-period statistics",
        "train_years": list(contract.train_years),
        "validation_year": contract.validation_year,
        "test_year": contract.test_year,
        "time_resolution": str(contract.raw["time"]["sampling"]),
        "horizon_hours": contract.horizon_hours,
        **benchmark.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    args.output.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    manifest = {
        "stage": "10.7",
        "status": "pass",
        "output": str(args.output),
        "sha256": digest,
        "train_years": list(contract.train_years),
        "validation_year": contract.validation_year,
        "test_year": contract.test_year,
        "test_year_used_for_capacity": False,
        "renewable_capacity_rule": "15% PV / 10% WT of training electric energy",
    }
    args.output.with_name(args.output.stem + "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
