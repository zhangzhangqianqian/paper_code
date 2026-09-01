from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_rsc_pf_common_statistics import (
    BASELINES,
    EXPECTED_WINDOWS,
    RSC_METHOD,
    SEEDS,
    analyze_common_statistics,
)


def _write_fixture(root: Path, *, omit: tuple[str, int] | None = None) -> None:
    safe = {
        RSC_METHOD: "RSC_PF",
        "iTransformer-PTO": "iTransformer_PTO",
        "DecisionFocused-Online": "DecisionFocused_Online",
        "DigitalTwins-Policy": "DigitalTwins_Policy",
    }
    for method in (RSC_METHOD, *BASELINES):
        for seed_index, model_seed in enumerate(SEEDS):
            if omit == (method, model_seed):
                continue
            directory = root / "validation" / safe[method] / f"seed_{model_seed}" / "evaluation"
            directory.mkdir(parents=True, exist_ok=True)
            baseline = 10.0 + seed_index / 100.0
            rsc = baseline - 1.0 if method == RSC_METHOD else baseline
            feasible = 0.8 + seed_index / 1000.0 if method == RSC_METHOD else 0.7 + seed_index / 1000.0
            arrays = {
                "operating_cost": np.full(EXPECTED_WINDOWS, rsc),
                "physical_carbon": np.full(EXPECTED_WINDOWS, rsc),
                "penalized_objective": np.full(EXPECTED_WINDOWS, rsc),
                "regret_vs_oracle": np.full(EXPECTED_WINDOWS, rsc),
                "shortage": np.full(EXPECTED_WINDOWS, rsc),
                "feasible": np.full(EXPECTED_WINDOWS, feasible),
            }
            np.savez_compressed(directory / "validation_metrics.npz", **arrays)
            (directory / "evaluation_receipt.json").write_text(
                json.dumps({
                    "method_id": method,
                    "seed": model_seed,
                    "stage": "validation",
                    "test_set_accessed": False,
                    "metrics": {"windows": EXPECTED_WINDOWS},
                }),
                encoding="utf-8",
            )


def test_statistics_rejects_test_like_report_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed test-set"):
        analyze_common_statistics(tmp_path / "test")


def test_statistics_requires_all_five_seed_arrays(tmp_path: Path) -> None:
    _write_fixture(tmp_path, omit=("DigitalTwins-Policy", 2030))
    with pytest.raises(FileNotFoundError):
        analyze_common_statistics(tmp_path, output_root=tmp_path / "out", replicates=10)


def test_statistics_writes_paired_ci_rows_and_conservative_metadata(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    report = analyze_common_statistics(
        tmp_path,
        output_root=tmp_path / "out",
        block_hours=168,
        replicates=25,
        seed=17,
    )
    assert len(report["rows"]) == 3 * 6
    assert report["test_set_accessed"] is False
    assert report["independent_unit"].endswith("(n=5)")
    assert Path(report["csv"]).is_file()
    assert Path(report["json"]).is_file()
    regret = next(row for row in report["rows"] if row["metric"] == "regret_vs_oracle" and row["method_b"] == BASELINES[0])
    assert regret["observed_difference_a_minus_b"] == -1.0
    assert regret["ci_lower"] == -1.0
    assert regret["ci_upper"] == -1.0
    assert "RSC-PF better" in regret["interpretation"]
    feasible = next(row for row in report["rows"] if row["metric"] == "feasible" and row["method_b"] == BASELINES[0])
    assert feasible["direction"] == "higher_is_better"
    assert "RSC-PF better" in feasible["interpretation"]
    saved = json.loads(Path(report["json"]).read_text(encoding="utf-8"))
    assert len(saved["source_hashes"]) == 40
    assert saved["limitations"]
