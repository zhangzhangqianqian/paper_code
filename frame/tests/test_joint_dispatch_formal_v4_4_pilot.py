from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.joint_dispatch.formal_v4_4_artifacts import sha256_file
from src.joint_dispatch.formal_v4_4_contract import load_formal_v4_4_contract
from src.joint_dispatch.formal_v4_4_pilot import EXPECTED_ROWS, run_pilot_v44


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_4.json"
REPORT = ROOT / "reports" / "joint_forecast_dispatch_formal_v4_2" / "formal_v4_2_20260905_g"
SOURCE = REPORT / "protocol" / "SOURCE_MANIFEST.json"
TRAIN = REPORT / "data" / "base_train.npz"
SELECTION = REPORT / "data" / "base_selection.npz"
BENCHMARK = REPORT / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
CAPACITY = REPORT / "gate0" / "CAPACITY_FREEZE.json"


def _authorized_gate0(tmp_path: Path) -> Path:
    contract = load_formal_v4_4_contract(CONTRACT); root = tmp_path / "gate0"; gate = root / "gate0"; gate.mkdir(parents=True)
    digest = sha256_file(SOURCE); train_hash = sha256_file(TRAIN); selection_hash = sha256_file(SELECTION)
    (root / "GATE0_TRANSITION.json").write_text(json.dumps({"schema": "formal-v4.4-gate0-transition-v1", "run_id": "gate0", "contract_sha256": contract.contract_sha256, "authorized_pilot": True, "evaluation_year_accessed": False}), encoding="utf-8")
    (gate / "GATE0_RECEIPT.json").write_text(json.dumps({"source_manifest_sha256": digest, "train_data_sha256": train_hash, "selection_data_sha256": selection_hash}), encoding="utf-8")
    np.savez_compressed(gate / "PILOT_SPLIT.npz", train=np.array([1, 2]), early_stop=np.array([3]), selection_full=np.array([0, 1]), selection_stress=np.array([0]))
    return root / "GATE0_TRANSITION.json"


def _executor(**_: object) -> dict:
    target = np.zeros((3, 4, 4)); target[..., 0] = 10.0; target[..., 3] = 2.0
    regimes = np.array([[0, 1, 1, 0], [0, 2, 2, 0], [1, 1, 2, 2]])
    target[regimes == 1, 1] = 5.0; target[regimes == 2, 2] = 5.0
    probability = np.full((3, 4, 3), 0.02)
    for i in range(3):
        for h in range(4): probability[i, h, regimes[i, h]] = 0.96
    return {
        "prediction": target.copy(), "target": target, "probability": probability, "prior_probability": np.roll(probability, 1, axis=-1), "regimes": regimes, "times": np.arange(12).reshape(3, 4),
        "comparisons": {"leakage_ratio": {"cooling": 0.5, "heating": 0.5}, "active_wape_ratio": {"cooling": 1.0, "heating": 1.0}, "electricity_gas_wape_ratio": {"electricity": 1.0, "gas": 1.0}, "four_task_score_ratio": 1.0},
        "joint": {"penalized_objective": 1.0, "shortage": 0.1, "gradient_norms": {"decision_to_gate": 1.0, "decision_to_magnitude": 1.0, "decision_to_scheduler": 1.0}},
        "decoupled": {"penalized_objective": 1.2, "shortage": 0.2, "gradient_norms": {"decision_to_base": 0.0}}, "physics": {"max_residual": 1.0e-8}, "rows": EXPECTED_ROWS,
    }


def test_pilot_runs_all_stages_without_2020(tmp_path: Path) -> None:
    transition = _authorized_gate0(tmp_path)
    decision = run_pilot_v44(contract_path=CONTRACT, gate0_transition=transition, source_manifest=SOURCE, base_train_data=TRAIN, base_selection_data=SELECTION, benchmark=BENCHMARK, capacity_receipt=CAPACITY, output_root=tmp_path, run_id="pilot", stage_executor=_executor)
    assert decision.accessed_years == (2015, 2016, 2017, 2018, 2019)
    assert set(decision.rows) == set(EXPECTED_ROWS)
    assert len(decision.audit_sha256) == 64
