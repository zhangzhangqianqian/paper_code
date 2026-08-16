"""Contract validation for the topology-protocol pilot and branch freeze."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ORDER = ("electricity", "cooling", "heating", "gas")
PHASE_A_MODELS = (("scheme2r", "H4"), ("dynamic_symmetric", "H1"))
PHASE_A_SEEDS = (2026, 2027, 2028)
FORMAL_SEEDS = (2026, 2027, 2028, 2029, 2030)
VALID_BRANCHES = (
    "core_prediction_changed",
    "gas_only_changed",
    "core_conclusion_stable",
)


class TopologyProtocolContractError(ValueError):
    """Raised when the topology protocol is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TopologyProtocolContractError(message)


def load_topology_contract(path: Path | str) -> dict[str, Any]:
    contract_path = Path(path)
    try:
        with contract_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TopologyProtocolContractError(
            f"cannot read topology protocol contract: {contract_path}"
        ) from exc
    _require(isinstance(value, dict), "topology protocol contract must be an object")
    validate_topology_contract(value)
    return value


def _validate_dates(protocol: Mapping[str, Any], name: str) -> None:
    required = (
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    )
    _require(
        all(isinstance(protocol.get(key), str) and protocol[key] for key in required),
        f"{name} must define all calendar boundaries",
    )
    _require(
        protocol["train_end"] < protocol["validation_start"]
        and protocol["validation_end"] < protocol["test_start"],
        f"{name} boundaries must be chronological",
    )


def _validate_model_item(item: Mapping[str, Any], expected: tuple[str, str]) -> None:
    _require(
        (item.get("model"), item.get("candidate_id")) == expected,
        f"unexpected frozen model candidate: {item.get('model')}/{item.get('candidate_id')}",
    )
    hyperparameters = item.get("hyperparameters")
    _require(isinstance(hyperparameters, Mapping), "model hyperparameters are missing")
    _require(
        hyperparameters.get("candidate_id") == expected[1],
        "hyperparameter candidate_id does not match model candidate_id",
    )


def validate_topology_contract(contract: Mapping[str, Any]) -> None:
    _require(
        contract.get("contract_version") == "topology_protocol_pilot_v1",
        "unexpected topology protocol contract version",
    )
    _require(
        contract.get("contract_status") == "frozen_for_phase_a",
        "contract must be frozen_for_phase_a before Task 1 completes",
    )
    _require(contract.get("dataset") == "kitakyushu_energy_station", "dataset mismatch")
    _require(tuple(contract.get("task_order", ())) == TASK_ORDER, "task order mismatch")

    window = contract.get("window")
    _require(isinstance(window, Mapping), "window section is missing")
    _require(window.get("sampling") == "1h", "sampling must be hourly")
    _require(window.get("lookback") == 24, "lookback must be 24")
    _require(window.get("horizon") == 4, "horizon must be 4")
    _require(tuple(window.get("output_shape", ())) == ("N", 4, 4), "output shape mismatch")
    _require(window.get("standardization") == "train_split_only_zscore", "standardization mismatch")
    _require(window.get("future_exogenous_allowed") is False, "future exogenous variables are forbidden")

    protocols = contract.get("protocols")
    _require(isinstance(protocols, Mapping), "protocols section is missing")
    for name in ("cross_topology", "post_ge_regular_operation"):
        protocol = protocols.get(name)
        _require(isinstance(protocol, Mapping), f"missing protocol: {name}")
        _validate_dates(protocol, name)
    _require(
        protocols["cross_topology"]["train_start"] == "2015-01-01 00:00:00",
        "cross_topology must start training in 2015",
    )
    _require(
        protocols["post_ge_regular_operation"]["train_start"] == "2017-01-01 00:00:00",
        "post_ge_regular_operation must start training in 2017",
    )

    phase_a = contract.get("phase_a")
    _require(isinstance(phase_a, Mapping), "phase_a section is missing")
    _require(tuple(phase_a.get("forbidden_years", ())) == (2021,), "Phase A must seal 2021")
    allowed_years = tuple(int(year) for year in phase_a.get("allowed_years", ()))
    _require(allowed_years == (2015, 2016, 2017, 2018, 2019, 2020), "Phase A years mismatch")
    _require(tuple(phase_a.get("seeds", ())) == PHASE_A_SEEDS, "Phase A seed mismatch")
    models = phase_a.get("models")
    _require(isinstance(models, list) and len(models) == 2, "Phase A must contain two models")
    for item, expected in zip(models, PHASE_A_MODELS):
        _require(isinstance(item, Mapping), "Phase A model item must be an object")
        _validate_model_item(item, expected)
    policy = phase_a.get("training_policy")
    _require(isinstance(policy, Mapping), "Phase A training policy is missing")
    expected_policy = {
        "device": "cpu",
        "loss": "SmoothL1Loss",
        "optimizer": "AdamW",
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "batch_size": 256,
        "max_epochs": 100,
        "early_stopping_patience": 12,
        "early_stopping_monitor": "validation_loss",
        "checkpoint_policy": "restore_best_validation_loss",
    }
    for key, expected in expected_policy.items():
        _require(policy.get(key) == expected, f"Phase A training policy mismatch: {key}")

    phase_b = contract.get("phase_b")
    _require(isinstance(phase_b, Mapping), "phase_b section is missing")
    _require(tuple(phase_b.get("formal_seeds", ())) == FORMAL_SEEDS, "formal seed mismatch")
    joint_models = phase_b.get("joint_models")
    expected_joint_models = {
        ("hard_share", "H2"),
        ("dynamic_symmetric", "H1"),
        ("mmoe-lite", "external_fixed"),
        ("ple-lite", "ple_lite_fixed_v1"),
        ("scheme2r", "H4"),
    }
    _require(isinstance(joint_models, list), "phase_b joint_models is missing")
    actual_joint_models = {
        (item.get("model"), item.get("candidate_id"))
        for item in joint_models
        if isinstance(item, Mapping)
    }
    _require(actual_joint_models == expected_joint_models, "Phase B joint model set mismatch")
    branch_matrices = phase_b.get("branch_matrices")
    _require(isinstance(branch_matrices, Mapping), "branch_matrices is missing")
    _require(set(branch_matrices) == set(VALID_BRANCHES), "Phase B branch matrix mismatch")

    statistics = contract.get("statistics")
    _require(isinstance(statistics, Mapping), "statistics section is missing")
    _require(
        statistics.get("method") == "paired_circular_moving_block_bootstrap",
        "the pilot must use circular moving-block bootstrap",
    )
    _require(statistics.get("block_length_origins") == 168, "block length must be 168 origins")
    _require(statistics.get("replicates") == 2000, "bootstrap repetitions must be 2000")
    _require(statistics.get("multiple_testing") == "benjamini_hochberg", "multiple-testing rule mismatch")

    rules = contract.get("branch_rules")
    _require(isinstance(rules, Mapping), "branch_rules section is missing")
    _require(tuple(rules.get("valid_branches", ())) == VALID_BRANCHES, "valid branch order mismatch")
    _require(rules.get("invalid_status") == "pilot_invalid", "invalid status mismatch")


def assert_phase_access(
    contract: Mapping[str, Any],
    *,
    phase: str,
    requested_years: Sequence[int],
    requested_splits: Sequence[str] = (),
) -> None:
    """Reject test-year access before branch freeze and reject unknown phases."""

    validate_topology_contract(contract)
    years = tuple(int(year) for year in requested_years)
    splits = tuple(str(split) for split in requested_splits)
    _require(phase in {"A", "B"}, f"unknown phase: {phase}")
    if phase == "A":
        _require(2021 not in years, "Phase A cannot read 2021")
        _require("test" not in splits, "Phase A cannot construct a test split")
        forbidden = set(contract["phase_a"]["forbidden_years"])
        _require(not forbidden.intersection(years), "Phase A requested a forbidden year")


def resolve_phase_b_matrix(
    contract: Mapping[str, Any], branch: str
) -> Mapping[str, Any]:
    validate_topology_contract(contract)
    _require(branch in VALID_BRANCHES, f"unknown or invalid Phase B branch: {branch}")
    return contract["phase_b"]["branch_matrices"][branch]


__all__ = [
    "FORMAL_SEEDS",
    "PHASE_A_MODELS",
    "PHASE_A_SEEDS",
    "TASK_ORDER",
    "TopologyProtocolContractError",
    "assert_phase_access",
    "load_topology_contract",
    "resolve_phase_b_matrix",
    "validate_topology_contract",
]
