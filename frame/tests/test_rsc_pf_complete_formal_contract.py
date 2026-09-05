from pathlib import Path

import pytest

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract


FRAME_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = FRAME_ROOT / "configs" / "rsc_pf_complete_formal_v1.json"


@pytest.fixture(scope="module")
def contract():
    return CompleteFormalContract.from_path(CONTRACT_PATH)


def test_primary_matrix_is_frozen_and_complete(contract):
    assert contract.primary_method_ids == (
        "RSC-PF", "Decoupled-RSC-PF", "Direct-Policy", "Scheme2R-PTO",
        "State-Conditioned-PTO", "Official iTransformer-PTO", "Differentiable-LP",
        "Seasonal-Naive-PTO", "Perfect-Information-MPC",
    )
    assert len(contract.expected_rows("gate1")) == 37
    assert len(contract.expected_rows("gate2")) == 37


def test_difflp_truthfully_declares_optimizer_at_inference(contract):
    spec = contract.method("Differentiable-LP")
    assert spec.reproduction_level == "cvxpylayers_methodology_adaptation"
    assert spec.inference_lp_calls_per_origin == 1


def test_selection_and_evaluation_origin_counts_are_not_conflated(contract):
    assert contract.selection_origin_count == 8709
    assert contract.evaluation_origin_count == 8757
    assert contract.selection_year == 2019
    assert contract.evaluation_year == 2020


def test_architecture_and_gas_semantics_are_frozen(contract):
    assert contract.payload["architecture"]["latent_control_dim"] == 15
    assert contract.payload["architecture"]["dispatch_dim"] == 21
    assert contract.payload["architecture"]["gas_semantics"] == "station_side_auxiliary_prior"
    assert contract.payload["architecture"]["allow_future_binary_decisions"] is False


def test_evaluation_access_requires_authorized_gate1(contract, tmp_path):
    with pytest.raises(PermissionError, match="authorized Gate 1"):
        contract.authorize_evaluation(tmp_path / "missing_gate1_transition.json")


def test_evaluation_access_accepts_only_matching_gate1_transition(contract, tmp_path):
    transition = tmp_path / "gate1_transition.json"
    transition.write_text(
        '{"contract_sha256": "wrong", "authorized_gate2": true, '
        '"evaluation_year_accessed": false, "test_set_accessed": false}',
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="does not authorize"):
        contract.authorize_evaluation(transition)

    transition.write_text(
        '{"contract_sha256": "' + contract.contract_sha256 + '", '
        '"authorized_gate2": true, "evaluation_year_accessed": false, '
        '"test_set_accessed": false}',
        encoding="utf-8",
    )
    authorized = contract.authorize_evaluation(transition)
    assert authorized["authorized_gate2"] is True

