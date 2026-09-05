from pathlib import Path

from scripts.audit_rsc_pf_complete_formal_inventory import FRAME_ROOT, build_inventory


def test_inventory_does_not_confuse_contract_with_completed_result():
    result = build_inventory(FRAME_ROOT)
    diff = result["methods"]["Differentiable-LP"]
    assert diff["training_function_present"] is True
    assert diff["native_gradient_gate_present"] is True
    assert diff["formal_result_present"] is False
    assert diff["status"] in {"missing_formal_result", "checkpoint_available"}
    assert result["matched_v2"]["formal_candidate"] is False


def test_inventory_records_primary_methods_and_future_access_boundary():
    result = build_inventory(FRAME_ROOT)
    assert "State-Conditioned-PTO" in result["primary_methods"]
    assert result["methods"]["Official iTransformer-PTO"]["checkpoint_available"] is True
    assert result["methods"]["Seasonal-Naive-PTO"]["training_function_present"] is False
    assert result["methods"]["Perfect-Information-MPC"]["training_function_present"] is False
    assert all("J_decoupled.pt" not in path for path in result["methods"]["RSC-PF"]["checkpoint_evidence"])
    assert all("J_joint.pt" not in path for path in result["methods"]["Decoupled-RSC-PF"]["checkpoint_evidence"])
    assert result["matched_v2"]["row_count"] == 16
    assert result["matched_v2"]["rsc_pf_seeds"] == [2026]
    assert result["future_access"] == {
        "evaluation_year_accessed": False,
        "excluded_year_accessed": False,
        "test_set_accessed": False,
    }


def test_inventory_is_read_only_and_does_not_require_an_output_path():
    before = Path(FRAME_ROOT) / "reports" / "rsc_pf_matched_closed_loop_2019" / "v2"
    result = build_inventory(FRAME_ROOT)
    assert result["read_only"] is True
    assert before.is_dir()
