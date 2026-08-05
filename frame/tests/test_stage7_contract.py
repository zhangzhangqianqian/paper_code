import json
import tempfile
import unittest
from pathlib import Path

from src.stage7_contract import Stage7ContractError, build_stage7_contract, write_stage7_contract


class Stage7ContractTest(unittest.TestCase):
    def _freeze_config(self):
        path = Path(__file__).resolve().parents[2] / "frame" / "reports" / "stage6_6_kitakyushu" / "stage6_selected_config.json"
        if not path.exists():
            self.skipTest("阶段 6.6 冻结配置尚未生成")
        config = json.loads(path.read_text(encoding="utf-8"))
        if "scheme2r_ablation_reference" not in config:
            self.skipTest("当前冻结配置为旧版，等待 Stage 6-R 重新生成")
        return config, path

    def test_build_contract_uses_frozen_models_and_seeds(self):
        config, path = self._freeze_config()
        contract = build_stage7_contract(config, path)
        self.assertEqual(contract["contract_version"], "stage7.0")
        self.assertEqual(contract["contract_status"], "ready_for_stage7_smoke")
        self.assertEqual(contract["tasks"], ["electricity", "cooling", "heating", "gas"])
        self.assertEqual(contract["models"]["primary"]["model"], "scheme2r")
        self.assertEqual(contract["models"]["primary"]["candidate_id"], "H3")
        self.assertEqual(contract["training_policy"]["formal_random_seeds"], [2026, 2027, 2028, 2029, 2030])
        self.assertTrue(contract["test_set_policy"]["test_reading_allowed_in_stage7"])
        self.assertFalse(contract["test_set_policy"]["test_reading_before_stage7"])

    def test_current_stage6r_freeze_uses_resolved_full_policy(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "frame"
            / "reports"
            / "stage6r_6_kitakyushu"
            / "stage6_selected_config.json"
        )
        if not path.exists():
            self.skipTest("当前 Stage 6-R 冻结文件尚未生成")
        config = json.loads(path.read_text(encoding="utf-8"))
        contract = build_stage7_contract(config, path)
        self.assertEqual(contract["models"]["primary"]["model"], "stl_matched")
        self.assertEqual(contract["models"]["comparison"]["model"], "scheme2r")
        self.assertEqual(contract["training_policy"]["batch_size"], 256)
        self.assertEqual(contract["training_policy"]["max_epochs"], 100)
        self.assertEqual(contract["training_policy"]["early_stopping_patience"], 12)
        self.assertEqual(contract["protocol_training_policies"]["small_sample"]["batch_size"], 32)

    def test_rejects_unfrozen_status(self):
        config, path = self._freeze_config()
        config["freeze_status"] = "draft"
        with self.assertRaises(Stage7ContractError):
            build_stage7_contract(config, path)

    def test_rejects_smoke_source(self):
        config, path = self._freeze_config()
        config["source_results"]["stage6_2"] = "frame/reports/stage6_2/smoke"
        with self.assertRaises(Stage7ContractError):
            build_stage7_contract(config, path)

    def test_writes_contract_and_report(self):
        config, path = self._freeze_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = root / "freeze.json"
            freeze.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            contract_path = root / "configs" / "stage7_contract.json"
            report_path = root / "reports" / "stage7_0_manifest.json"
            report = write_stage7_contract(freeze, contract_path, report_path)
            self.assertEqual(report["stage"], "7.0")
            self.assertTrue(contract_path.exists())
            self.assertTrue(report_path.exists())
