"""阶段10.1参数证据与可辨识性审计测试。"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.contracts import load_scheduling_contract  # noqa: E402
from src.scheduling.parameter_audit import (  # noqa: E402
    audit_identifiability,
    read_parameter_ledger,
)


class SchedulingParameterAuditTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_scheduling_contract()
        self.ledger_path = PROJECT_ROOT / "configs" / "scheduling_parameter_ledger_v2.csv"

    def test_ledger_is_complete_and_traceable(self):
        ledger = read_parameter_ledger(self.ledger_path)
        self.assertGreaterEqual(len(ledger.records), 10)
        report = audit_identifiability(
            self.contract,
            ledger,
            {"scaling_years": self.contract.train_years, "test_year_used_for_scaling": False},
        )
        self.assertTrue(report.passed, report.to_dict())
        self.assertEqual(report.parameter_evidence, "pass")
        self.assertEqual(report.decision_space, "pass")

    def test_missing_required_field_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "bad.csv"
            with self.ledger_path.open("r", encoding="utf-8") as source, target.open("w", encoding="utf-8", newline="") as output:
                rows = list(csv.DictReader(source))
                fields = list(rows[0])
                fields.remove("validation_range")
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerow({key: value for key, value in rows[0].items() if key in fields})
            with self.assertRaises(ValueError):
                read_parameter_ledger(target)

    def test_test_year_scaling_fails_the_audit(self):
        ledger = read_parameter_ledger(self.ledger_path)
        report = audit_identifiability(
            self.contract,
            ledger,
            {"scaling_years": self.contract.train_years, "test_year_used_for_scaling": True},
        )
        self.assertFalse(report.passed)
        self.assertIn("test_year_used_for_scaling必须为false", report.issues)

    def test_missing_intertemporal_state_fails(self):
        ledger = read_parameter_ledger(self.ledger_path)
        records = tuple(record for record in ledger.records if record.component != "bess")
        ledger_without_bess = type(ledger)(records=records)
        report = audit_identifiability(
            self.contract,
            ledger_without_bess,
            {"scaling_years": self.contract.train_years, "test_year_used_for_scaling": False},
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.decision_space, "fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
