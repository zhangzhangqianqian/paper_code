from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_scheduling_formal_v2 import (  # noqa: E402
    _complete_run,
    _scenario_parameters,
    _write_run_atomically,
    build_run_matrix,
    filter_run_matrix,
)


class FormalRunnerTest(unittest.TestCase):
    def test_matrix_is_five_models_five_seeds_and_six_scenarios(self):
        matrix = build_run_matrix({"models": ["a", "b", "c", "d", "e"], "seeds": [1, 2, 3, 4, 5], "simulated_scenarios": ["core", "a", "b", "c", "d", "e"]})
        self.assertEqual(len(matrix), 5 * 5 * 7)
        self.assertEqual(sum(row["track"] == "real_replay" for row in matrix), 25)
        self.assertEqual(sum(row["track"] == "simulated_dispatch" for row in matrix), 150)

    def test_scenarios_change_only_the_declared_degrees_of_freedom(self):
        base = {
            "bess_power_capacity": 2.0,
            "bess_energy_capacity": 8.0,
            "chp_electric_capacity": 3.0,
            "chp_heat_capacity": 4.0,
            "carbon_price_sensitivity": 0.2,
            "carbon_price_default": 0.0,
        }
        no_bess = _scenario_parameters(base, "no_bess")
        self.assertEqual(no_bess["bess_power_capacity"], 0.0)
        self.assertEqual(no_bess["bess_energy_capacity"], 0.0)
        self.assertEqual(no_bess["chp_electric_capacity"], 3.0)
        no_chp = _scenario_parameters(base, "no_chp")
        self.assertEqual(no_chp["chp_electric_capacity"], 0.0)
        self.assertEqual(no_chp["chp_heat_capacity"], 0.0)
        carbon = _scenario_parameters(base, "carbon_price_sensitivity")
        self.assertEqual(carbon["carbon_price"], 0.2)

    def test_run_writer_is_atomic_and_hash_checked(self):
        run = {"track": "simulated_dispatch", "model": "scheme2r", "seed": 2026, "scenario": "core"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final, _ = _write_run_atomically(root, run, pd.DataFrame({"value": [1.0]}), {"value_mean": 1.0}, None)
            self.assertTrue(_complete_run(final))
            self.assertTrue((final / "rows.csv").exists())
            manifest = json.loads((final / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "scheduling-formal-run-v3")

    def test_smoke_filters_do_not_change_the_frozen_matrix(self):
        contract = {"models": ["a", "b"], "seeds": [1, 2], "simulated_scenarios": ["core", "x"]}
        full = build_run_matrix(contract)
        selected = filter_run_matrix(full, models=["a"], seeds=[1], scenarios=["core"])
        self.assertEqual(len(full), 2 * 2 * 3)
        self.assertEqual(len(selected), 1)
        self.assertTrue(all(row["model"] == "a" and row["seed"] == 1 for row in selected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
