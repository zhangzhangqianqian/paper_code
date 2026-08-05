import unittest

from scripts.generate_formal_run_matrix import build_formal_run_matrix


class FormalRunMatrixTests(unittest.TestCase):
    def test_matrix_has_no_duplicates_and_reuses_a4(self):
        freeze = {
            "primary_model": {"model": "scheme2r", "candidate_id": "H2"},
            "comparison_model": {"model": "dynamic_symmetric", "candidate_id": "H3"},
        }
        rows = build_formal_run_matrix(freeze, {})
        self.assertEqual(len(rows), 114)
        self.assertEqual(sum(row["execution"] == "train" for row in rows), 100)
        self.assertEqual(sum(row["execution"] == "reuse" for row in rows), 10)
        self.assertEqual(sum(row["execution"] == "deterministic" for row in rows), 4)
        fingerprints = {
            (row["stage"], row["protocol"], row["model"], row["candidate_id"], row["seed"], row["execution"])
            for row in rows
        }
        self.assertEqual(len(fingerprints), len(rows))
        self.assertEqual(sum(row["model"] == "A4" for row in rows), 10)
        self.assertEqual(sum(row["model"] == "stl_matched" for row in rows), 10)


if __name__ == "__main__":
    unittest.main()
