from __future__ import annotations

import unittest

from scripts.run_real_station_replay import _season


class RealStationReplayScriptTest(unittest.TestCase):
    def test_season_partition(self):
        self.assertEqual(_season(__import__("pandas").Timestamp("2021-01-01")), "winter")
        self.assertEqual(_season(__import__("pandas").Timestamp("2021-07-01")), "summer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
