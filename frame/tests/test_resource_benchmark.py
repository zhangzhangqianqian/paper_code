import unittest

import torch
from torch import nn

from src.resource_benchmark import benchmark_model_resources


class ResourceBenchmarkTests(unittest.TestCase):
    def test_resource_fields_are_recorded(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2))
        loads = torch.zeros(2, 3)
        exog = torch.zeros(2, 1)

        class Wrapper(nn.Module):
            def __init__(self, child):
                super().__init__()
                self.child = child

            def forward(self, loads, exog):
                return self.child(loads)

        result = benchmark_model_resources(Wrapper(model), loads, exog, warmup=1, iterations=2)
        self.assertGreater(result["parameter_count"], 0)
        self.assertGreater(result["macs_per_batch_estimated"], 0)
        self.assertGreater(result["cpu_latency_ms_median"], 0.0)

    def test_loads_only_mode_is_supported(self):
        model = nn.Linear(3, 2)

        class Wrapper(nn.Module):
            def forward(self, loads):
                return model(loads)

        result = benchmark_model_resources(
            Wrapper(), torch.zeros(2, 3), None,
            warmup=1, iterations=2, input_mode="loads_only"
        )
        self.assertEqual(result["input_mode"], "loads_only")
        self.assertEqual(result["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
