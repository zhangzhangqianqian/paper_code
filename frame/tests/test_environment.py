"""阶段0环境测试。

运行方式：
    D:\anaconda\envs\pytorch\python.exe frame\tests\test_environment.py
"""

import os
import sys
import unittest

import torch


EXPECTED_THREADS = 8


class EnvironmentTest(unittest.TestCase):
    def test_python_and_torch_versions_are_available(self):
        self.assertGreaterEqual(sys.version_info[:2], (3, 9))
        self.assertTrue(torch.__version__.startswith("2.8."))

    def test_cpu_only_execution(self):
        self.assertFalse(torch.cuda.is_available())
        torch.set_num_threads(EXPECTED_THREADS)
        self.assertEqual(torch.get_num_threads(), EXPECTED_THREADS)

    def test_basic_tensor_forward_and_backward(self):
        torch.manual_seed(2026)
        x = torch.randn(4, 24, 3, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(3, 3, dtype=torch.float32, requires_grad=True)
        output = x @ weight
        loss = output.square().mean()
        loss.backward()

        self.assertEqual(tuple(output.shape), (4, 24, 3))
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.isfinite(loss).item())

    def test_random_seed_reproducibility(self):
        torch.manual_seed(2026)
        first = torch.randn(8)
        torch.manual_seed(2026)
        second = torch.randn(8)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main(verbosity=2)
