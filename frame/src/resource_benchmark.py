"""轻量模型资源记录接口。

正式实验不在每个训练 epoch 内重复做 1000 次计时；模型训练完成后，
可在独立脚本中调用 :func:`benchmark_model_resources`，将结构、CPU 延迟、
吞吐量和可估算的 MACs 写入资源表。
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, Mapping

import torch
from torch import nn

try:  # optional; the benchmark remains usable without psutil
    import psutil
except ImportError:  # pragma: no cover - depends on local environment
    psutil = None


def _mac_counter(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: object) -> int:
    if isinstance(module, nn.Linear):
        batch = (
            int(inputs[0].numel() // module.in_features)
            if inputs and isinstance(inputs[0], torch.Tensor)
            else 1
        )
        return int(batch * module.in_features * module.out_features)
    if isinstance(module, nn.Conv1d):
        if not inputs or not isinstance(output, torch.Tensor):
            return 0
        batch, channels, length = output.shape
        kernel = module.kernel_size[0]
        return int(batch * length * module.out_channels * (module.in_channels // module.groups) * kernel)
    return 0


def benchmark_model_resources(
    model: nn.Module,
    loads: torch.Tensor,
    exog: torch.Tensor | None,
    *,
    warmup: int = 100,
    iterations: int = 1000,
    input_mode: str = "loads_and_exog",
) -> Dict[str, Any]:
    """Benchmark one model on CPU and return reproducible resource fields.

    ``loads`` and ``exog`` should already have the frozen input shape. For a
    loads-only model pass ``exog=None`` and ``input_mode="loads_only"``. The
    function is deliberately explicit about its benchmark counts so a report
    can distinguish a formal result from a short smoke timing.
    """

    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup必须非负且iterations必须为正数")
    if input_mode not in {"loads_only", "loads_and_exog"}:
        raise ValueError("input_mode must be loads_only or loads_and_exog")
    if input_mode == "loads_and_exog" and exog is None:
        raise ValueError("loads_and_exog benchmark requires exog")
    model = model.to("cpu").eval()
    loads = loads.to("cpu")
    if exog is not None:
        exog = exog.to("cpu")
    macs = 0
    hooks = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: object) -> None:
        nonlocal macs
        macs += _mac_counter(module, inputs, output)

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            hooks.append(module.register_forward_hook(hook))
    def forward() -> object:
        if input_mode == "loads_only":
            return model(loads)
        return model(loads, exog)

    with torch.inference_mode():
        for _ in range(warmup):
            forward()
        samples: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            forward()
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    for handle in hooks:
        handle.remove()
    samples.sort()
    q1 = samples[len(samples) // 4]
    q3 = samples[(3 * len(samples)) // 4]
    median = statistics.median(samples)
    batch = int(loads.shape[0])
    peak_rss_bytes = None
    if psutil is not None:
        try:
            peak_rss_bytes = int(psutil.Process().memory_info().rss)
        except (OSError, psutil.Error):
            peak_rss_bytes = None
    return {
        "parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "macs_per_batch_estimated": int(macs // max(warmup + iterations, 1)),
        "flops_per_batch_estimated": int(2 * (macs // max(warmup + iterations, 1))),
        "benchmark_warmup": int(warmup),
        "benchmark_iterations": int(iterations),
        "cpu_latency_ms_median": float(median),
        "cpu_latency_ms_iqr": float(q3 - q1),
        "cpu_throughput_samples_per_second": float(batch / (median / 1000.0)),
        "input_mode": input_mode,
        "batch_size": batch,
        "peak_rss_bytes": peak_rss_bytes,
    }
