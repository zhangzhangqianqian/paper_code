from __future__ import annotations

import pytest

from src.joint_dispatch.formal_v4_resources import ResourceProjection, benchmark_callable, project_resource_budget


def test_resource_projection_enforces_runtime_and_margin_thresholds():
    projection = project_resource_budget({"model": {"p95": 0.01}}, windows=100, methods=1, seeds=1, disk_margin_fraction=0.5, memory_margin_fraction=0.5)
    projection.validate()
    with pytest.raises(ValueError, match="runtime"):
        ResourceProjection({}, 25.0, 0.5, 0.5).validate()
    with pytest.raises(ValueError, match="margin"):
        ResourceProjection({}, 1.0, 0.1, 0.5).validate()


def test_callable_benchmark_reports_percentiles():
    stats = benchmark_callable("noop", lambda: None, warmup=1, iterations=3)
    assert stats["samples"] == 3.0
    assert stats["p95"] >= stats["p50"] >= 0.0
