"""Bounded resource benchmarking and fail-closed projections."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class ResourceProjection:
    component_seconds: Mapping[str, Mapping[str, float]]
    projected_p95_hours: float
    disk_margin_fraction: float
    memory_margin_fraction: float
    method_id: str = "formal-v4.1"

    def validate(self, *, max_projected_hours: float = 24.0, minimum_margin: float = 0.20) -> None:
        if self.projected_p95_hours > float(max_projected_hours):
            raise ValueError("projected p95 runtime exceeds formal-v4 limit")
        if self.disk_margin_fraction < float(minimum_margin) or self.memory_margin_fraction < float(minimum_margin):
            raise ValueError("disk or memory margin is below formal-v4 limit")
        if self.projected_p95_hours < 0.0:
            raise ValueError("projected runtime must be non-negative")

    def to_payload(self) -> dict[str, Any]:
        return {"schema_version": "formal-v4.1-resource-projection-v1", "method_id": self.method_id, "component_seconds": {str(k): dict(v) for k, v in self.component_seconds.items()}, "projected_p95_hours": self.projected_p95_hours, "disk_margin_fraction": self.disk_margin_fraction, "memory_margin_fraction": self.memory_margin_fraction}

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite resource receipt: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class MethodResourceProjection:
    method_id: str
    sample_count: int
    p50_seconds: float
    p95_seconds: float
    projected_hours: float
    peak_memory_bytes: int

    def validate(self) -> None:
        if self.sample_count != 500:
            raise ValueError("each formal-v4 resource row must contain 500 samples")
        values = (self.p50_seconds, self.p95_seconds, self.projected_hours)
        if not all(float(value) >= 0.0 for value in values):
            raise ValueError(f"resource projection values must be non-negative for {self.method_id}")
        if self.peak_memory_bytes < 0:
            raise ValueError(f"resource projection peak memory is negative for {self.method_id}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "method_id": self.method_id,
            "sample_count": self.sample_count,
            "p50_seconds": self.p50_seconds,
            "p95_seconds": self.p95_seconds,
            "projected_hours": self.projected_hours,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


@dataclass(frozen=True)
class ResourceProjectionReceipt:
    rows: Sequence[MethodResourceProjection]
    worst_case_method_id: str
    worst_case_projected_hours: float
    free_memory_fraction: float
    free_disk_fraction: float

    def validate(self, *, max_projected_hours: float = 24.0, minimum_margin: float = 0.20) -> None:
        if not self.rows:
            raise ValueError("resource projection must contain method rows")
        for row in self.rows:
            row.validate()
        ids = {row.method_id for row in self.rows}
        if self.worst_case_method_id not in ids:
            raise ValueError("worst-case resource method is not present in rows")
        maximum = max(float(row.projected_hours) for row in self.rows)
        if abs(float(self.worst_case_projected_hours) - maximum) > 1.0e-9:
            raise ValueError("worst-case projected duration is not the maximum row")
        if maximum > float(max_projected_hours):
            raise ValueError("projected p95 runtime exceeds formal-v4 limit")
        if float(self.free_memory_fraction) < float(minimum_margin) or float(self.free_disk_fraction) < float(minimum_margin):
            raise ValueError("disk or memory margin is below formal-v4 limit")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": "formal-v4.1-resource-projection-v1",
            "rows": [row.to_dict() for row in self.rows],
            "worst_case_method_id": self.worst_case_method_id,
            "worst_case_projected_hours": self.worst_case_projected_hours,
            "free_memory_fraction": self.free_memory_fraction,
            "free_disk_fraction": self.free_disk_fraction,
        }


def benchmark_callable(name: str, callable_: Callable[[], Any], *, warmup: int = 1, iterations: int = 5) -> dict[str, float]:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    for _ in range(warmup):
        callable_()
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter(); callable_(); timings.append(time.perf_counter() - start)
    values = sorted(timings)
    p95_index = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
    return {"p50": float(statistics.median(values)), "p95": float(values[p95_index]), "max": float(max(values)), "samples": float(len(values))}


def project_resource_budget(component_seconds: Mapping[str, Mapping[str, float]], *, windows: int, methods: int, seeds: int, disk_margin_fraction: float, memory_margin_fraction: float, safety_factor: float = 2.0) -> ResourceProjection:
    if windows <= 0 or methods <= 0 or seeds <= 0 or safety_factor <= 0.0:
        raise ValueError("resource projection counts and safety factor must be positive")
    total_seconds = sum(float(value.get("p95", 0.0)) for value in component_seconds.values()) * windows * methods * seeds * safety_factor
    return ResourceProjection(component_seconds, total_seconds / 3600.0, float(disk_margin_fraction), float(memory_margin_fraction))


__all__ = ["MethodResourceProjection", "ResourceProjection", "ResourceProjectionReceipt", "benchmark_callable", "project_resource_budget"]
