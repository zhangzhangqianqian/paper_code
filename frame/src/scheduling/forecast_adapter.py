"""阶段10.5：复用 Stage 7 正式预测并对齐到调度输入。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np


TASK_ORDER: Tuple[str, ...] = ("electricity", "cooling", "heating", "gas")


@dataclass(frozen=True)
class ForecastRun:
    model: str
    candidate: str
    seed: int
    protocol: str
    prediction: np.ndarray
    origin_times: np.ndarray
    source_path: Path
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class DispatchForecastSet:
    origin_times: np.ndarray
    demand: np.ndarray
    gas: np.ndarray
    renewable: np.ndarray
    clipped_negative_count: int
    source_model: str


def _find_run_directory(source_dir: Path, model: str, candidate: str, seed: int) -> Path:
    expected_name = f"seed_{int(seed)}"
    matches = []
    for prediction_path in source_dir.rglob("predictions_test.npz"):
        run_dir = prediction_path.parent
        if run_dir.name != expected_name:
            continue
        parts = {part.lower() for part in run_dir.parts}
        if model.lower() not in parts or candidate.lower() not in parts:
            continue
        matches.append(run_dir)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"无法唯一定位预测运行：model={model}, candidate={candidate}, seed={seed}; matches={matches}"
        )
    return matches[0]


def load_forecast_run(
    model: str,
    candidate: str,
    seed: int,
    source_dir: str | Path,
) -> ForecastRun:
    """从正式运行目录读取已逆标准化的 test prediction。"""

    root = Path(source_dir)
    run_dir = _find_run_directory(root, model, candidate, seed)
    prediction_file = run_dir / "predictions_test.npz"
    manifest_file = run_dir / "run_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"预测运行缺少run_manifest.json：{run_dir}")
    import json

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    with np.load(prediction_file, allow_pickle=False) as payload:
        prediction = np.asarray(payload["prediction"], dtype=np.float64)
        if "target_times" not in payload:
            raise ValueError(f"预测文件缺少target_times：{prediction_file}")
        origin_times = np.asarray(payload["target_times"], dtype="datetime64[ns]")
    if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (4, 4):
        raise ValueError(f"预测张量必须为[N,4,4]，实际为{prediction.shape}")
    if len(origin_times) != prediction.shape[0]:
        raise ValueError("origin_times数量与预测样本数不一致")
    tasks = tuple(manifest.get("tasks", ()))
    if tasks and tasks != TASK_ORDER:
        raise ValueError(f"预测运行任务顺序错误：{tasks}")
    protocol = str(manifest.get("protocol", "unknown"))
    return ForecastRun(
        model=model,
        candidate=candidate,
        seed=int(seed),
        protocol=protocol,
        prediction=prediction,
        origin_times=origin_times,
        source_path=prediction_file,
        manifest=manifest,
    )


def align_dispatch_inputs(
    load_forecast: ForecastRun,
    renewable_forecast: Mapping[str, object] | np.ndarray,
) -> DispatchForecastSet:
    """将四任务预测和 PV/WT 预测对齐为调度器输入。"""

    prediction = np.asarray(load_forecast.prediction, dtype=np.float64)
    if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (4, 4):
        raise ValueError("负荷预测必须为[N,4,4]")
    if isinstance(renewable_forecast, Mapping):
        renewable = np.asarray(renewable_forecast["predictions"], dtype=np.float64)
        renewable_times = np.asarray(renewable_forecast["origin_times"], dtype="datetime64[ns]")
    else:
        renewable = np.asarray(renewable_forecast, dtype=np.float64)
        renewable_times = load_forecast.origin_times
    if renewable.ndim != 3 or tuple(renewable.shape[1:]) != (4, 2):
        raise ValueError("PV/WT预测必须为[N,4,2]")
    if len(renewable_times) != len(renewable) or not np.array_equal(
        renewable_times, load_forecast.origin_times
    ):
        raise ValueError("负荷预测与PV/WT预测的origin_times未严格对齐")
    negative_mask = prediction < 0.0
    clipped = np.maximum(prediction, 0.0)
    return DispatchForecastSet(
        origin_times=load_forecast.origin_times.copy(),
        demand=clipped[:, :, :3],
        gas=clipped[:, :, 3],
        renewable=np.maximum(renewable, 0.0),
        clipped_negative_count=int(negative_mask.sum()),
        source_model=load_forecast.model,
    )
