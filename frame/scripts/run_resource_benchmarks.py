"""Benchmark CPU inference resources for the frozen Stage 7-R checkpoints.

This script never calls ``fit_model``.  It reconstructs the frozen model
interfaces, loads existing checkpoints, and records batch-1 latency together
with batch-128 throughput.  The resulting CSV is consumed by Stage 7.6 and
kept separate from the immutable prediction artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_stage7_3 import (  # noqa: E402
    _build_model,
    _standardize_protocol,
)
from run_stage7_4 import _build_ablation_model  # noqa: E402
from run_stage7_6 import (  # noqa: E402
    _iter_runs,
    build_stage7_6_source_plan,
    stage7_6_expected_counts,
)
from run_stage7_stl_reference import _build_task_model  # noqa: E402
from src.baselines import persistence_forecast, seasonal_naive_forecast  # noqa: E402
from src.data_pipeline import select_training_frame  # noqa: E402
from src.external_models import (  # noqa: E402
    DLinearBaseline,
    MMoELiteBaseline,
    SOFTSBaseline,
)
from src.kitakyushu_pipeline import (  # noqa: E402
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    KITAKYUSHU_TASKS,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)
from src.models import build_forecasting_model  # noqa: E402
from src.resource_benchmark import benchmark_model_resources  # noqa: E402
from src.training import load_checkpoint  # noqa: E402


FORMAL_PROTOCOLS = ("full", "small_sample")
HORIZON = 4
LOOKBACK = 24


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class _IndependentSTLWrapper(nn.Module):
    """Run the four independent matched-STL task models as one interface."""

    def __init__(self, models: Sequence[nn.Module]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, loads: Tensor, exog: Tensor | None = None) -> Tensor:
        if exog is None:
            raise ValueError("matched STL requires historical exogenous inputs")
        return torch.cat([model(loads, exog) for model in self.models], dim=-1)


class _PersistenceModule(nn.Module):
    def __init__(self, horizon: int = HORIZON) -> None:
        super().__init__()
        self.horizon = horizon

    def forward(self, loads: Tensor, exog: Tensor | None = None) -> Tensor:
        return loads[:, -1:, :].repeat(1, self.horizon, 1)


class _SeasonalNaiveModule(nn.Module):
    def __init__(self, horizon: int = HORIZON, season_length: int = 24) -> None:
        super().__init__()
        self.horizon = horizon
        self.season_length = season_length

    def forward(self, loads: Tensor, exog: Tensor | None = None) -> Tensor:
        cycle = loads[:, -self.season_length :, :]
        indices = torch.arange(self.horizon, device=loads.device) % self.season_length
        return cycle.index_select(1, indices)


def _protocol_windows(data_dir: Path) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    frame, _ = read_kitakyushu_canonical(data_dir, years=tuple(range(2015, 2022)))
    cleaned, _ = clean_kitakyushu_dataframe(frame)
    windows: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for protocol, split in (
        ("full", KITAKYUSHU_SPLIT),
        ("small_sample", KITAKYUSHU_SMALL_SAMPLE_SPLIT),
    ):
        windows[protocol], _ = _standardize_protocol(cleaned, split)
    return windows


def _scheme2r_hyperparameters(freeze: Mapping[str, object], candidate_id: str) -> Mapping[str, object]:
    for key in ("primary_model", "comparison_model", "scheme2r_ablation_reference"):
        item = freeze.get(key)
        if isinstance(item, Mapping) and str(item.get("candidate_id")) == candidate_id:
            hp = item.get("hyperparameters")
            if isinstance(hp, Mapping):
                return hp
    raise ValueError(f"no frozen hyperparameters for candidate {candidate_id}")


def _build_benchmark_model(
    manifest: Mapping[str, object], freeze: Mapping[str, object]
) -> Tuple[nn.Module, str, List[Path]]:
    model_name = str(manifest.get("model"))
    candidate_id = str(manifest.get("candidate_id"))
    input_mode = str(manifest.get("input_mode", "loads_and_exog"))
    model_config = manifest.get("model_config")
    if not isinstance(model_config, Mapping):
        model_config = {}
    run_dir = Path(str(manifest["run_dir"]))

    if model_name == "persistence":
        return _PersistenceModule(), "loads_only", []
    if model_name == "seasonal_naive":
        return _SeasonalNaiveModule(), "loads_only", []
    if model_name == "dlinear":
        return (
            DLinearBaseline(
                lookback=LOOKBACK,
                horizon=HORIZON,
                task_count=len(KITAKYUSHU_TASKS),
                moving_avg=int(model_config.get("moving_avg", 5)),
            ),
            "loads_only",
            [run_dir / "best_model.pt"],
        )
    if model_name == "mmoe-lite":
        return (
            MMoELiteBaseline(
                lookback=LOOKBACK,
                horizon=HORIZON,
                task_count=len(KITAKYUSHU_TASKS),
                exog_dim=len(KITAKYUSHU_EXOG_COLUMNS),
                expert_count=int(model_config.get("expert_count", 4)),
                expert_hidden_dim=int(model_config.get("expert_hidden_dim", 32)),
                representation_dim=int(model_config.get("representation_dim", 32)),
                head_hidden_dim=int(model_config.get("head_hidden_dim", 16)),
                dropout=float(model_config.get("dropout", 0.1)),
            ),
            input_mode,
            [run_dir / "best_model.pt"],
        )
    if model_name == "softs":
        return (
            SOFTSBaseline(
                lookback=LOOKBACK,
                horizon=HORIZON,
                task_count=len(KITAKYUSHU_TASKS),
                d_model=int(model_config.get("d_model", 32)),
                d_core=int(model_config.get("d_core", 16)),
                d_ff=int(model_config.get("d_ff", 64)),
                e_layers=int(model_config.get("e_layers", 1)),
                dropout=float(model_config.get("dropout", 0.1)),
                use_instance_norm=bool(model_config.get("use_instance_norm", True)),
                stochastic_pooling=bool(model_config.get("stochastic_pooling", True)),
            ),
            "loads_only",
            [run_dir / "best_model.pt"],
        )
    if model_name == "scheme2r_loads_only":
        hp = _scheme2r_hyperparameters(freeze, candidate_id.replace("scheme2r_ablation_reference", "H4"))
        return (
            build_forecasting_model(
                "scheme2r",
                exog_dim=0,
                task_count=len(KITAKYUSHU_TASKS),
                hidden_dim=int(hp["hidden_dim"]),
                lookback=LOOKBACK,
                kernel_size=int(hp["scheme2r_kernel_size"]),
                dilations=tuple(hp["scheme2r_dilations"]),
                dropout=float(hp["dropout"]),
                horizon=HORIZON,
                head_hidden_dim=int(hp["prediction_head_hidden_dim"]),
                gate_hidden_dim=int(hp["scheme2r_gate_hidden_dim"]),
                step_embedding_dim=int(hp["scheme2r_step_embedding_dim"]),
                rank=int(hp["scheme2r_rank"]),
            ),
            "loads_only",
            [run_dir / "best_model.pt"],
        )

    hp = _scheme2r_hyperparameters(freeze, candidate_id)
    if model_name == "stl_matched":
        task_models = [
            _build_task_model(index, hp) for index in range(len(KITAKYUSHU_TASKS))
        ]
        checkpoints = []
        task_checkpoints = manifest.get("task_checkpoints", {})
        for task in KITAKYUSHU_TASKS:
            item = task_checkpoints.get(task, {}) if isinstance(task_checkpoints, Mapping) else {}
            checkpoints.append(run_dir / str(item.get("file", f"best_model_{task}.pt")))
        return _IndependentSTLWrapper(task_models), input_mode, checkpoints
    if model_name.startswith("A"):
        return _build_ablation_model(model_name, hp, len(KITAKYUSHU_EXOG_COLUMNS)), input_mode, [run_dir / "best_model.pt"]
    if model_name == "scheme2r":
        return _build_model(model_name, hp, len(KITAKYUSHU_EXOG_COLUMNS)), input_mode, [run_dir / "best_model.pt"]
    raise ValueError(f"unsupported resource benchmark model: {model_name}")


def _load_model(model: nn.Module, checkpoint_paths: Sequence[Path]) -> None:
    if not checkpoint_paths:
        return
    if isinstance(model, _IndependentSTLWrapper):
        if len(checkpoint_paths) != len(model.models):
            raise ValueError("matched STL checkpoint count does not match task model count")
        for child, path in zip(model.models, checkpoint_paths):
            load_checkpoint(child, path, "cpu")
        return
    load_checkpoint(model, checkpoint_paths[0], "cpu")


def _benchmark_one(
    record: Mapping[str, object],
    windows: Mapping[str, Mapping[str, np.ndarray]],
    freeze: Mapping[str, object],
    warmup: int,
    iterations: int,
    threads: int,
) -> Dict[str, object]:
    manifest = dict(record["manifest"])
    manifest["run_dir"] = str(record["run_dir"])
    model, input_mode, checkpoints = _build_benchmark_model(manifest, freeze)
    _load_model(model, checkpoints)
    torch.set_num_threads(int(threads))
    test = windows[str(manifest["protocol"])]["test"]
    outputs: List[Dict[str, object]] = []
    peak_rss = 0
    for batch_size in (1, 128):
        loads = torch.from_numpy(test["loads"][:batch_size])
        exog = torch.from_numpy(test["exog"][:batch_size]) if input_mode == "loads_and_exog" else None
        result = benchmark_model_resources(
            model,
            loads,
            exog,
            warmup=warmup,
            iterations=iterations,
            input_mode=input_mode,
        )
        peak_rss = max(peak_rss, int(result.get("peak_rss_bytes") or 0))
        outputs.append(result)
    batch1, batch128 = outputs
    return {
        "stage": record["stage"],
        "source_group": record["source_group"],
        "protocol": manifest.get("protocol"),
        "model": manifest.get("model"),
        "candidate_id": manifest.get("candidate_id"),
        "seed": manifest.get("seed") if manifest.get("seed") is not None else "deterministic",
        "run_dir": str(record["run_dir"]),
        "input_mode": input_mode,
        "parameter_count": int(batch1["parameter_count"]),
        "macs_per_batch_estimated": int(batch128["macs_per_batch_estimated"]),
        "flops_per_batch_estimated": int(batch128["flops_per_batch_estimated"]),
        "cpu_latency_ms_median": float(batch1["cpu_latency_ms_median"]),
        "cpu_latency_ms_iqr": float(batch1["cpu_latency_ms_iqr"]),
        "cpu_throughput_samples_per_second": float(batch128["cpu_throughput_samples_per_second"]),
        "peak_rss_bytes": peak_rss or None,
        "batch1_latency_ms_median": float(batch1["cpu_latency_ms_median"]),
        "batch128_latency_ms_median": float(batch128["cpu_latency_ms_median"]),
        "benchmark_warmup": int(warmup),
        "benchmark_iterations": int(iterations),
        "status": "passed",
        "error": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Stage 7-R checkpoint resources")
    parser.add_argument("--kitakyushu-data-dir", default="D:/Paper/Kitakyushu dataset")
    parser.add_argument("--freeze-config", default="frame/reports/stage6r_6_kitakyushu/stage6_selected_config.json")
    parser.add_argument("--stage7-3-dir", default="frame/reports/stage7r_3_kitakyushu_formal")
    parser.add_argument("--stage7-4-dir", default="frame/reports/stage7r_4_kitakyushu_formal")
    parser.add_argument("--stage7-5-dir", default="frame/reports/stage7r_5_kitakyushu_formal")
    parser.add_argument("--stage7-stl-dir", default="frame/reports/stage7r_stl_reference_kitakyushu_formal")
    parser.add_argument("--output-csv", default="frame/reports/stage7r_resource_benchmarks.csv")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.threads <= 0:
        raise ValueError("warmup/iterations/threads 参数不合法")
    freeze = _read_json(_resolve(args.freeze_config))
    counts = stage7_6_expected_counts(freeze)
    roots = {
        "7.3": _resolve(args.stage7_3_dir),
        "7.4": _resolve(args.stage7_4_dir),
        "7.5": _resolve(args.stage7_5_dir),
        "7R.STL": _resolve(args.stage7_stl_dir),
    }
    records: List[Dict[str, object]] = []
    for item in build_stage7_6_source_plan(freeze):
        records.extend(
            _iter_runs(
                roots[str(item["stage"])],
                str(item["stage"]),
                str(item["source_group"]),
                int(item["expected_runs"]),
            )
        )
    if args.max_runs is not None:
        if args.max_runs <= 0:
            raise ValueError("max-runs 必须为正整数")
        records = records[: args.max_runs]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "resource",
                    "status": "dry_run",
                    "record_count": len(records),
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "threads": args.threads,
                    "output_csv": str(_resolve(args.output_csv)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    windows = _protocol_windows(_resolve(args.kitakyushu_data_dir))
    output_path = _resolve(args.output_csv)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        try:
            row = _benchmark_one(
                record,
                windows,
                freeze,
                args.warmup,
                args.iterations,
                args.threads,
            )
        except Exception as exc:  # preserve a transparent failure row
            manifest = record["manifest"]
            row = {
                "stage": record["stage"],
                "source_group": record["source_group"],
                "protocol": manifest.get("protocol"),
                "model": manifest.get("model"),
                "candidate_id": manifest.get("candidate_id"),
                "seed": manifest.get("seed") if manifest.get("seed") is not None else "deterministic",
                "run_dir": str(record["run_dir"]),
                "input_mode": manifest.get("input_mode", ""),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(row)
        rows.append(row)
        print(json.dumps({"stage": "resource", "status": row["status"], "indexed": index, "total": len(records), "run_dir": row["run_dir"]}, ensure_ascii=False), flush=True)
    _write_csv(rows, output_path)
    summary = {
        "status": "passed" if not failures else "failed",
        "record_count": len(rows),
        "failed_count": len(failures),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "threads": args.threads,
        "output_csv": str(output_path),
        "expected_formal_runs": sum(int(item["expected_runs"]) for item in build_stage7_6_source_plan(freeze)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
