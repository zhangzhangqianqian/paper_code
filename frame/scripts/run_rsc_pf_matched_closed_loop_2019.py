"""Run the frozen 2019 matched closed-loop diagnostic matrix.

The command is deliberately fail-closed: it accepts only the declared 2019
selection view, never opens 2020/2021 artifacts, and writes each row through a
small atomic directory transaction.  It is suitable for smoke runs and for
the later full 8,709-origin protocol run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.external_v46_data import ExternalV46Normalization, load_external_v46_split
from src.joint_dispatch.formal_v4_4_pilot_executor import _parameters
from src.joint_dispatch.matched_closed_loop import MatchedClosedLoopResult, run_matched_closed_loop, run_perfect_information_mpc_reference
from src.joint_dispatch.matched_closed_loop_providers import build_external_provider, load_frozen_rsc_pf_provider, run_and_verify_rsc_pf_legacy_replay
from src.joint_dispatch.closed_loop_metrics import regime_aware_forecast_metrics


METHOD_ORDER = ("RSC-PF", "iTransformer-PTO", "DecisionFocused-Online", "DigitalTwins-Policy")
EXTERNAL_METHODS = METHOD_ORDER[1:]
SEEDS = (2026, 2027, 2028, 2029, 2030)
EXPECTED_GATE1_RATIO = 1.0353263112927777


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path, root: Path = FRAME_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _forbidden(path: str | Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    parts = {part for part in normalized.split("/") if part}
    return bool(parts.intersection({"2020", "2021", "test", "test_set", "sealed_test", "test-set", "sealed-test"})) or any(token in normalized for token in ("2020_evaluation", "2021_test", "test-set"))


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("matched closed-loop configuration must be a JSON object")
    # Frozen paths are repository-relative by contract, regardless of where
    # the JSON file itself lives.
    validate_config(payload, FRAME_ROOT)
    return payload


def validate_config(config: Mapping[str, Any], base_dir: Path = FRAME_ROOT) -> None:
    if config.get("schema_version") != "rsc-pf-matched-closed-loop-2019-v1":
        raise ValueError("matched closed-loop schema is invalid")
    if int(config.get("selection_year", -1)) != 2019 or int(config.get("evaluation_year", -1)) != 2020:
        raise ValueError("selection/evaluation years are not frozen")
    if tuple(config.get("excluded_years", ())) != (2021,) or config.get("allow_evaluation_access") is not False:
        raise ValueError("evaluation/excluded-year boundary is invalid")
    for key in ("selection_file", "training_file", "normalization_source_file", "rsc_pf_contract", "rsc_pf_run_root", "rsc_pf_pilot_receipt", "rsc_pf_legacy_rollout", "benchmark_path", "capacity_receipt_path", "external_config", "external_source_receipt", "external_output_root"):
        if key not in config:
            raise ValueError(f"configuration is missing {key}")
        resolved = _resolve(config[key], base_dir)
        if _forbidden(resolved):
            raise ValueError(f"{key} points to a forbidden evaluation/test path")
    if tuple(config.get("external_seeds", ())) != SEEDS:
        raise ValueError("external seed matrix is not frozen")
    if int(config.get("rsc_pf_seed", -1)) != 2026:
        raise ValueError("RSC-PF seed is not frozen")
    latency = config.get("latency")
    if not isinstance(latency, Mapping) or tuple(latency.get("method_order", ())) != METHOD_ORDER:
        raise ValueError("latency method order is not frozen")


def _load_pilot_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("authorized_gate1") is not False or payload.get("evaluation_year_accessed") is not False:
        raise PermissionError("v4.6-b pilot is not a closed diagnostic")
    ratio = float(payload.get("measurements", {}).get("electricity_wape_ratio", float("nan")))
    if ratio != EXPECTED_GATE1_RATIO:
        raise ValueError("v4.6-b Gate-1 evidence has changed")
    if "electricity_wape_ratio" not in payload.get("failures", []):
        raise ValueError("v4.6-b pilot receipt does not preserve its failed criterion")
    return payload


def _rsc_normalization(path: Path) -> Any:
    # The formal checkpoint was fitted from float64 source arrays before
    # statistics were cast to float32.  Preserve that order for replay.
    with np.load(path, allow_pickle=False) as payload:
        fields = {name: np.asarray(payload[name], dtype=np.float64) for name in ("load_history", "exog_history", "device_history")}
    means = {name: value.mean(axis=(0, 1)).astype(np.float32) for name, value in fields.items()}
    scales = {name: np.where(value.std(axis=(0, 1)) < 1.0e-6, 1.0, value.std(axis=(0, 1))).astype(np.float32) for name, value in fields.items()}
    return SimpleNamespace(field_mean={"load": means["load_history"], "exog": means["exog_history"], "device": means["device_history"]}, field_scale={"load": scales["load_history"], "exog": scales["exog_history"], "device": scales["device_history"]}, train_years=(2015, 2016, 2017, 2018))


def _hardware(config: Mapping[str, Any]) -> dict[str, Any]:
    latency = config["latency"]
    torch.set_num_threads(int(latency.get("torch_num_threads", 1)))
    try:
        torch.set_num_interop_threads(int(latency.get("torch_num_interop_threads", 1)))
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work;
        # the chosen value is still recorded and audited.
        pass
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    return {"platform": platform.platform(), "python": platform.python_version(), "pytorch": torch.__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__, "device": latency.get("device", "cpu"), "neural_dtype": latency.get("neural_dtype", "float32"), "solver_method": latency.get("solver_method", "highs"), "torch_num_threads": torch.get_num_threads()}


def _write_atomic_row(row_dir: Path, result: MatchedClosedLoopResult, receipt: Mapping[str, Any], *, resume: bool, partial: bool = False) -> None:
    """Atomically write a completed row or an interruption checkpoint.

    A partial checkpoint is a complete, self-describing prefix of the rollout;
    ``--resume`` deterministically replays the requested prefix and promotes
    it to the normal row directory.  This keeps the causal state contract
    auditable without pretending that a solver/model process survived an OS
    interruption.
    """

    target = row_dir.with_name(row_dir.name + ".partial") if partial else row_dir
    if target.is_dir():
        if not resume:
            raise FileExistsError(f"existing row/checkpoint: {target}")
        if not partial and row_dir.is_dir():
            return
        if partial:
            # A repeated interrupted rehearsal may leave the previous prefix
            # in place.  It is safe to replace only this narrowly scoped
            # checkpoint after the caller has explicitly requested --resume.
            for child in target.iterdir():
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    for nested in child.rglob("*"):
                        if nested.is_file() or nested.is_symlink():
                            nested.unlink()
                    for nested in sorted((p for p in child.rglob("*") if p.is_dir()), reverse=True):
                        nested.rmdir()
                    child.rmdir()
            target.rmdir()
    if not partial and target.with_name(target.name + ".partial").is_dir() and resume:
        # Keep the old checkpoint for provenance while the resumed row is
        # written atomically.  It is removed only after promotion succeeds.
        old_partial = target.with_name(target.name + ".partial")
    else:
        old_partial = None
    writing = target.with_name(target.name + ".writing")
    if writing.exists():
        raise FileExistsError(f"incomplete writing directory exists: {writing}")
    writing.mkdir(parents=True)
    np.savez_compressed(writing / "rollout.npz", **dict(result.arrays))
    (writing / "receipt.json").write_text(json.dumps(dict(receipt), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    writing.rename(target)
    if old_partial is not None and old_partial.is_dir():
        # The final row is now durable; remove only the narrowly scoped
        # checkpoint directory that it supersedes.
        for child in old_partial.iterdir():
            child.unlink()
        old_partial.rmdir()


def _row_receipt(result: MatchedClosedLoopResult, provider: Any, *, config_hash: str, method: str, seed: int, smoke: bool, hardware: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "rsc-pf-matched-closed-loop-row-v1", "method_id": method, "seed": int(seed), "rows": int(len(result.arrays["times"])), "metrics": dict(result.metrics), "optimizer_role": str(provider.optimizer_role), "inference_lp_calls": int(result.inference_lp_calls), "reference_lp_calls": int(result.reference_lp_calls), "test_set_accessed": False, "evaluation_year_accessed": False, "config_sha256": config_hash, "smoke": bool(smoke), "hardware": dict(hardware), "provenance": dict(getattr(provider, "provenance", {}))}


def _selection_and_parameters(config: Mapping[str, Any]):
    selection = load_external_v46_split(_resolve(config["selection_file"]), "pilot")
    train = load_external_v46_split(_resolve(config["training_file"]), "train")
    normalization_source = load_external_v46_split(_resolve(config["normalization_source_file"]), "train")
    if len(selection) != 8709:
        raise ValueError(f"frozen 2019 selection must contain 8,709 origins, got {len(selection)}")
    if set(selection.target_times.astype("datetime64[Y]").astype(int) + 1970) != {2019}:
        raise ValueError("selection file contains a year other than 2019")
    benchmark = yaml.safe_load(_resolve(config["benchmark_path"]).read_text(encoding="utf-8"))
    capacity = json.loads(_resolve(config["capacity_receipt_path"]).read_text(encoding="utf-8"))
    return selection, train, _parameters(benchmark, capacity), _rsc_normalization(_resolve(config["normalization_source_file"])), ExternalV46Normalization.fit(normalization_source)


def _provider(method: str, seed: int, config: Mapping[str, Any], parameters: Mapping[str, Any], rsc_norm: Any, ext_norm: Any) -> Any:
    if method == "RSC-PF":
        return load_frozen_rsc_pf_provider(_resolve(config["rsc_pf_run_root"]), _resolve(config["rsc_pf_contract"]), _resolve(config["benchmark_path"]), _resolve(config["capacity_receipt_path"]), normalization=rsc_norm)
    provider_config = {"external_output_root": str(_resolve(config["external_output_root"])), "normalization": ext_norm, "parameters": dict(parameters)}
    return build_external_provider(method, int(seed), provider_config)


def run_matrix(config_path: str | Path, *, methods: tuple[str, ...] = METHOD_ORDER, seeds: tuple[int, ...] | None = None, limit: int | None = None, warmup_origins: int | None = None, stop_after: int | None = None, output_root: str | Path | None = None, resume: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    destination = _resolve(output_root if output_root is not None else config["output_root"])
    smoke = "smoke" in {part.lower() for part in destination.parts}
    if any(value is not None for value in (limit, stop_after)) or warmup_origins is not None:
        if not smoke:
            raise ValueError("limit, stop-after, and warm-up overrides require an output path segment named smoke")
    selection, train, parameters, rsc_norm, ext_norm = _selection_and_parameters(config)
    if limit is not None:
        if int(limit) <= 0 or int(limit) > len(selection):
            raise ValueError("limit is outside the selection range")
        selection = selection.take(np.arange(int(limit), dtype=np.int64))
    if stop_after is not None:
        if int(stop_after) <= 0 or int(stop_after) > len(selection):
            raise ValueError("stop-after is outside the selection range")
        selection = selection.take(np.arange(int(stop_after), dtype=np.int64))
    warmup = int(config.get("warmup_origins", 100) if warmup_origins is None else warmup_origins)
    if warmup < 0 or warmup >= len(selection):
        raise ValueError("warmup_origins is outside the selected run")
    if not smoke and len(selection) != 8709:
        raise ValueError("full execution requires exactly 8,709 origins")
    selected_seeds = tuple(SEEDS if seeds is None else seeds)
    if methods != METHOD_ORDER:
        methods = tuple(methods)
    root = destination.resolve(); root.mkdir(parents=True, exist_ok=True)
    config_hash = _sha256(Path(config_path).resolve())
    hardware = _hardware(config)
    _load_pilot_receipt(_resolve(config["rsc_pf_pilot_receipt"]))
    rows: list[dict[str, Any]] = []
    for method in methods:
        row_seeds = (int(config.get("rsc_pf_seed", 2026)),) if method == "RSC-PF" else selected_seeds
        for seed in row_seeds:
            row_dir = root / "rows" / method.replace("/", "_").replace(" ", "_") / f"seed_{seed}"
            provider = _provider(method, seed, config, parameters, rsc_norm, ext_norm)
            result = run_matched_closed_loop(selection, provider, parameters, method_id=method, seed=seed, warmup_origins=warmup)
            receipt = _row_receipt(result, provider, config_hash=config_hash, method=method, seed=seed, smoke=smoke, hardware=hardware)
            _write_atomic_row(row_dir, result, receipt, resume=resume, partial=stop_after is not None)
            rows.append({"method_id": method, "seed": int(seed), "path": str(row_dir), "optimizer_role": str(provider.optimizer_role), "inference_lp_calls": int(result.inference_lp_calls), "metrics": dict(result.metrics), "reproduction_level": getattr(provider, "reproduction_level", "frozen_local")})
    reference = run_perfect_information_mpc_reference(selection, parameters)
    reference_dir = root / "reference"; reference_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(reference_dir / "perfect_information_mpc.npz", **dict(reference.arrays))
    (reference_dir / "PERFECT_INFORMATION_MPC.json").write_text(json.dumps({"schema": "perfect-information-mpc-reference-v1", "method_id": "Perfect-Information-MPC", "rows": len(selection), "reference_lp_calls": reference.reference_lp_calls, "deployable": False, "test_set_accessed": False, "evaluation_year_accessed": False}, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema": "rsc-pf-matched-closed-loop-manifest-v1", "selection_year": 2019, "evaluation_year": 2020, "excluded_years": [2021], "rows": rows, "reference": {"method_id": "Perfect-Information-MPC", "path": str(reference_dir), "reference_lp_calls": int(reference.reference_lp_calls)}, "gate1_authorized": False, "formal_candidate": False, "gate1_failure": {"criterion": "electricity_wape_ratio", "observed": EXPECTED_GATE1_RATIO, "limit": 1.02}, "test_set_accessed": False, "evaluation_year_accessed": False, "smoke": smoke, "origins": len(selection), "config_sha256": config_hash, "interrupted": bool(stop_after is not None), "cursor": int(len(selection)) if stop_after is not None else None, "thermal_active_scales": {"cooling": float(np.std(train.forecast_target[..., 1][train.forecast_target[..., 1] > 1e-9]) or 1.0), "heating": float(np.std(train.forecast_target[..., 2][train.forecast_target[..., 2] > 1e-9]) or 1.0)}}
    (root / "MATCHED_CLOSED_LOOP_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    if stop_after is not None:
        raise SystemExit(75)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FRAME_ROOT / "configs" / "rsc_pf_matched_closed_loop_2019.json")
    parser.add_argument("--method", choices=METHOD_ORDER)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup-origins", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-rsc-pf-legacy-only", action="store_true")
    args = parser.parse_args()
    methods = METHOD_ORDER if args.all_methods or not args.method else (args.method,)
    seeds = SEEDS if args.all_seeds or args.seed is None else (args.seed,)
    if args.verify_rsc_pf_legacy_only:
        config = load_config(args.config)
        # Replay must use the same frozen numerical environment as the
        # matched runner; otherwise threaded reductions can create small
        # checkpoint-reconstruction drift and invalidate strict hashes.
        _hardware(config)
        selection, train, parameters, rsc_norm, _ = _selection_and_parameters(config)
        provider = load_frozen_rsc_pf_provider(_resolve(config["rsc_pf_run_root"]), _resolve(config["rsc_pf_contract"]), _resolve(config["benchmark_path"]), _resolve(config["capacity_receipt_path"]), normalization=rsc_norm)
        materialized_root = _resolve(config["selection_file"]).parent.parent
        receipt = run_and_verify_rsc_pf_legacy_replay(
            provider,
            materialized_root,
            _resolve(config["rsc_pf_legacy_rollout"]),
            _resolve(config["output_root"]),
            train_data=_resolve(config.get("legacy_replay_train_data", config["training_file"])),
            benchmark=_resolve(config["benchmark_path"]),
            capacity_receipt=_resolve(config["capacity_receipt_path"]),
            atol=float(config.get("replay_atol", 1.0e-5)),
            rtol=float(config.get("replay_rtol", 1.0e-6)),
        )
        print(json.dumps({"status": "PASS", "rows": receipt["rows"], "scheduler_soc_source": receipt["scheduler_soc_source"], "evaluation_year_accessed": False}, ensure_ascii=False))
        return 0
    manifest = run_matrix(args.config, methods=tuple(methods), seeds=tuple(seeds), limit=args.limit, warmup_origins=args.warmup_origins, stop_after=args.stop_after, output_root=args.output_root, resume=args.resume)
    print(json.dumps({"rows": len(manifest["rows"]), "origins": manifest["origins"], "smoke": manifest["smoke"], "test_set_accessed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
