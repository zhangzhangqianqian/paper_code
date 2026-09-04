"""Fail-closed runner for the joint forecast--dispatch experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from typing import Sequence

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.contract import load_joint_training_contract


def _run_smoke(data_root: Path, output_dir: Path, limit: int, contract) -> dict[str, object]:
    import numpy as np
    import torch
    import yaml

    from src.joint_dispatch.data import JointNormalization, load_joint_split
    from src.joint_dispatch.model import JointForecastDispatchModel
    from src.joint_dispatch.rollout import ClosedLoopState, rollout_joint_policy
    from src.joint_dispatch.training import build_joint_optimizer, joint_train_step

    train, normalization, _ = load_joint_split(data_root / "train.npz")
    validation, _, _ = load_joint_split(data_root / "validation.npz")
    if normalization is None:
        normalization = JointNormalization.fit(train)
    train_norm = normalization.transform(train).take(np.arange(min(limit, len(train))))
    validation_norm = normalization.transform(validation).take(np.arange(min(max(1, limit // 2), len(validation))))
    benchmark = yaml.safe_load(contract.path("benchmark_path").read_text(encoding="utf-8"))
    model = JointForecastDispatchModel(
        task_mean=torch.from_numpy(normalization.load_mean),
        task_scale=torch.from_numpy(normalization.load_scale),
        physical_feature_mean=torch.from_numpy(np.concatenate((normalization.load_mean, normalization.scheduler_mean))),
        physical_feature_scale=torch.from_numpy(np.concatenate((normalization.load_scale, normalization.scheduler_scale))),
        decoder_parameters=benchmark["values"],
        dropout=0.0,
    )
    optimizer = build_joint_optimizer(model, variant="joint_from_scratch")
    tensors = {
        "load_history": torch.from_numpy(train_norm.load_history),
        "exog_history": torch.from_numpy(train_norm.exog_history),
        "device_history": torch.from_numpy(train_norm.device_history),
        "device_status": torch.from_numpy(train_norm.device_status),
        "scheduler_context": torch.from_numpy(train.scheduler_context[: len(train_norm)]),
        "previous_chp": torch.from_numpy(train_norm.previous_chp),
        "target_normalized": torch.from_numpy(train_norm.forecast_target),
        "target_physical": torch.from_numpy(train.forecast_target[: len(train_norm)]),
        "teacher_dispatch": torch.from_numpy(train.teacher_dispatch[: len(train_norm)]),
        "oracle_first_step_objective": torch.from_numpy(train.oracle_first_step_objective[: len(train_norm)]),
    }
    before = [parameter.detach().clone() for parameter in model.parameters()]
    records = []
    for epoch in range(3):
        records.append(joint_train_step(model, tensors, benchmark["values"], optimizer, epoch=epoch))
    after = list(model.parameters())
    changed = any(not torch.equal(old, new.detach()) for old, new in zip(before, after))
    latest = records[-1]
    finite = all(bool(torch.isfinite(value).all()) for value in (latest.loss.total, latest.loss.forecast, latest.loss.imitation, latest.loss.regret))
    # The smoke gate also exercises a 48-hour model-generated rollout.  It is
    # intentionally built from the causal train windows only; this is a
    # runtime/state-transition check, not a reported validation or test result.
    closed_loop_48h = False
    closed_loop_reason = "not attempted"
    rollout_outcome_finite = False
    rollout_state_finite = False
    try:
        stream_len = min(51, len(train_norm))
        if stream_len < 51:
            raise ValueError("smoke artifact must contain at least 51 chronological train windows")
        model.eval()
        initial_state = ClosedLoopState(
            soc=torch.from_numpy(train.scheduler_context[:1, 0, -1]).reshape(1, 1),
            previous_chp=torch.from_numpy(train_norm.previous_chp[:1]),
            device_history=torch.from_numpy(train_norm.device_history[:1]),
            device_status=torch.from_numpy(train_norm.device_status[:1]),
            load_history=torch.from_numpy(train_norm.load_history[:1]),
            exog_history=torch.from_numpy(train_norm.exog_history[:1]),
        )
        stream = {
            "loads": torch.from_numpy(train.forecast_target[:stream_len, 0, :]),
            "loads_for_history": torch.from_numpy(train_norm.forecast_target[:stream_len, 0, :]),
            "exog": torch.from_numpy(np.concatenate((
                train_norm.exog_history[1:stream_len + 1, -1, :],
                train_norm.exog_history[-1:, -1, :],
            ), axis=0)[:stream_len]),
            "pv_available": torch.from_numpy(train.scheduler_context[:stream_len, 0, 0]),
            "wt_available": torch.from_numpy(train.scheduler_context[:stream_len, 0, 1]),
            "grid_price": torch.from_numpy(train.scheduler_context[:stream_len, 0, 2]),
            "gas_price": torch.from_numpy(train.scheduler_context[:stream_len, 0, 3]),
            "carbon_price": torch.from_numpy(train.scheduler_context[:stream_len, 0, 4]),
            "device_history_mean": normalization.device_mean,
            "device_history_scale": normalization.device_scale,
        }
        with torch.inference_mode():
            rollout = rollout_joint_policy(model, stream, initial_state, benchmark["values"])
        rollout_outcome_finite = all(
            bool(torch.isfinite(outcome.penalized_objective).all()) for outcome in rollout.outcomes
        )
        rollout_state_finite = all(
            bool(torch.isfinite(state.device_history).all()) for state in rollout.states
        )
        closed_loop_48h = len(rollout.outcomes) == 48 and rollout_outcome_finite and rollout_state_finite
        closed_loop_reason = "48 causal train-window steps; smoke-only state-transition check"
    except (RuntimeError, ValueError) as exc:
        closed_loop_reason = f"rollout failed: {exc}"
    receipt = {
        "stage": "smoke", "formal": False, "complete": bool(
            finite and changed and latest.gradient_audit.forecaster_nonzero
            and latest.gradient_audit.scheduler_nonzero and closed_loop_48h
        ),
        "train_windows": len(train_norm), "validation_windows": len(validation_norm), "epochs": 3,
        "dispatch_shape": [len(train_norm), 4, 21], "online_exact_lp_calls": 0,
        "finite_losses": finite, "parameters_changed": changed,
        "forecaster_gradient_nonzero": latest.gradient_audit.forecaster_nonzero,
        "scheduler_gradient_nonzero": latest.gradient_audit.scheduler_nonzero,
        "decision_only_forecaster_gradient": latest.gradient_audit.forecaster_norm_from_decision_only,
        "closed_loop_48h": closed_loop_48h,
        "closed_loop_reason": closed_loop_reason,
        "rollout_outcome_finite": rollout_outcome_finite,
        "rollout_state_finite": rollout_state_finite,
        "rollout_steps": len(rollout.outcomes) if 'rollout' in locals() else 0,
    }
    (output_dir / "smoke_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def _model_from_normalization(normalization, benchmark: dict[str, object]):
    import numpy as np
    import torch
    from src.joint_dispatch.model import JointForecastDispatchModel

    return JointForecastDispatchModel(
        task_mean=torch.from_numpy(normalization.load_mean),
        task_scale=torch.from_numpy(normalization.load_scale),
        physical_feature_mean=torch.from_numpy(np.concatenate((normalization.load_mean, normalization.scheduler_mean))),
        physical_feature_scale=torch.from_numpy(np.concatenate((normalization.load_scale, normalization.scheduler_scale))),
        decoder_parameters=benchmark["values"],
        dropout=0.0,
    )


def _batch_tensors(raw_split, normalized_split, indices, torch):
    index = indices
    return {
        "load_history": torch.from_numpy(normalized_split.load_history[index]),
        "exog_history": torch.from_numpy(normalized_split.exog_history[index]),
        "device_history": torch.from_numpy(normalized_split.device_history[index]),
        "device_status": torch.from_numpy(normalized_split.device_status[index]),
        "scheduler_context": torch.from_numpy(raw_split.scheduler_context[index]),
        "previous_chp": torch.from_numpy(raw_split.previous_chp[index]),
        "target_normalized": torch.from_numpy(normalized_split.forecast_target[index]),
        "target_physical": torch.from_numpy(raw_split.forecast_target[index]),
        "teacher_dispatch": torch.from_numpy(raw_split.teacher_dispatch[index]),
        "oracle_first_step_objective": torch.from_numpy(raw_split.oracle_first_step_objective[index]),
    }


def _collect_rollin_history(model, raw_split, normalized_split, normalization, benchmark):
    """Collect one causal model-policy history pass over train origins.

    The rollout receives realized loads only for post-action state updates and
    stores normalized histories for the next model call.  The final three
    origins cannot be executed because their four-hour stream is exhausted;
    those rows retain their causal LP histories so the timestamp-aligned
    50/50 mixture remains complete and deterministic.
    """
    import numpy as np
    import torch
    from src.joint_dispatch.data import JointWindowSplit
    from src.joint_dispatch.rollout import ClosedLoopState, rollout_joint_policy

    total = len(raw_split)
    if total < 4:
        raise ValueError("train split must contain at least four windows for roll-in")
    state = ClosedLoopState(
        soc=torch.from_numpy(raw_split.scheduler_context[:1, 0, -1]).reshape(1, 1),
        previous_chp=torch.from_numpy(normalized_split.previous_chp[:1]),
        device_history=torch.from_numpy(normalized_split.device_history[:1]),
        device_status=torch.from_numpy(normalized_split.device_status[:1]),
        load_history=torch.from_numpy(normalized_split.load_history[:1]),
        exog_history=torch.from_numpy(normalized_split.exog_history[:1]),
    )
    loads = np.array(normalized_split.load_history, copy=True)
    exogs = np.array(normalized_split.exog_history, copy=True)
    devices = np.array(normalized_split.device_history, copy=True)
    statuses = np.array(normalized_split.device_status, copy=True)
    previous = np.array(raw_split.previous_chp, copy=True)
    callback_index = [0]

    def capture(snapshot):
        index = callback_index[0]
        if index >= total:
            return
        loads[index] = snapshot.load_history[0].detach().cpu().numpy()
        exogs[index] = snapshot.exog_history[0].detach().cpu().numpy()
        devices[index] = snapshot.device_history[0].detach().cpu().numpy()
        statuses[index] = snapshot.device_status[0].detach().cpu().numpy()
        previous[index] = snapshot.previous_chp[0].detach().cpu().numpy()
        callback_index[0] += 1
    exog_stream = np.concatenate((
        normalized_split.exog_history[1:total + 1, -1, :],
        normalized_split.exog_history[-1:, -1, :],
    ), axis=0)[:total]
    inputs = {
        "loads": torch.from_numpy(raw_split.forecast_target[:, 0, :]),
        "loads_for_history": torch.from_numpy(normalized_split.forecast_target[:, 0, :]),
        "exog": torch.from_numpy(exog_stream),
        "pv_available": torch.from_numpy(raw_split.scheduler_context[:, 0, 0]),
        "wt_available": torch.from_numpy(raw_split.scheduler_context[:, 0, 1]),
        "grid_price": torch.from_numpy(raw_split.scheduler_context[:, 0, 2]),
        "gas_price": torch.from_numpy(raw_split.scheduler_context[:, 0, 3]),
        "carbon_price": torch.from_numpy(raw_split.scheduler_context[:, 0, 4]),
        "device_history_mean": normalization.device_mean,
        "device_history_scale": normalization.device_scale,
    }
    model.eval()
    with torch.inference_mode():
        rollout = rollout_joint_policy(
            model, inputs, state, benchmark["values"],
            retain_states=False, state_callback=capture,
        )
    executed = callback_index[0]
    # Convert normalized histories back to raw units for the immutable window
    # container; the runner applies the training-only normalization again
    # before feeding the model.
    raw_loads = loads * normalization.load_scale.reshape(1, 1, -1) + normalization.load_mean.reshape(1, 1, -1)
    raw_exogs = exogs * normalization.exog_scale.reshape(1, 1, -1) + normalization.exog_mean.reshape(1, 1, -1)
    raw_devices = devices * normalization.device_scale.reshape(1, 1, -1) + normalization.device_mean.reshape(1, 1, -1)
    return JointWindowSplit(
        load_history=raw_loads.astype(np.float32), exog_history=raw_exogs.astype(np.float32),
        device_history=raw_devices.astype(np.float32), device_status=statuses.astype(np.float32),
        forecast_target=np.array(raw_split.forecast_target, copy=True),
        scheduler_context=np.array(raw_split.scheduler_context, copy=True),
        previous_chp=previous.astype(np.float32),
        teacher_dispatch=np.array(raw_split.teacher_dispatch, copy=True),
        oracle_first_step_objective=np.array(raw_split.oracle_first_step_objective, copy=True),
        target_times=np.array(raw_split.target_times, copy=True), split="train",
        history_source="joint_policy_rollin",
    ), executed


def _closed_loop_metrics(model, raw_split, normalized_split, benchmark: dict[str, object], *, normalization=None, max_hours: int | None = None) -> dict[str, object]:
    import numpy as np
    import torch
    from src.joint_dispatch.evaluation import evaluate_forecast, realized_dispatch_summary
    from src.joint_dispatch.rollout import ClosedLoopState, rollout_joint_policy

    total = len(raw_split)
    # Each window origin is one hour later than the previous one.  The last
    # three windows do not have a complete four-hour forecast and are excluded
    # by the rollout contract.
    if max_hours is not None:
        total = min(total, int(max_hours) + 3)
    if total < 4:
        raise ValueError("validation split must contain at least four windows")
    state = ClosedLoopState(
        soc=torch.from_numpy(raw_split.scheduler_context[:1, 0, -1]).reshape(1, 1),
        previous_chp=torch.from_numpy(normalized_split.previous_chp[:1]),
        device_history=torch.from_numpy(normalized_split.device_history[:1]),
        device_status=torch.from_numpy(normalized_split.device_status[:1]),
        load_history=torch.from_numpy(normalized_split.load_history[:1]),
        exog_history=torch.from_numpy(normalized_split.exog_history[:1]),
    )
    inputs = {
        "loads": torch.from_numpy(raw_split.forecast_target[:total, 0, :]),
        "loads_for_history": torch.from_numpy(normalized_split.forecast_target[:total, 0, :]),
        "exog": torch.from_numpy(np.concatenate((
            normalized_split.exog_history[1:total + 1, -1, :],
            normalized_split.exog_history[-1:, -1, :],
        ), axis=0)[:total]),
        "device_history_mean": None if normalization is None else torch.from_numpy(normalization.device_mean),
        "device_history_scale": None if normalization is None else torch.from_numpy(normalization.device_scale),
        "pv_available": torch.from_numpy(raw_split.scheduler_context[:total, 0, 0]),
        "wt_available": torch.from_numpy(raw_split.scheduler_context[:total, 0, 1]),
        "grid_price": torch.from_numpy(raw_split.scheduler_context[:total, 0, 2]),
        "gas_price": torch.from_numpy(raw_split.scheduler_context[:total, 0, 3]),
        "carbon_price": torch.from_numpy(raw_split.scheduler_context[:total, 0, 4]),
    }
    model.eval()
    with torch.inference_mode():
        rollout = rollout_joint_policy(model, inputs, state, benchmark["values"])
    horizon_windows = len(rollout.outcomes)
    if horizon_windows <= 0:
        raise ValueError("closed-loop validation produced no steps")
    forecast_prediction = torch.cat(list(rollout.forecasts), dim=0).cpu().numpy()
    forecast_target = raw_split.forecast_target[:horizon_windows]
    dispatch = torch.cat([outcome.realized_dispatch.unsqueeze(1) for outcome in rollout.outcomes], dim=0).cpu().numpy()
    demand = raw_split.forecast_target[:horizon_windows, :1, :3]
    dispatch_summary = realized_dispatch_summary(
        dispatch,
        demand,
        grid_price=raw_split.scheduler_context[:horizon_windows, :1, 2],
        gas_price=raw_split.scheduler_context[:horizon_windows, :1, 3],
        carbon_price=raw_split.scheduler_context[:horizon_windows, :1, 4],
        grid_emission_factor=float(benchmark["values"].get("grid_emission_factor", 0.5)),
        gas_emission_factor=float(benchmark["values"].get("gas_emission_factor", 0.25)),
        unserved_penalty=float(benchmark["values"]["unserved_penalty"]),
    )
    oracle = torch.from_numpy(raw_split.oracle_first_step_objective[:horizon_windows])
    penalized = torch.stack([outcome.penalized_objective.reshape(-1)[0] for outcome in rollout.outcomes])
    regret = float(((penalized - oracle) / oracle.abs().clamp_min(1.0)).mean().item())
    regret_series = ((penalized - oracle) / oracle.abs().clamp_min(1.0)).detach().cpu().numpy().astype(float).tolist()
    return {
        "steps": horizon_windows,
        "forecast_mae": evaluate_forecast(forecast_prediction, forecast_target).mae.tolist(),
        "forecast_rmse": evaluate_forecast(forecast_prediction, forecast_target).rmse.tolist(),
        "forecast_wape": evaluate_forecast(forecast_prediction, forecast_target).wape.tolist(),
        "decision_regret": regret,
        "decision_regret_series": regret_series,
        "closed_loop_summary": dict(dispatch_summary),
        "online_exact_lp_calls": 0,
        "state_finite": all(bool(torch.isfinite(item.device_history).all()) for item in rollout.states),
        "forecast_finite": bool(np.isfinite(forecast_prediction).all()),
    }


def _frozen_pto_metrics(raw_split, benchmark: dict[str, object]) -> dict[str, object]:
    import numpy as np
    import torch
    from src.joint_dispatch.evaluation import evaluate_forecast, realized_dispatch_summary
    from src.joint_dispatch.rollout import apply_first_step_recourse

    horizon_windows = len(raw_split)
    prediction = raw_split.load_history[:, :4, :]
    target = raw_split.forecast_target
    dispatch = raw_split.teacher_dispatch
    summary = realized_dispatch_summary(
        dispatch,
        target[:, :, :3],
        grid_price=raw_split.scheduler_context[:, :, 2],
        gas_price=raw_split.scheduler_context[:, :, 3],
        carbon_price=raw_split.scheduler_context[:, :, 4],
        grid_emission_factor=float(benchmark["values"].get("grid_emission_factor", 0.5)),
        gas_emission_factor=float(benchmark["values"].get("gas_emission_factor", 0.25)),
        unserved_penalty=float(benchmark["values"]["unserved_penalty"]),
    )
    first_outcome = apply_first_step_recourse(
        torch.from_numpy(dispatch[:, 0, :]),
        torch.from_numpy(target[:, 0, :3]),
        torch.from_numpy(raw_split.scheduler_context[:, 0, 0]),
        torch.from_numpy(raw_split.scheduler_context[:, 0, 1]),
        benchmark["values"],
        grid_price=torch.from_numpy(raw_split.scheduler_context[:, 0, 2]),
        gas_price=torch.from_numpy(raw_split.scheduler_context[:, 0, 3]),
        carbon_price=torch.from_numpy(raw_split.scheduler_context[:, 0, 4]),
    )
    oracle = torch.from_numpy(raw_split.oracle_first_step_objective)
    relative_regret = float(
        ((first_outcome.penalized_objective - oracle) / oracle.abs().clamp_min(1.0)).mean().item()
    )
    regret_series = ((first_outcome.penalized_objective - oracle) / oracle.abs().clamp_min(1.0)).detach().cpu().numpy().astype(float).tolist()
    # The frozen PTO labels already carry the causal SOC/CHP state from the
    # sequential offline teacher.  The first-hour objective is therefore the
    # comparable modular baseline for each validation origin.
    return {
        "steps": horizon_windows,
        "forecast_mae": evaluate_forecast(prediction, target).mae.tolist(),
        "forecast_rmse": evaluate_forecast(prediction, target).rmse.tolist(),
        "forecast_wape": evaluate_forecast(prediction, target).wape.tolist(),
        "decision_regret": relative_regret,
        "decision_regret_series": regret_series,
        "closed_loop_summary": dict(summary),
        "online_exact_lp_calls": 0,
        "state_finite": bool(np.isfinite(dispatch).all()),
        "forecast_finite": bool(np.isfinite(prediction).all()),
        "baseline_definition": "seasonal-naive-24 modular forecast followed by causal offline LP teacher",
    }


def _run_validation_candidate(
    data_root: Path,
    output_dir: Path,
    contract,
    *,
    variant: str,
    seed: int,
    epochs: int,
    batch_size: int,
    max_validation_hours: int | None,
    warm_start_root: Path | None,
    resume: bool,
) -> dict[str, object]:
    import json
    import random
    import numpy as np
    import torch
    import yaml
    from src.joint_dispatch.data import load_joint_split
    from src.joint_dispatch.training import build_joint_optimizer, joint_train_step, save_joint_checkpoint, mix_history_windows, rollin_refresh_epoch

    if variant not in {"frozen_pto", "joint_from_scratch", "warm_started_joint"}:
        raise SystemExit("unknown validation variant")
    if variant == "warm_started_joint" and warm_start_root is None:
        raise SystemExit("warm_started_joint requires --warm-start-root")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    train, normalization, _ = load_joint_split(data_root / "train.npz")
    validation, _, _ = load_joint_split(data_root / "validation.npz")
    if normalization is None:
        raise ValueError("train artifact must contain train-only normalization")
    benchmark = yaml.safe_load(contract.path("benchmark_path").read_text(encoding="utf-8"))
    normalized_train = normalization.transform(train)
    normalized_validation = normalization.transform(validation)
    candidate_dir = output_dir / variant / f"seed_{seed}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = candidate_dir / "validation_receipt.json"
    if resume and receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        # Re-run a warm-start candidate if an earlier revision produced a
        # receipt with zero loaded tensors (legacy checkpoints use
        # ``model_state_dict`` rather than ``model``).
        warm_start_loaded = existing.get("warm_start", {}).get("loaded", []) if isinstance(existing.get("warm_start"), dict) else []
        needs_rollin_refresh = variant in {"joint_from_scratch", "warm_started_joint"} and not bool(existing.get("rollin_refresh", {}).get("implemented", False))
        needs_regret_series = "decision_regret_series" not in existing.get("metrics", {})
        if not (variant == "warm_started_joint" and not warm_start_loaded) and not needs_rollin_refresh and not needs_regret_series:
            return existing
    if not resume and any(candidate_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite validation candidate: {candidate_dir}")

    if variant == "frozen_pto":
        metrics = _frozen_pto_metrics(validation, benchmark)
        receipt = {
            "schema_version": contract.schema_version, "stage": "validation", "formal": True,
            "complete": True, "variant": variant, "seed": seed, "epochs": 0,
            "data_root": str(data_root), "test_set_accessed": False,
            "gradient_audit": None, "metrics": metrics,
        }
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        return receipt

    model = _model_from_normalization(normalization, benchmark)
    warm_start_receipt = None
    if variant == "warm_started_joint":
        from src.joint_dispatch.training import load_allowed_warm_start

        source = warm_start_root / f"seed_{seed}" / "best_model.pt"
        if not source.exists():
            raise SystemExit(f"warm-start checkpoint is missing: {source}")
        warm_start_receipt = load_allowed_warm_start(
            model,
            source,
            variant=variant,
            forecast_prefixes=tuple(contract.warm_start_key_prefixes["forecast"]),
            scheduler_prefixes=tuple(contract.warm_start_key_prefixes["scheduler"]),
        )
    optimizer = build_joint_optimizer(model, variant=variant)
    rng = np.random.default_rng(seed)
    last_result = None
    refresh_epoch = rollin_refresh_epoch(epochs)
    rollin_history = None
    rollin_executed = 0
    for epoch in range(epochs):
        if epoch == refresh_epoch and rollin_history is None:
            rollin_history, rollin_executed = _collect_rollin_history(
                model, train, normalized_train, normalization, benchmark,
            )
        epoch_split = train if rollin_history is None else mix_history_windows(
            train, rollin_history,
            model_history_fraction=0.5,
            seed=seed + epoch,
        )
        epoch_normalized = normalized_train if epoch_split is train else normalization.transform(epoch_split)
        model.train()
        permutation = rng.permutation(len(epoch_split))
        for start in range(0, len(epoch_split), batch_size):
            indices = permutation[start : start + batch_size]
            result = joint_train_step(
                model,
                _batch_tensors(epoch_split, epoch_normalized, indices, torch),
                benchmark["values"],
                optimizer,
                epoch=epoch,
            )
            last_result = result
        if not last_result.gradient_audit.forecaster_nonzero or not last_result.gradient_audit.scheduler_nonzero:
            raise RuntimeError(f"joint seed {seed} lost gradient coupling at epoch {epoch}")
    if last_result is None:
        raise RuntimeError("joint candidate completed no optimizer steps")
    checkpoint = candidate_dir / "last_checkpoint.pt"
    save_joint_checkpoint(
        checkpoint, model, optimizer, epoch=epochs - 1, variant=variant, seed=seed,
        best_validation_score=None,
        gradient_audit=last_result.gradient_audit,
        loss_weights={
            "forecast": last_result.weights.forecast,
            "imitation": last_result.weights.imitation,
            "decision": last_result.weights.decision,
        },
    )
    metrics = _closed_loop_metrics(model, validation, normalized_validation, benchmark, normalization=normalization, max_hours=max_validation_hours)
    receipt = {
        "schema_version": contract.schema_version, "stage": "validation", "formal": True,
        "complete": bool(metrics["state_finite"] and metrics["forecast_finite"] and metrics["online_exact_lp_calls"] == 0),
        "variant": variant, "seed": seed, "epochs": epochs, "batch_size": batch_size,
        "data_root": str(data_root), "test_set_accessed": False,
        "gradient_audit": {
            "forecaster_norm_from_total": last_result.gradient_audit.forecaster_norm_from_total,
            "scheduler_norm_from_total": last_result.gradient_audit.scheduler_norm_from_total,
            "forecaster_norm_from_decision_only": last_result.gradient_audit.forecaster_norm_from_decision_only,
            "forecaster_nonzero": last_result.gradient_audit.forecaster_nonzero,
            "scheduler_nonzero": last_result.gradient_audit.scheduler_nonzero,
        },
        "warm_start": None if warm_start_receipt is None else {
            "loaded": list(warm_start_receipt.loaded),
            "skipped": list(warm_start_receipt.skipped),
            "rejected": list(warm_start_receipt.rejected),
            "source_sha256": warm_start_receipt.source_sha256,
        },
        "rollin_refresh": {
            "implemented": True,
            "refresh_epoch_zero_based": int(refresh_epoch),
            "executed_steps": int(rollin_executed),
            "model_history_fraction": 0.5,
            "training_only": True,
            "history_source": "joint_policy_rollin",
        },
        "checkpoint": str(checkpoint), "metrics": metrics,
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def _freeze_validation_selection(validation_dir: Path, selection_dir: Path, contract) -> dict[str, object]:
    """Compare validation-only receipts and freeze one immutable candidate."""

    import json
    import math
    import numpy as np
    from src.joint_dispatch.evaluation import paired_moving_block_bootstrap

    expected_seeds = tuple(int(seed) for seed in contract.seeds)
    receipts: dict[tuple[str, int], dict[str, object]] = {}
    variants = ["frozen_pto", "joint_from_scratch"]
    warm_dir = validation_dir / "warm_started_joint"
    warm_paths = [warm_dir / f"seed_{seed}" / "validation_receipt.json" for seed in expected_seeds]
    warm_activated = all(path.exists() for path in warm_paths)
    if warm_activated:
        variants.append("warm_started_joint")
    for variant in variants:
        for seed in expected_seeds:
            path = validation_dir / variant / f"seed_{seed}" / "validation_receipt.json"
            if not path.exists():
                raise SystemExit(f"selection requires validation receipt: {path}")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if not bool(receipt.get("complete", False)) or bool(receipt.get("test_set_accessed", True)):
                raise SystemExit(f"selection found incomplete or test-contaminated receipt: {path}")
            receipts[(variant, seed)] = receipt

    baseline_wape = np.nanmean(
        np.asarray([receipts[("frozen_pto", seed)]["metrics"]["forecast_wape"] for seed in expected_seeds], dtype=float),
        axis=(0, 1),
    )
    candidate_rows = []
    passing_by_variant: dict[str, list[dict[str, object]]] = {}
    for variant in variants[1:]:
        rows = []
        for seed in expected_seeds:
            receipt = receipts[(variant, seed)]
            wape = np.nanmean(np.asarray(receipt["metrics"]["forecast_wape"], dtype=float), axis=0)
            rigid = wape[:3]
            baseline_rigid = baseline_wape[:3]
            aggregate_gate = bool(np.isfinite(rigid).all() and np.mean(rigid) <= 1.02 * np.mean(baseline_rigid))
            task_gate = bool(np.isfinite(rigid).all() and np.all(rigid <= 1.05 * baseline_rigid))
            gradient_gate = bool(
                receipt.get("gradient_audit", {}).get("forecaster_norm_from_decision_only", 0.0) > 0.0
                and receipt.get("gradient_audit", {}).get("scheduler_norm_from_total", 0.0) > 0.0
            )
            warm_load_gate = variant != "warm_started_joint" or bool(receipt.get("warm_start", {}).get("loaded", []))
            row = {
                "variant": variant, "seed": seed,
                "aggregate_rigid_wape": float(np.mean(rigid)) if np.isfinite(rigid).all() else math.inf,
                "rigid_task_wape": rigid.tolist(),
                "forecast_gate_aggregate": aggregate_gate,
                "forecast_gate_per_task": task_gate,
                "gradient_gate": gradient_gate,
                "warm_start_load_gate": warm_load_gate,
                "decision_regret": float(receipt["metrics"]["decision_regret"]),
                "checkpoint": receipt.get("checkpoint"),
            }
            row["all_gates"] = bool(receipt.get("complete", False) and aggregate_gate and task_gate and gradient_gate and warm_load_gate)
            rows.append(row)
            candidate_rows.append(row)
        passing_by_variant[variant] = [row for row in rows if row["all_gates"] and np.isfinite(row["decision_regret"])]

    warm_bootstrap = None
    if warm_activated:
        try:
            warm_series = np.asarray([receipts[("warm_started_joint", seed)]["metrics"]["decision_regret_series"] for seed in expected_seeds], dtype=float)
            scratch_series = np.asarray([receipts[("joint_from_scratch", seed)]["metrics"]["decision_regret_series"] for seed in expected_seeds], dtype=float)
            if warm_series.shape == scratch_series.shape and warm_series.ndim == 2 and warm_series.shape[1] >= 168:
                warm_bootstrap = paired_moving_block_bootstrap(warm_series, scratch_series, block_hours=168, replicates=2000, seed=2026)
        except (KeyError, TypeError, ValueError):
            warm_bootstrap = None

    selected = None
    selected_variant = "frozen_pto"
    selection_reason = "no joint candidate passed all predeclared validation gates"
    scratch_passing = passing_by_variant.get("joint_from_scratch", [])
    warm_passing = passing_by_variant.get("warm_started_joint", [])
    if scratch_passing:
        selected = min(scratch_passing, key=lambda row: (row["decision_regret"], row["aggregate_rigid_wape"], row["seed"]))
        selected_variant = "joint_from_scratch"
        selection_reason = "joint-from-scratch candidate passed validation forecast, gradient, and finite-state gates"
    if warm_activated and warm_passing:
        warm_ci_pass = bool(
            warm_bootstrap is not None
            and float(warm_bootstrap["ci_upper"]) < 0.0
        )
        if warm_ci_pass:
            selected = min(warm_passing, key=lambda row: (row["decision_regret"], row["aggregate_rigid_wape"], row["seed"]))
            selected_variant = "warm_started_joint"
            selection_reason = "warm-start candidate passed all gates and paired 168-hour regret bootstrap CI was below zero"
        elif not scratch_passing:
            selected = None
            selection_reason = "neither joint candidate passed the full gate; retaining frozen PTO baseline"
    if selected is None:
        selected = {
            "variant": "frozen_pto", "seed": expected_seeds[0],
            "decision_regret": float(np.mean([receipts[("frozen_pto", seed)]["metrics"]["decision_regret"] for seed in expected_seeds])),
            "aggregate_rigid_wape": float(np.mean(baseline_wape[:3])),
            "rigid_task_wape": baseline_wape[:3].tolist(),
            "all_gates": True,
            "checkpoint": None,
        }
        selected_variant = "frozen_pto"
    selection = {
        "schema_version": contract.schema_version,
        "stage": "select",
        "formal": True,
        "complete": True,
        "test_set_accessed": False,
        "validation_seeds": list(expected_seeds),
        "baseline_variant": "frozen_pto",
        "baseline_definition": receipts[("frozen_pto", expected_seeds[0])]["metrics"].get("baseline_definition"),
        "baseline_rigid_task_wape": baseline_wape[:3].tolist(),
        "candidate_rows": candidate_rows,
        "selected_variant": selected_variant,
        "selected_seed": int(selected["seed"]),
        "selected_checkpoint": selected.get("checkpoint"),
        "selection_reason": selection_reason,
        "selection_rule": "minimum closed-loop decision regret among gate-passing validation checkpoints; tie-break by rigid-task WAPE then seed",
        "warm_start_status": "activated_due_to_joint_forecast_gate_failure" if warm_activated else "not_activated_by_protocol",
        "warm_start_bootstrap": warm_bootstrap,
    }
    selection_dir.mkdir(parents=True, exist_ok=True)
    path = selection_dir / "selection_receipt.json"
    if path.exists():
        raise SystemExit(f"selection receipt already exists and is immutable: {path}")
    path.write_text(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return selection


def _run_sealed_test(data_root: Path, output_dir: Path, selection: dict[str, object], contract, *, resume: bool) -> dict[str, object]:
    """Evaluate exactly the frozen validation choice on the sealed test split."""

    import json
    import numpy as np
    import torch
    import yaml
    from src.joint_dispatch.data import load_joint_split

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "test_receipt.json"
    if path.exists():
        if resume:
            return json.loads(path.read_text(encoding="utf-8"))
        raise SystemExit(f"test receipt already exists and is immutable: {path}")
    test, _, _ = load_joint_split(data_root / "test.npz")
    train, normalization, _ = load_joint_split(data_root / "train.npz")
    if normalization is None:
        raise ValueError("sealed test evaluation requires the persisted train normalization")
    benchmark = yaml.safe_load(contract.path("benchmark_path").read_text(encoding="utf-8"))
    selected_variant = str(selection.get("selected_variant"))
    selected_checkpoint = selection.get("selected_checkpoint")
    if selected_variant == "frozen_pto":
        metrics = _frozen_pto_metrics(test, benchmark)
        checkpoint = None
    elif selected_variant in {"joint_from_scratch", "warm_started_joint"}:
        if not selected_checkpoint:
            raise SystemExit("selection names a trainable variant without a checkpoint")
        checkpoint_path = Path(str(selected_checkpoint))
        if not checkpoint_path.exists():
            raise SystemExit(f"selected checkpoint is missing: {checkpoint_path}")
        model = _model_from_normalization(normalization, benchmark)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("model_state_dict", payload))
        model.load_state_dict(state, strict=True)
        metrics = _closed_loop_metrics(
            model, test, normalization.transform(test), benchmark, normalization=normalization,
        )
        checkpoint = str(checkpoint_path.resolve())
    else:
        raise SystemExit(f"unknown selected variant: {selected_variant}")
    receipt = {
        "schema_version": contract.schema_version,
        "stage": "test",
        "formal": True,
        "complete": bool(metrics.get("state_finite", False) and metrics.get("forecast_finite", False) and metrics.get("online_exact_lp_calls", 1) == 0),
        "selected_variant": selected_variant,
        "selected_seed": selection.get("selected_seed"),
        "selected_checkpoint": checkpoint,
        "data_root": str(data_root.resolve()),
        "test_set_accessed": True,
        "online_exact_lp_calls": 0,
        "metrics": metrics,
    }
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_contract_v1.json")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=("dry-run", "smoke", "validation", "select", "test"), required=True)
    parser.add_argument("--variant", choices=("joint_from_scratch", "warm_started_joint", "frozen_pto"), default="joint_from_scratch")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--max-validation-hours", type=int, default=None)
    parser.add_argument("--warm-start-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def _artifact_exists(root: Path, names: Sequence[str]) -> list[str]:
    return [name for name in names if not (root / name).exists()]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = load_joint_training_contract(args.contract)
    data_root = Path(args.data_root) if args.data_root is not None else contract.path("output_root") / "data"
    output_dir = Path(args.output_dir) if args.output_dir is not None else contract.path("output_root") / args.stage
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": contract.schema_version,
        "stage": args.stage,
        "variant": args.variant,
        "seed": int(args.seed),
        "formal": False,
        "complete": False,
        "online_exact_lp_calls": 0,
        "python_version": platform.python_version(),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
    }
    if args.stage == "dry-run":
        receipt["path_checks"] = {
            "contract": str(args.contract.resolve()),
            "data_root_exists": data_root.exists(),
            "output_root_exists": output_dir.exists(),
        }
        (output_dir / "dry_run_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "smoke":
        if args.smoke_limit is None or args.smoke_limit <= 0:
            raise SystemExit("smoke stage requires a positive --smoke-limit")
        receipt["smoke_limit"] = int(args.smoke_limit)
        missing = _artifact_exists(data_root, ("train.npz", "validation.npz"))
        if missing:
            raise SystemExit(f"smoke data artifacts are missing: {missing}; run the gated data builder first")
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise SystemExit("smoke stage requires optional PyTorch; no experiment was started") from exc
        resource = output_dir.parent / "resource_benchmark" / "resource_benchmark_receipt.json"
        if not resource.exists():
            raise SystemExit("smoke execution requires the passing 500-solve resource receipt")
        receipt = _run_smoke(data_root, output_dir, int(args.smoke_limit), contract)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0 if receipt["complete"] else 2
    if args.stage == "validation":
        if (data_root / "test.npz").exists():
            raise SystemExit("validation stage must not read the sealed test split")
        missing = _artifact_exists(data_root, ("train.npz", "validation.npz"))
        if missing:
            raise SystemExit(f"validation data artifacts are missing: {missing}")
        resource = output_dir.parent / "resource_benchmark" / "resource_benchmark_receipt.json"
        smoke = output_dir.parent / "smoke" / "smoke_receipt.json"
        if not resource.exists() or not smoke.exists():
            raise SystemExit("validation execution requires passing resource and smoke receipts")
        smoke_payload = json.loads(smoke.read_text(encoding="utf-8"))
        if not bool(smoke_payload.get("complete", False)):
            raise SystemExit("validation execution requires a complete smoke receipt")
        seeds = tuple(contract.seeds) if args.all_seeds else (int(args.seed),)
        receipts = []
        for seed in seeds:
            receipts.append(_run_validation_candidate(
                data_root,
                output_dir,
                contract,
                variant=args.variant,
                seed=seed,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                max_validation_hours=args.max_validation_hours,
                warm_start_root=args.warm_start_root,
                resume=bool(args.resume),
            ))
        manifest = {
            "schema_version": contract.schema_version,
            "stage": "validation",
            "formal": True,
            "complete": all(bool(item.get("complete", False)) for item in receipts),
            "variant": args.variant,
            "seeds": list(seeds),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "test_set_accessed": False,
            "candidates": [
                {"variant": item.get("variant"), "seed": item.get("seed"), "complete": item.get("complete"), "metrics": item.get("metrics")}
                for item in receipts
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{args.variant}_validation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0 if manifest["complete"] else 2
    selection = output_dir.parent / "selection" / "selection_receipt.json"
    if args.stage == "select":
        missing = _artifact_exists(output_dir.parent / "validation", ("validation_receipt.json",))
        # Candidate receipts are stored below validation/<variant>/seed_*;
        # retain the older check only for compatibility with external callers.
        if missing and not (output_dir.parent / "validation").exists():
            raise SystemExit(f"selection requires complete validation receipt: {missing}")
        if selection.exists():
            raise SystemExit(f"selection receipt already exists and is immutable: {selection}")
        frozen = _freeze_validation_selection(output_dir.parent / "validation", output_dir, contract)
        print(json.dumps(frozen, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "test":
        if not selection.exists():
            raise SystemExit(f"test stage requires frozen selection receipt: {selection}")
        test_path = data_root / "test.npz"
        if not test_path.exists():
            raise SystemExit(f"test stage requires sealed test data: {test_path}; build it after selection")
        selection_payload = json.loads(selection.read_text(encoding="utf-8"))
        if not bool(selection_payload.get("complete", False)) or bool(selection_payload.get("test_set_accessed", True)):
            raise SystemExit("test stage requires a complete uncontaminated selection receipt")
        test_receipt = _run_sealed_test(data_root, output_dir, selection_payload, contract, resume=bool(args.resume))
        print(json.dumps(test_receipt, ensure_ascii=False, indent=2))
        return 0 if bool(test_receipt.get("complete", False)) else 2
    raise AssertionError("unreachable stage")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
