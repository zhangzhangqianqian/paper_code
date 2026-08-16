"""Strict registration and alignment of frozen 2020 validation artifacts.

This module is intentionally limited to validation artifacts from the existing
cross-topology protocol.  It never opens a test prediction file and never
copies the large NPZ files into the new protocol directory.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

from .data_pipeline import save_json


TASK_COUNT = 4
HORIZON = 4
EXPECTED_ORIGINS = 8781
EXPECTED_SEEDS: Tuple[int, ...] = (2026, 2027, 2028)
EXPECTED_MODELS: Tuple[Tuple[str, str], ...] = (
    ("scheme2r", "H4"),
    ("dynamic_symmetric", "H1"),
)
REQUIRED_KEYS = {"prediction", "target", "target_times"}
_PATH_RE = re.compile(
    r"(?:^|[\\/])full[\\/]([^\\/]+)[\\/]([^\\/]+)[\\/]seed_(20\d{2})[\\/]predictions_validation\.npz$",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ValidationArtifact:
    path: Path
    model: str
    candidate_id: str
    seed: int
    prediction: np.ndarray
    target: np.ndarray
    target_times: np.ndarray
    sha256: str

    @property
    def sample_count(self) -> int:
        return int(self.prediction.shape[0])

    @property
    def target_start(self) -> str:
        return str(self.target_times[0])

    @property
    def target_end(self) -> str:
        return str(self.target_times[-1])

    def subset(self, indices: np.ndarray) -> "ValidationArtifact":
        indices = np.asarray(indices, dtype=np.int64)
        return ValidationArtifact(
            path=self.path,
            model=self.model,
            candidate_id=self.candidate_id,
            seed=self.seed,
            prediction=self.prediction[indices],
            target=self.target[indices],
            target_times=self.target_times[indices],
            sha256=self.sha256,
        )


def _parse_path(path: Path) -> tuple[str, str, int]:
    if path.name.lower() != "predictions_validation.npz":
        raise ValueError("only predictions_validation.npz is accepted")
    if "predictions_test" in path.name.lower():
        raise ValueError("test prediction artifacts are forbidden")
    match = _PATH_RE.search(str(path))
    if match is None:
        raise ValueError(
            "validation artifact path must match full/<model>/<candidate>/seed_<seed>/"
            "predictions_validation.npz"
        )
    return match.group(1).lower().replace("-", "_"), match.group(2), int(match.group(3))


def load_validation_artifact(
    path: str | Path,
    *,
    expected_model: str | None = None,
    expected_candidate_id: str | None = None,
    expected_seed: int | None = None,
    require_expected_origin_count: bool = False,
) -> ValidationArtifact:
    """Load and validate one frozen validation NPZ without accepting test files."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    model, candidate_id, seed = _parse_path(artifact_path)
    if expected_model is not None and model != expected_model.lower().replace("-", "_"):
        raise ValueError(f"model path mismatch: expected {expected_model}, got {model}")
    if expected_candidate_id is not None and candidate_id != str(expected_candidate_id):
        raise ValueError(f"candidate path mismatch: expected {expected_candidate_id}, got {candidate_id}")
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError(f"seed path mismatch: expected {expected_seed}, got {seed}")
    with np.load(artifact_path, allow_pickle=False) as values:
        if not REQUIRED_KEYS.issubset(values.files):
            raise ValueError(f"artifact is missing required keys: {sorted(REQUIRED_KEYS - set(values.files))}")
        prediction = np.asarray(values["prediction"], dtype=np.float32)
        target = np.asarray(values["target"], dtype=np.float32)
        target_times = np.asarray(values["target_times"]).astype("datetime64[ns]")
    if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (HORIZON, TASK_COUNT):
        raise ValueError(f"prediction must have shape [N,{HORIZON},{TASK_COUNT}]")
    if target.shape != prediction.shape:
        raise ValueError("prediction and target shapes must match")
    if target_times.ndim != 1 or len(target_times) != len(prediction):
        raise ValueError("target_times must be one-dimensional and match prediction length")
    if len(target_times) == 0:
        raise ValueError("validation artifact cannot be empty")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    if len(np.unique(target_times)) != len(target_times):
        raise ValueError("target_times must be unique")
    if np.any(np.diff(target_times) <= np.timedelta64(0, "ns")):
        raise ValueError("target_times must be strictly increasing")
    years = target_times.astype("datetime64[Y]").astype(int) + 1970
    if np.any(years != 2020):
        raise ValueError("cross-topology registration accepts only 2020 validation origins")
    if require_expected_origin_count and len(target_times) != EXPECTED_ORIGINS:
        raise ValueError(f"expected {EXPECTED_ORIGINS} validation origins, got {len(target_times)}")
    return ValidationArtifact(
        path=artifact_path,
        model=model,
        candidate_id=candidate_id,
        seed=seed,
        prediction=prediction,
        target=target,
        target_times=target_times,
        sha256=_sha256(artifact_path),
    )


def align_validation_origins(
    left: ValidationArtifact,
    right: ValidationArtifact,
) -> tuple[ValidationArtifact, ValidationArtifact]:
    """Intersect validation origins and verify target equality after alignment."""

    if left.seed != right.seed:
        raise ValueError("cannot align artifacts from different seeds")
    if left.target_times.shape == right.target_times.shape and np.array_equal(left.target_times, right.target_times):
        left_indices = right_indices = np.arange(len(left.target_times), dtype=np.int64)
    else:
        common, left_indices, right_indices = np.intersect1d(
            left.target_times, right.target_times, assume_unique=True, return_indices=True
        )
        if len(common) == 0:
            raise ValueError("validation artifacts have no common origins")
    left_aligned = left.subset(left_indices)
    right_aligned = right.subset(right_indices)
    if not np.array_equal(left_aligned.target_times, right_aligned.target_times):
        raise ValueError("aligned target origins disagree")
    if not np.allclose(left_aligned.target, right_aligned.target, rtol=0.0, atol=1e-6):
        raise ValueError("aligned target arrays disagree")
    return left_aligned, right_aligned


def _source_path(repo_root: Path, model: str, candidate: str, seed: int) -> Path:
    if (model, candidate) == ("scheme2r", "H4"):
        base = repo_root / "frame" / "reports" / "stage7r_3_kitakyushu_formal"
    elif (model, candidate) == ("dynamic_symmetric", "H1"):
        base = repo_root / "frame" / "reports" / "stage7r_7_joint_baselines_formal"
    else:
        raise ValueError(f"unsupported frozen cross-topology source: {model}/{candidate}")
    return base / "full" / model / candidate / f"seed_{seed}" / "predictions_validation.npz"


def register_cross_topology_artifacts(
    repo_root: str | Path,
    output_dir: str | Path,
    *,
    require_expected_origin_count: bool = True,
) -> dict[str, object]:
    """Validate and register the six existing 2020 artifacts."""

    root = Path(repo_root)
    output_root = Path(output_dir)
    artifacts: list[dict[str, object]] = []
    for model, candidate in EXPECTED_MODELS:
        for seed in EXPECTED_SEEDS:
            path = _source_path(root, model, candidate, seed)
            artifact = load_validation_artifact(
                path,
                expected_model=model,
                expected_candidate_id=candidate,
                expected_seed=seed,
                require_expected_origin_count=require_expected_origin_count,
            )
            artifacts.append({
                "path": str(path.relative_to(root)),
                "model": artifact.model,
                "candidate_id": artifact.candidate_id,
                "seed": artifact.seed,
                "target_start": artifact.target_start,
                "target_end": artifact.target_end,
                "sample_count": artifact.sample_count,
                "sha256": artifact.sha256,
            })
    registry = {
        "stage": "topology_protocol_pilot_task6",
        "protocol": "cross_topology",
        "validation_year": 2020,
        "test_set_accessed": False,
        "artifacts": artifacts,
    }
    save_json(registry, output_root / "cross_topology_artifact_registry.json")
    return registry


__all__ = [
    "EXPECTED_MODELS",
    "EXPECTED_ORIGINS",
    "EXPECTED_SEEDS",
    "ValidationArtifact",
    "align_validation_origins",
    "load_validation_artifact",
    "register_cross_topology_artifacts",
]
