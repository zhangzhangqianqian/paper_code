"""Source-hash-bound adapter for the official THUML iTransformer backbone."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor, nn


OFFICIAL_REPOSITORY = "https://github.com/thuml/iTransformer"
OFFICIAL_BACKBONE_CLASS = "model.iTransformer.Model"
OFFICIAL_COMMIT = "c2426e68ca13f74aaec08045c5c724d8ad328124"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_itransformer_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != "formal-v4.1-itransformer-source-v1":
        raise ValueError("iTransformer receipt schema is not formal-v4.1")
    if receipt.get("repository") != OFFICIAL_REPOSITORY:
        raise ValueError("iTransformer receipt repository is not the official THUML source")
    if receipt.get("backbone_class") != OFFICIAL_BACKBONE_CLASS:
        raise ValueError("iTransformer receipt does not identify model.iTransformer.Model")
    if receipt.get("reproduction_level") != "official_backbone_adaptation":
        raise ValueError("iTransformer receipt must be marked official_backbone_adaptation")
    if receipt.get("verified") is not True or not receipt.get("commit"):
        raise ValueError("iTransformer source receipt is not verified")
    if receipt.get("commit") != OFFICIAL_COMMIT:
        raise ValueError("iTransformer receipt commit is not the frozen THUML commit")
    source_root = receipt.get("source_root")
    if not isinstance(source_root, str) or not source_root or Path(source_root).is_absolute() or ".." in Path(source_root).parts:
        raise ValueError("iTransformer receipt source_root must be repository-relative")
    license_file = receipt.get("license_file")
    if not isinstance(license_file, str) or not license_file or Path(license_file).is_absolute() or ".." in Path(license_file).parts:
        raise ValueError("iTransformer receipt license_file must be source-root-relative")
    if not isinstance(receipt.get("license_sha256"), str) or len(receipt["license_sha256"]) != 64:
        raise ValueError("iTransformer license hash is invalid")


def verify_itransformer_source_files(source_root: str | Path, receipt: Mapping[str, Any], *, require_license: bool = False) -> None:
    """Rehash every imported upstream file at Gate time."""

    validate_itransformer_receipt(receipt)
    root = Path(source_root).resolve()
    try:
        actual_commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("iTransformer source root is not a readable Git checkout") from exc
    if actual_commit != str(receipt["commit"]):
        raise ValueError("iTransformer Git HEAD does not match the receipt commit")
    imported = receipt.get("imported_file_hashes", {})
    if not isinstance(imported, Mapping) or not imported:
        raise ValueError("iTransformer receipt must list imported source hashes")
    for relative, expected in imported.items():
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("iTransformer receipt path escapes source root") from exc
        if not path.is_file() or _sha256(path) != str(expected):
            raise ValueError(f"iTransformer upstream file hash mismatch: {relative}")
    if require_license:
        license_path = root / str(receipt["license_file"])
        expected = receipt.get("license_sha256")
        if not isinstance(expected, str) or len(expected) != 64 or not license_path.is_file() or _sha256(license_path) != expected:
            raise ValueError("iTransformer license hash verification failed")


class OfficialITransformerAdapter(nn.Module):
    """Use the upstream ``models.iTransformer.Model`` with disclosed heads."""

    def __init__(self, source_root: str | Path, receipt_path: str | Path, config: Mapping[str, object] | None = None) -> None:
        super().__init__()
        self.source_root = Path(source_root).resolve()
        self.receipt_path = Path(receipt_path).resolve()
        if not self.receipt_path.exists():
            raise FileNotFoundError(f"iTransformer source receipt is missing: {self.receipt_path}")
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, Mapping):
            raise ValueError("iTransformer source receipt must be a JSON object")
        validate_itransformer_receipt(receipt)
        verify_itransformer_source_files(self.source_root, receipt, require_license=str(receipt.get("schema_version", "")).startswith("formal-v4.1"))
        module_dir = str(self.source_root)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        try:
            module = importlib.import_module("model.iTransformer")
        except Exception as exc:  # pragma: no cover - depends on upstream dependencies
            raise ImportError("unable to import official model.iTransformer.Model") from exc
        if not hasattr(module, "Model"):
            raise ImportError("official model.iTransformer module has no Model class")
        options = dict(config or {})
        defaults = {
            "seq_len": 24, "pred_len": 4, "enc_in": 4, "d_model": 64, "n_heads": 4,
            "e_layers": 2, "d_ff": 256, "dropout": 0.0, "factor": 1,
            "activation": "gelu", "output_attention": False, "class_strategy": "projection",
            "use_norm": True, "embed": "timeF", "freq": "h",
        }
        defaults.update(options)
        self.backbone = module.Model(SimpleNamespace(**defaults))
        self.receipt = dict(receipt)

    def forward(self, load_history: Tensor) -> Tensor:
        if load_history.ndim != 3 or tuple(load_history.shape[1:]) != (24, 4):
            raise ValueError("load_history must have shape [B,24,4]")
        if not bool(torch.isfinite(load_history).all()):
            raise ValueError("load_history must be finite")
        # The official implementation has historically used the
        # (x_enc, x_mark_enc, x_dec, x_mark_dec) signature; a small number of
        # upstream revisions expose the simpler one-tensor signature.  Both
        # are upstream calls, never a local inverted-token substitute.
        try:
            output = self.backbone(load_history, None, None, None)
        except TypeError:
            output = self.backbone(load_history)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, Tensor):
            raise ValueError("official iTransformer forward did not return a tensor")
        if output.shape[-2:] == (4, 4):
            result = output
        else:
            raise ValueError("official iTransformer adaptation returned an unexpected shape")
        return result.contiguous()


__all__ = ["OFFICIAL_BACKBONE_CLASS", "OFFICIAL_COMMIT", "OFFICIAL_REPOSITORY", "OfficialITransformerAdapter", "validate_itransformer_receipt", "verify_itransformer_source_files"]
