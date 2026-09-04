"""Verify and freeze the local official THUML iTransformer source once."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_gate0_evidence import write_immutable_json
from src.joint_dispatch.formal_v4_itransformer import OFFICIAL_BACKBONE_CLASS, OFFICIAL_COMMIT, OFFICIAL_REPOSITORY, validate_itransformer_receipt, verify_itransformer_source_files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(source_root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.source_root.resolve()
    receipt_path = args.output_receipt.resolve()
    if not destination.is_dir():
        raise FileNotFoundError(f"local iTransformer source root is missing: {destination}")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite iTransformer receipt: {receipt_path}")
    commit = _git_commit(destination)
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"local iTransformer commit {commit} does not match frozen commit {OFFICIAL_COMMIT}")
    imported_files = {}
    for path in sorted(destination.rglob("*.py")):
        relative = path.relative_to(destination).as_posix()
        imported_files[relative] = _sha256(path)
    license_file = next((path for path in (destination / "LICENSE", destination / "LICENSE.md", destination / "LICENSE.txt") if path.exists()), None)
    if license_file is None:
        raise FileNotFoundError("official iTransformer LICENSE file is missing")
    try:
        source_root = destination.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("iTransformer source root must be inside the repository") from exc
    payload = {
        "schema_version": "formal-v4.1-itransformer-source-v1",
        "protocol_id": "formal-v4.1-itransformer-source-freeze-v1",
        "source_root": source_root,
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "backbone_class": OFFICIAL_BACKBONE_CLASS,
        "imported_file_hashes": imported_files,
        "license_file": license_file.relative_to(destination).as_posix(),
        "license_sha256": _sha256(license_file),
        "reproduction_level": "official_backbone_adaptation",
        "verified": True,
    }
    validate_itransformer_receipt(payload)
    verify_itransformer_source_files(destination, payload, require_license=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(receipt_path, payload)
    print("iTransformer source receipt written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
