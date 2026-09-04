"""Fetch and freeze the official THUML iTransformer source once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_itransformer import OFFICIAL_BACKBONE_CLASS, OFFICIAL_REPOSITORY


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--protocol-amendment", type=Path, default=None)
    args = parser.parse_args()
    destination = args.destination.resolve()
    receipt_path = args.receipt.resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", OFFICIAL_REPOSITORY, str(destination)], check=True)
    commit = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        if existing.get("commit") != commit and args.protocol_amendment is None:
            raise RuntimeError("refusing a different iTransformer commit without a protocol amendment")
    imported_files = {}
    for path in sorted(destination.rglob("*.py")):
        relative = path.relative_to(destination).as_posix()
        imported_files[relative] = _sha256(path)
    license_path = next((path for path in (destination / "LICENSE", destination / "LICENSE.md", destination / "LICENSE.txt") if path.exists()), None)
    if license_path is None:
        raise FileNotFoundError("official iTransformer LICENSE file is missing")
    payload = {
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "backbone_class": OFFICIAL_BACKBONE_CLASS,
        "imported_file_hashes": imported_files,
        "license_path": license_path.relative_to(destination).as_posix(),
        "license_sha256": _sha256(license_path),
        "reproduction_level": "official_backbone_adaptation",
        "verified": True,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
