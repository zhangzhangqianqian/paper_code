"""Build the external, implementation-bound formal-v4.4 source manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_4_artifacts import sha256_file  # noqa: E402
from src.joint_dispatch.formal_v4_4_contract import load_formal_v4_4_contract  # noqa: E402
from src.joint_dispatch.formal_v4_4_provenance import build_source_manifest, write_source_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    contract = load_formal_v4_4_contract(args.contract)
    manifest = build_source_manifest(repo_root=args.repo_root, run_id=args.run_id, contract_sha256=contract.contract_sha256)
    digest = write_source_manifest(args.output, manifest)
    print(f"source_manifest={args.output.resolve()}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
