"""Export exact versions for the isolated formal-v4 DiffLP environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_diffopt import write_diffopt_lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, default=FRAME_ROOT / "requirements" / "formal_v4_diffopt.in")
    args = parser.parse_args()
    payload = write_diffopt_lock(args.output.resolve(), args.requirements.resolve())
    print(payload)
    probe_ok = payload.get("native_layer_probe", {}).get("eligible_for_gate0", False)
    return 0 if payload["status"] == "resolved" and all(payload["packages"].values()) and probe_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
