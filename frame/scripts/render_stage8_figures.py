"""Render Stage 8 figures in a plotting-only process.

Keeping Matplotlib out of the PyTorch inference process avoids OpenMP runtime
collisions on Windows CPU environments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.stage8_figures import generate_all_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render frozen Stage 8 figures")
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()
    output = Path(args.input_dir).resolve()
    files = generate_all_figures(output)
    print({"status": "passed", "figure_count": len(files), "figures": files})


if __name__ == "__main__":
    main()

