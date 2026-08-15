"""CSV/JSON reporting helpers for the two scheduling tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from .metrics import summarize_real_replay, summarize_simulated_dispatch


def write_dual_track_report(
    real_rows: pd.DataFrame,
    simulated_rows: pd.DataFrame,
    output_dir: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> Mapping[str, str]:
    """Write separate raw and summary tables for R and S tracks."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    real_rows.to_csv(destination / "real_replay_raw.csv", index=False, encoding="utf-8-sig")
    simulated_rows.to_csv(destination / "simulated_dispatch_raw.csv", index=False, encoding="utf-8-sig")
    summarize_real_replay(real_rows).to_csv(destination / "real_replay_summary.csv", index=False, encoding="utf-8-sig")
    summarize_simulated_dispatch(simulated_rows).to_csv(destination / "simulated_dispatch_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {"track_separation": ["real_replay", "simulated_dispatch"], **dict(metadata or {})}
    (destination / "dual_track_report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_dir": str(destination)}
