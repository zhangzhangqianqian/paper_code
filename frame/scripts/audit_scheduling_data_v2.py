"""阶段10.3：审计双轨调度数据权限和年度时间范围。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.data import build_scheduling_frame, make_plan_view, make_settlement_view  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = build_scheduling_frame(args.data_dir, years=(2015, 2020, 2021))
    origin = "2021-01-02 00:00:00"
    plan = make_plan_view(frame, origin)
    settlement = make_settlement_view(frame, origin)
    forbidden = [column for column in plan.history.columns if column.startswith("actual_")]
    result = {
        "stage": "10.3",
        "years": list(frame.years),
        "rows": int(len(frame.data)),
        "columns": list(frame.data.columns),
        "plan_visible_columns": list(plan.visible_columns),
        "settlement_columns": list(settlement.protected_columns),
        "plan_contains_actual_columns": bool(forbidden),
        "plan_history_rows": int(len(plan.history)),
        "settlement_rows": int(len(settlement.realized)),
        "status": "pass" if not forbidden else "fail",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data_audit_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
