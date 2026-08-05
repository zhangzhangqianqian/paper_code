"""Generate the final Stage 7-R run matrix without touching data or training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage7_contract import EXPECTED_SEEDS  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层必须是对象：{path}")
    return value


def build_formal_run_matrix(
    freeze: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    primary = freeze.get("primary_model")
    comparison = freeze.get("comparison_model")
    scheme2r_reference = freeze.get("scheme2r_ablation_reference")
    if (
        not isinstance(primary, Mapping)
        or not isinstance(comparison, Mapping)
        or not isinstance(scheme2r_reference, Mapping)
    ):
        raise ValueError("冻结文件缺少主模型或比较模型")
    selected = [primary, comparison]
    rows: List[Dict[str, Any]] = []

    def add(stage: str, protocol: str, model: str, candidate: str, seed: Any, execution: str, reason: str) -> None:
        rows.append(
            {
                "stage": stage,
                "protocol": protocol,
                "model": model,
                "candidate_id": candidate,
                "seed": seed,
                "execution": execution,
                "reason": reason,
                "run_id": f"{stage}/{protocol}/{model}/{candidate}/{seed}",
            }
        )

    for item in selected:
        model = str(item["model"])
        candidate = str(item["candidate_id"])
        for protocol in ("full", "small_sample"):
            for seed in EXPECTED_SEEDS:
                add("7.3", protocol, model, candidate, seed, "train", "主模型与主要内部对照")

    scheme2r_candidate = str(scheme2r_reference["candidate_id"])
    for protocol in ("full", "small_sample"):
        for ablation in ("A0", "A1", "A2", "A3"):
            for seed in EXPECTED_SEEDS:
                add("7.4", protocol, ablation, scheme2r_candidate, seed, "train", "Scheme2R递进消融")
        for seed in EXPECTED_SEEDS:
            a4_execution = "reuse" if str(primary["model"]) == "scheme2r" else "train"
            a4_reason = (
                "复用 Scheme2R 主模型，禁止重复训练"
                if a4_execution == "reuse"
                else "主模型非 Scheme2R，单独训练 Scheme2R 完整消融模型"
            )
            add("7.4", protocol, "A4", scheme2r_candidate, seed, a4_execution, a4_reason)

    for protocol in ("full", "small_sample"):
        for model in ("persistence", "seasonal_naive"):
            add("7.5", protocol, model, "deterministic", "deterministic", "deterministic", "确定性基线")
        for model in ("dlinear", "mmoe_lite", "softs"):
            for seed in EXPECTED_SEEDS:
                add("7.5", protocol, model, "external_fixed", seed, "train", "学习型外部基线")
        for seed in EXPECTED_SEEDS:
            add(
                "7.5",
                protocol,
                "scheme2r_loads_only",
                scheme2r_candidate,
                seed,
                "train",
                "与 loads-only 外部基线公平比较的输入控制",
            )

    for seed in EXPECTED_SEEDS:
        for protocol in ("full", "small_sample"):
            add("7R.STL", protocol, "stl_matched", scheme2r_candidate, seed, "train", "Scheme2R结构匹配单任务参照")

    fingerprints = set()
    for row in rows:
        fingerprint = (
            row["stage"], row["protocol"], row["model"], row["candidate_id"],
            row["seed"], row["execution"],
        )
        if fingerprint in fingerprints:
            raise ValueError(f"运行矩阵存在重复指纹：{fingerprint!r}")
        fingerprints.add(fingerprint)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Stage 7-R 正式运行矩阵")
    parser.add_argument("--freeze-config", default="frame/reports/stage6r_6_kitakyushu/stage6_selected_config.json")
    parser.add_argument("--contract", default="frame/configs/stage7r_contract.json")
    parser.add_argument("--output-dir", default="frame/reports/stage7r_matrix")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze_path = _resolve(args.freeze_config)
    contract_path = _resolve(args.contract)
    freeze = _read_json(freeze_path)
    contract = _read_json(contract_path)
    if contract.get("contract_status") != "ready_for_stage7_smoke":
        raise ValueError(
            "formal run matrix requires a fresh Stage 7.0 contract; "
            "regenerate it after Stage 6-R freeze"
        )
    rows = build_formal_run_matrix(freeze, contract)
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    result = {
        "status": "dry_run" if args.dry_run else "generated",
        "freeze_config": str(freeze_path),
        "contract": str(contract_path),
        "total_effective_runs": len(rows),
        "trained_runs": sum(row["execution"] == "train" for row in rows),
        "reused_runs": sum(row["execution"] == "reuse" for row in rows),
        "runs_by_stage": counts,
        "duplicate_count": 0,
    }
    if args.dry_run:
        result["runs"] = rows
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    output = _resolve(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"输出目录非空，请使用 --force：{output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "formal_run_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result["output_files"] = ["formal_run_matrix.csv", "formal_run_matrix.json"]
    with (output / "formal_run_matrix.json").open("w", encoding="utf-8") as handle:
        json.dump({**result, "runs": rows}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
