#!/usr/bin/env python3
"""Validate and deterministically merge Step 164 training artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from scripts.run_full_history_engine_training import (
    ARTIFACT_VERSION,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_hash,
    validate_artifact_schema,
)


ROOT = Path(__file__).resolve().parents[1]


def artifact_key(artifact: dict[str, Any]) -> tuple[Any, ...]:
    return (
        artifact["engine_group"],
        artifact["engine_name"],
        artifact["training_mode"],
        artifact["worker_id"],
        artifact["draw_cutoff"],
        artifact["day_type"],
    )


def discover_artifact_paths(input_dir: Path) -> list[Path]:
    paths = []
    for path in input_dir.rglob("*.json"):
        if "worker_rows" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("artifact_version") == ARTIFACT_VERSION
        ):
            paths.append(path)
    return sorted(paths)


def discover_dataset_metadata(input_dir: Path) -> list[dict[str, Any]]:
    output = []
    rows_dir = input_dir / "worker_rows"
    if not rows_dir.exists():
        return output
    for path in sorted(rows_dir.glob("*__dataset.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        output.append({"path": str(path), **payload})
    return output


def find_duplicate_keys(
    artifacts: Sequence[dict[str, Any]],
) -> dict[tuple[Any, ...], int]:
    counts = Counter(artifact_key(artifact) for artifact in artifacts)
    return {key: count for key, count in counts.items() if count > 1}


def load_and_validate(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = []
    errors = []
    for path in paths:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            validate_artifact_schema(artifact)
            valid.append({"path": str(path), "artifact": artifact})
        except Exception as exc:
            errors.append(
                {
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return valid, errors


def summarize(
    valid_items: Sequence[dict[str, Any]],
    errors: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    artifacts = [item["artifact"] for item in valid_items]
    by_group = Counter(item["engine_group"] for item in artifacts)
    by_mode = Counter(item["training_mode"] for item in artifacts)
    by_firewall = Counter(item["temporal_firewall_status"] for item in artifacts)
    pair_counts = defaultdict(list)
    residuals = defaultdict(list)
    skipped = []
    for item in artifacts:
        pair_counts[item["engine_group"]].append(item["training_pair_count"])
        residual = item.get("residual_summary", {}).get("mse_per_observation")
        if residual is not None:
            residuals[item["engine_group"]].append(float(residual))
        if item.get("feature_stats", {}).get("status", "").startswith("SKIP"):
            skipped.append(item["engine_name"])
    return {
        "artifact_count": len(artifacts),
        "artifacts_by_engine_group": dict(sorted(by_group.items())),
        "artifacts_by_training_mode": dict(sorted(by_mode.items())),
        "temporal_firewall_status": dict(sorted(by_firewall.items())),
        "pair_count_summary": {
            group: {
                "artifact_count": len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for group, values in sorted(pair_counts.items())
        },
        "residual_summary": {
            group: {
                "count": len(values),
                "minimum_mse": min(values),
                "maximum_mse": max(values),
                "mean_mse": sum(values) / len(values),
            }
            for group, values in sorted(residuals.items())
            if values
        },
        "validation_errors": list(errors),
        "skipped_models": skipped,
        "all_hashes_valid": not errors,
        "all_firewalls_pass": all(
            item["temporal_firewall_status"] == "PASS_TARGET_DRAW_LE_CUTOFF"
            for item in artifacts
        ),
        "retrospective_artifacts_not_for_live": all(
            item["not_for_live_prediction"]
            for item in artifacts
            if item["training_mode"] == "retrospective_full_history"
        ),
    }


def render_report(
    summary: dict[str, Any],
    valid_items: Sequence[dict[str, Any]],
    duplicate_keys: dict[tuple[Any, ...], int],
    dataset_metadata: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "STEP 164 — FULL 40-YEAR ENGINE TRAINING FOUNDATION — OFFLINE ONLY",
        "ProductionPredictionChanged: NO",
        "DBWritePerformed: NO",
        "PredictionLedgerWritePerformed: NO",
        "DeepCandidateLedgerWritePerformed: NO",
        f"ArtifactCount: {summary['artifact_count']}",
        f"AllHashesValid: {summary['all_hashes_valid']}",
        f"AllTemporalFirewallsPass: {summary['all_firewalls_pass']}",
        f"DuplicateArtifactKeys: {len(duplicate_keys)}",
        "",
        "DRAW HISTORY DISCOVERY",
    ]
    if dataset_metadata:
        dataset = dataset_metadata[0]["dataset"]
        lines.extend(
            [
                f"DrawNoRange: {dataset['draw_no_range']}",
                f"DrawDateRange: {dataset['draw_date_range']}",
                f"DrawCount: {dataset['draw_count']}",
                f"Phase1DrawCount: {dataset['phase1_draw_count']}",
                f"Phase2DrawCount: {dataset['phase2_draw_count']}",
                f"ConsecutivePairCount: {dataset['consecutive_pair_count']}",
                f"DayTypeDistribution: {dataset['day_type_distribution']}",
                f"WinnerCountDistribution: {dataset['winner_count_distribution']}",
                f"DrawHistorySchema: {dataset.get('draw_history_schema', [])}",
                (
                    "ChronologicalDrawCacheExists: "
                    f"{dataset.get('engine_foundation_discovery', {}).get('chronological_draw_cache_exists')}"
                ),
                (
                    "LiveTrainingWindowDefault: "
                    f"{dataset.get('engine_foundation_discovery', {}).get('live_training_window_default')}"
                ),
                (
                    "OfflineFullHistoryWindowZeroSupported: "
                    f"{dataset.get('engine_foundation_discovery', {}).get('offline_full_history_window_zero_supported')}"
                ),
            ]
        )
    else:
        lines.append("DatasetMetadata: UNAVAILABLE")
    lines.extend(
        [
        "",
        "ARTIFACTS BY ENGINE GROUP",
        ]
    )
    for group in ("A", "B", "C", "D"):
        lines.append(
            f"ENGINE GROUP {group}: "
            f"{summary['artifacts_by_engine_group'].get(group, 0)}"
        )
    lines.extend(["", "ARTIFACTS BY TRAINING MODE"])
    for mode, count in summary["artifacts_by_training_mode"].items():
        lines.append(f"{mode}: {count}")
    lines.extend(["", "TRAINED ARTIFACTS"])
    for item in sorted(
        valid_items,
        key=lambda value: artifact_key(value["artifact"]),
    ):
        artifact = item["artifact"]
        lines.append(
            f"- [{artifact['engine_group']}] {artifact['engine_name']} "
            f"mode={artifact['training_mode']} cutoff={artifact['draw_cutoff']} "
            f"pairs={artifact['training_pair_count']} "
            f"TemporalFirewall={artifact['temporal_firewall_status']} "
            f"not_for_live={artifact['not_for_live_prediction']}"
        )
    if summary["validation_errors"]:
        lines.extend(["", "VALIDATION ERRORS"])
        for error in summary["validation_errors"]:
            lines.append(f"- {error['path']}: {error['error']}")
    lines.extend(
        [
            "",
            "FOUNDATION STATUS",
            (
                "phase1_base artifacts use target DrawNo <= 4050."
                if all(
                    artifact["draw_cutoff"] <= 4050
                    and (
                        artifact["training_draw_range"] is None
                        or artifact["training_draw_range"][1] <= 4050
                    )
                    for artifact in (
                        item["artifact"] for item in valid_items
                    )
                    if artifact["training_mode"] == "phase1_base"
                )
                else "phase1_base cutoff violation detected."
            ),
            "Retrospective schema requires RETROSPECTIVE_FULL_HISTORY_NOT_FOR_LIVE_PREDICTION.",
            "No predictive-success claim is made.",
            "ProductionPromotionRecommended: NO",
            "",
            "FOUR TERMINAL PHASE1 TRAINING COMMANDS",
            "cd /Users/kojeffrey/4D-ML-LOOP",
            "mkdir -p /tmp/j4d_step164_logs",
            (
                'PYTHONPATH="$PWD/backend:$PWD" J4D_NO_SQL_WRITE=1 '
                ".venv/bin/python scripts/run_full_history_engine_training.py "
                "--worker-id 1 --worker-count 4 --engine-group A "
                "--training-mode phase1_base --output-dir "
                "artifacts/full_history_training --no-sql-write --verbose "
                "| tee /tmp/j4d_step164_logs/worker1_groupA_phase1.log"
            ),
            (
                'PYTHONPATH="$PWD/backend:$PWD" J4D_NO_SQL_WRITE=1 '
                ".venv/bin/python scripts/run_full_history_engine_training.py "
                "--worker-id 2 --worker-count 4 --engine-group B "
                "--training-mode phase1_base --output-dir "
                "artifacts/full_history_training --no-sql-write --verbose "
                "| tee /tmp/j4d_step164_logs/worker2_groupB_phase1.log"
            ),
            (
                'PYTHONPATH="$PWD/backend:$PWD" J4D_NO_SQL_WRITE=1 '
                ".venv/bin/python scripts/run_full_history_engine_training.py "
                "--worker-id 3 --worker-count 4 --engine-group C "
                "--training-mode phase1_base --output-dir "
                "artifacts/full_history_training --no-sql-write --verbose "
                "| tee /tmp/j4d_step164_logs/worker3_groupC_phase1.log"
            ),
            (
                'PYTHONPATH="$PWD/backend:$PWD" J4D_NO_SQL_WRITE=1 '
                ".venv/bin/python scripts/run_full_history_engine_training.py "
                "--worker-id 4 --worker-count 4 --engine-group D "
                "--training-mode phase1_base --output-dir "
                "artifacts/full_history_training --no-sql-write --verbose "
                "| tee /tmp/j4d_step164_logs/worker4_groupD_phase1.log"
            ),
            "",
            "MERGE COMMAND",
            (
                'PYTHONPATH="$PWD/backend:$PWD" .venv/bin/python '
                "scripts/merge_full_history_engine_training.py "
                "--input-dir artifacts/full_history_training "
                "--report reports/step_164_full_history_engine_training_report.txt "
                "--matrices reports/step_164_full_history_engine_training_matrices.json "
                "--rows reports/step_164_full_history_engine_training_rows.jsonl"
            ),
            "",
            "OPTIONAL RETROSPECTIVE FULL HISTORY — NOT FOR LIVE PREDICTION",
            (
                "Run the same four commands with "
                "--training-mode retrospective_full_history and separate log names. "
                "Every resulting artifact is forced to carry "
                "RETROSPECTIVE_FULL_HISTORY_NOT_FOR_LIVE_PREDICTION."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = discover_artifact_paths(args.input_dir)
    dataset_metadata = discover_dataset_metadata(args.input_dir)
    valid_items, errors = load_and_validate(paths)
    duplicates = find_duplicate_keys(
        [item["artifact"] for item in valid_items]
    )
    summary = summarize(valid_items, errors)
    summary["duplicate_artifact_keys"] = [
        {"key": list(key), "count": count}
        for key, count in sorted(duplicates.items())
    ]
    summary["required_artifact_fields"] = list(REQUIRED_ARTIFACT_FIELDS)
    summary["dataset_metadata_files"] = dataset_metadata

    rows = []
    for item in sorted(
        valid_items,
        key=lambda value: artifact_key(value["artifact"]),
    ):
        artifact = item["artifact"]
        rows.append(
            {
                "row_type": "artifact",
                "path": item["path"],
                "engine_group": artifact["engine_group"],
                "engine_name": artifact["engine_name"],
                "training_mode": artifact["training_mode"],
                "worker_id": artifact["worker_id"],
                "draw_cutoff": artifact["draw_cutoff"],
                "training_pair_count": artifact["training_pair_count"],
                "residual_summary": artifact["residual_summary"],
                "temporal_firewall_status": artifact[
                    "temporal_firewall_status"
                ],
                "not_for_live_prediction": artifact[
                    "not_for_live_prediction"
                ],
                "sha256_hash": artifact["sha256_hash"],
            }
        )
    rows.extend({"row_type": "validation_error", **error} for error in errors)

    for output in (args.report, args.matrices, args.rows):
        output.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_report(summary, valid_items, duplicates, dataset_metadata),
        encoding="utf-8",
    )
    args.matrices.write_text(
        json.dumps(
            {
                "metadata": {
                    "step": "164",
                    "offline_only": True,
                    "db_write_performed": False,
                },
                "summary": summary,
                "dataset_metadata": dataset_metadata,
                "artifacts": [
                    {"path": item["path"], **item["artifact"]}
                    for item in valid_items
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.rows.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print("STEP 164 — FULL 40-YEAR ENGINE TRAINING FOUNDATION — MERGE")
    print(f"ArtifactsFound: {len(paths)}")
    print(f"ArtifactsValid: {len(valid_items)}")
    print(f"ValidationErrors: {len(errors)}")
    print(f"DuplicateKeys: {len(duplicates)}")
    print("DBWritePerformed: NO")
    if errors or duplicates:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
