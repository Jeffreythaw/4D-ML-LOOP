#!/usr/bin/env python3
"""Promote validated 40-year artifacts into a compact read-only live pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_full_history_engine_training import (  # noqa: E402
    discover_artifact_paths,
    load_and_validate,
)


PACK_VERSION = "j4d.live40.v1"
EXPECTED_GROUP_COUNTS = {"A": 13, "B": 9, "C": 3, "D": 1}


def pack_hash(payload: dict[str, Any]) -> str:
    """Hash a pack while excluding its self-referential hash field."""
    canonical = {
        key: value for key, value in payload.items() if key != "pack_sha256"
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_pack(input_dir: Path) -> dict[str, Any]:
    """Build a deterministic promotion pack from validated retrospective artifacts."""
    paths = discover_artifact_paths(input_dir)
    valid_items, errors = load_and_validate(paths)
    if errors:
        raise RuntimeError(f"Artifact validation failed: {errors[:3]}")
    artifacts = [
        item["artifact"]
        for item in valid_items
        if item["artifact"]["training_mode"] == "retrospective_full_history"
    ]
    group_counts = {
        group: sum(item["engine_group"] == group for item in artifacts)
        for group in EXPECTED_GROUP_COUNTS
    }
    if group_counts != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(
            f"Expected retrospective groups {EXPECTED_GROUP_COUNTS}, got {group_counts}"
        )
    cutoffs = {int(item["draw_cutoff"]) for item in artifacts}
    ranges = {
        tuple(item["training_draw_range"])
        for item in artifacts
        if item["training_draw_range"] is not None
    }
    if len(cutoffs) != 1:
        raise RuntimeError(f"Retrospective artifacts have mixed cutoffs: {cutoffs}")
    cutoff = next(iter(cutoffs))
    if any(
        item["temporal_firewall_status"] != "PASS_TARGET_DRAW_LE_CUTOFF"
        for item in artifacts
    ):
        raise RuntimeError("A retrospective artifact failed its temporal firewall")
    if any(not item["not_for_live_prediction"] for item in artifacts):
        raise RuntimeError("Retrospective source artifact is missing its safety label")

    models = []
    for artifact in sorted(
        artifacts,
        key=lambda item: (item["engine_group"], item["engine_name"]),
    ):
        models.append(
            {
                "engine_group": artifact["engine_group"],
                "engine_name": artifact["engine_name"],
                "day_type": artifact["day_type"],
                "model_type": artifact["model_type"],
                "modulus": artifact["modulus"],
                "formula_space": artifact["formula_space"],
                "matrix_m": artifact["matrix_m"],
                "bias_b": artifact["bias_b"],
                "coefficients": artifact["coefficients"],
                "feature_stats": artifact["feature_stats"],
                "residual_summary": artifact["residual_summary"],
                "training_pair_count": artifact["training_pair_count"],
                "training_draw_range": artifact["training_draw_range"],
                "source_sha256": artifact["sha256_hash"],
            }
        )

    pack = {
        "pack_version": PACK_VERSION,
        "created_at_utc": max(
            str(item["created_at_utc"]) for item in artifacts
        ),
        "promotion_policy": {
            "enabled": True,
            "minimum_source_draw_no": cutoff,
            "target_blind": True,
            "historical_replay_before_cutoff_prohibited": True,
            "source_artifacts_retain_retrospective_safety_label": True,
            "fallback_on_load_or_validation_error": True,
        },
        "dataset": {
            "draw_cutoff": cutoff,
            "draw_no_range": [
                min(value[0] for value in ranges),
                max(value[1] for value in ranges),
            ],
            "maximum_training_pair_count": max(
                int(item["training_pair_count"]) for item in artifacts
            ),
        },
        "source_artifact_count": len(artifacts),
        "source_artifact_group_counts": group_counts,
        "models": models,
        "pack_sha256": "",
    }
    pack["pack_sha256"] = pack_hash(pack)
    return pack


def write_pack(pack: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "artifacts/full_history_training",
    )
    parser.add_argument(
        "--root-output",
        type=Path,
        default=ROOT / "live_knowledge/full_history_engine_pack.json",
    )
    parser.add_argument(
        "--backend-output",
        type=Path,
        default=ROOT / "backend/live_knowledge/full_history_engine_pack.json",
    )
    args = parser.parse_args()
    pack = build_pack(args.input_dir)
    write_pack(pack, args.root_output)
    write_pack(pack, args.backend_output)
    print("LIVE 40-YEAR KNOWLEDGE PACK BUILT")
    print(f"SourceArtifacts: {pack['source_artifact_count']}")
    print(f"DrawCutoff: {pack['dataset']['draw_cutoff']}")
    print(f"PackHash: {pack['pack_sha256']}")
    print("PredictionGenerated: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
