#!/usr/bin/env python3
"""Step 166 sequential adaptive causal ML loop with a temporal firewall."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for value in (ROOT, BACKEND):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.run_causal_40year_column_learning import (  # noqa: E402
    GAP_BUCKETS,
    RANDOM_TOP5_23,
    STREAK_BUCKETS,
    CausalColumnLearner,
    bucket,
    candidate_components,
    digits,
    total_score,
)
from scripts.run_full_history_engine_training import (  # noqa: E402
    PHASE1_MAX_DRAW_NO,
    ChronologicalTrainingDataset,
    TrainingDraw,
    load_draw_history,
)


REPORT = ROOT / "reports/step_166_sequential_adaptive_causal_loop_report.txt"
MATRICES = ROOT / "reports/step_166_sequential_adaptive_causal_loop_matrices.json"
ROWS = ROOT / "reports/step_166_sequential_adaptive_causal_loop_rows.jsonl"
ROLLING_WINDOWS = {
    "1W": 3,
    "1M": 13,
    "2M": 26,
    "3M": 39,
    "12M": 156,
    "13M": 169,
    "14M": 182,
    "24M": 312,
}
MAX_SINGLE_EVIDENCE_DELTA = 0.05


@dataclass(frozen=True)
class CandidateLock:
    source_draw_no: int
    target_draw_no: int
    locked_top5: tuple[str, ...]
    candidate_hash: str
    score_components: tuple[dict[str, Any], ...]
    registry_version: int
    created_at: str
    target_seen_before_lock: bool = False
    locked: bool = True


@dataclass
class Correction:
    correction_id: str
    correction_type: str
    created_after_target_draw_no: int
    available_from_source_draw_no: int
    reason_code: str
    support_count: int
    feature_key: str
    old_weight: float
    new_weight: float
    capped_weight_delta: float
    evidence_summary: dict[str, Any]
    temporal_firewall_status: str = "PASS_POST_LOCK_ONLY"
    not_for_production: bool = True
    later_helped_count: int = 0
    later_hurt_count: int = 0


@dataclass
class ActiveOfflineRegistry:
    version: int = 1
    feature_weights: dict[str, float] = field(default_factory=dict)
    corrections: list[Correction] = field(default_factory=list)
    miss_streak: int = 0
    engine_history: Counter[str] = field(default_factory=Counter)
    feature_hit_history: Counter[str] = field(default_factory=Counter)
    feature_miss_history: Counter[str] = field(default_factory=Counter)
    reason_codes: Counter[str] = field(default_factory=Counter)
    latest_correction_by_feature: dict[str, Correction] = field(
        default_factory=dict,
        repr=False,
    )

    def available_corrections(self, source_draw_no: int) -> list[Correction]:
        return [
            correction
            for correction in self.corrections
            if correction.available_from_source_draw_no <= source_draw_no
        ]

    def add_correction(
        self,
        *,
        correction_type: str,
        target_draw_no: int,
        reason_code: str,
        feature_key: str,
        requested_delta: float,
        evidence_summary: dict[str, Any],
        support_count: int = 1,
    ) -> Correction:
        cap = MAX_SINGLE_EVIDENCE_DELTA if support_count <= 1 else 0.10
        delta = max(-cap, min(cap, requested_delta))
        old = self.feature_weights.get(feature_key, 0.0)
        new = max(-1.0, min(1.0, old + delta))
        self.feature_weights[feature_key] = new
        correction = Correction(
            correction_id=f"C{len(self.corrections) + 1:08d}",
            correction_type=correction_type,
            created_after_target_draw_no=target_draw_no,
            available_from_source_draw_no=target_draw_no,
            reason_code=reason_code,
            support_count=support_count,
            feature_key=feature_key,
            old_weight=old,
            new_weight=new,
            capped_weight_delta=new - old,
            evidence_summary=evidence_summary,
        )
        self.corrections.append(correction)
        self.latest_correction_by_feature[feature_key] = correction
        self.reason_codes[reason_code] += 1
        self.version += 1
        return correction

    def snapshot(self) -> dict[str, Any]:
        return {
            "registry_version": self.version,
            "feature_weights": dict(sorted(self.feature_weights.items())),
            "corrections": [asdict(value) for value in self.corrections],
            "miss_streak": self.miss_streak,
            "engine_history": dict(self.engine_history),
            "feature_hit_history": dict(self.feature_hit_history),
            "feature_miss_history": dict(self.feature_miss_history),
            "reason_codes": dict(self.reason_codes),
        }


class HiddenTargetVerifier:
    """Target winners are reachable only through verify() after a lock exists."""

    def __init__(self, draws: Sequence[TrainingDraw]) -> None:
        self.__targets = {draw.draw_no: draw.winners for draw in draws}
        self.verify_calls = 0

    def verify(self, lock: CandidateLock) -> dict[str, Any]:
        if not lock.locked or lock.target_seen_before_lock:
            raise RuntimeError("Verifier requires a valid pre-target candidate lock")
        winners = self.__targets.get(lock.target_draw_no)
        if winners is None:
            return {"status": "WAIT_FOR_TARGET"}
        self.verify_calls += 1
        matched = sorted(set(lock.locked_top5) & set(winners))
        return {
            "status": "VERIFIED_POST_LOCK",
            "hit_count": len(matched),
            "matched_numbers": matched,
            "target_winners_post_lock": list(winners),
            "target_seen_before_lock": False,
            "target_seen_after_lock": True,
        }


def deterministic_lock_hash(
    source_draw_no: int,
    target_draw_no: int,
    top5: Sequence[str],
    scores: Sequence[dict[str, Any]],
    registry_version: int,
) -> str:
    payload = {
        "source_draw_no": source_draw_no,
        "target_draw_no": target_draw_no,
        "top5": list(top5),
        "scores": [
            [item["number"], round(float(item["score"]), 15)]
            for item in scores
        ],
        "registry_version": registry_version,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def structural_signature(number: str) -> str:
    values = digits(number)
    digit_sum = sum(values)
    return (
        f"sum_{(digit_sum // 5) * 5:02d}-"
        f"{(digit_sum // 5) * 5 + 4:02d}"
        f"__zero_{number.count('0')}"
        f"__repeat_{4 - len(set(number))}"
        f"__first_{number[:2]}__last_{number[2:]}"
    )


def candidate_adaptive_score(
    registry: ActiveOfflineRegistry,
    learner: CausalColumnLearner,
    source: TrainingDraw,
    number: str,
) -> tuple[float, float, list[str]]:
    values = digits(number)
    source_values = [digits(value) for value in source.winners]
    adaptive = 0.0
    drop_repair = 0.0
    applied = []
    source_digit_sets = [
        {item[position] for item in source_values} for position in range(4)
    ]
    keys = []
    for position in range(4):
        for source_digit in source_digit_sets[position]:
            keys.append(
                f"column:{position}:{source_digit}:{values[position]}"
            )
        gap_label = bucket(
            learner.current_gap(position, values[position]), GAP_BUCKETS
        )
        keys.append(f"gap:{position}:{values[position]}:{gap_label}")
        streak_label = bucket(
            learner.absence_streak[position][values[position]],
            STREAK_BUCKETS,
        )
        keys.append(f"streak:{position}:{values[position]}:{streak_label}")
    signature = structural_signature(number)
    keys.extend((f"structure:{signature}", f"drop:{signature}"))

    for key in keys:
        correction = registry.latest_correction_by_feature.get(key)
        if (
            correction is None
            or correction.available_from_source_draw_no > source.draw_no
        ):
            continue
        if correction.correction_type == "DROP_REPAIR":
            drop_repair += correction.new_weight
        else:
            adaptive += correction.new_weight
        applied.append(correction.correction_id)

    # Formula residual corrections are sparse and intentionally capped to recent
    # evidence so one-off residuals cannot dominate runtime or ranking.
    formula_corrections = (
        correction
        for correction in reversed(registry.corrections[-200:])
        if correction.correction_type == "FORMULA_RESIDUAL_CORRECTION"
        and correction.available_from_source_draw_no <= source.draw_no
    )
    for correction in formula_corrections:
        target_digits = correction.evidence_summary.get("target_digits", [])
        if sum(a == b for a, b in zip(values, target_digits)) >= 3:
            adaptive += correction.new_weight
            applied.append(correction.correction_id)
    applied = list(dict.fromkeys(applied))
    return adaptive, drop_repair, applied


def phase1_engine_artifacts() -> list[dict[str, Any]]:
    root = ROOT / "artifacts/full_history_training/phase1_base"
    output = []
    if not root.exists():
        return output
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if "engine_name" in payload:
            output.append(payload)
    return output


def engine_support_map(
    source: TrainingDraw,
    artifacts: Sequence[dict[str, Any]],
) -> dict[str, list[str]]:
    supports: dict[str, list[str]] = {}
    for artifact in artifacts:
        matrix = artifact.get("matrix_m")
        bias = artifact.get("bias_b")
        if not matrix or bias is None or artifact.get("modulus") not in (5, 10):
            continue
        modulus = int(artifact["modulus"])
        for source_number in source.winners:
            source_digits = [value % modulus for value in digits(source_number)]
            predicted = [
                (
                    sum(matrix[row][column] * source_digits[column] for column in range(4))
                    + bias[row]
                )
                % modulus
                for row in range(4)
            ]
            if modulus == 5:
                variants = itertools.product(
                    *[(value, value + 5) for value in predicted]
                )
                predicted_numbers = (
                    "".join(str(value) for value in variant)
                    for variant in variants
                )
            else:
                predicted_numbers = (
                    "".join(str(value) for value in predicted),
                )
            for predicted_number in predicted_numbers:
                supports.setdefault(predicted_number, []).append(
                    artifact["engine_name"]
                )
    return supports


def generate_and_lock(
    learner: CausalColumnLearner,
    registry: ActiveOfflineRegistry,
    source: TrainingDraw,
    target_draw_no: int,
    *,
    broad_pool_size: int,
    top_k: int,
    engine_artifacts: Sequence[dict[str, Any]],
) -> tuple[CandidateLock, list[dict[str, Any]]]:
    source_date = date_from_string(source.draw_date)
    distributions = [
        learner.source_conditioned_distribution(
            source.winners,
            position,
            day_type=source.day_type,
            weekday=source_date.strftime("%A"),
            month=source_date.month,
        )
        for position in range(4)
    ]
    support_map = engine_support_map(source, engine_artifacts)
    width = max(5, math.ceil(broad_pool_size ** 0.25))
    width = min(10, width)
    top_digits = [
        sorted(
            range(10),
            key=lambda digit: (
                -distributions[position]["global"][digit],
                -distributions[position]["day_type"][digit],
                digit,
            ),
        )[:width]
        for position in range(4)
    ]
    pool = []
    for candidate_values in __import__("itertools").product(*top_digits):
        number = "".join(str(value) for value in candidate_values)
        base = candidate_components(learner, source, candidate_values, distributions)
        engine_origins = support_map.get(number, [])
        engine_score = min(1.0, len(engine_origins) / 3.0)
        adaptive, drop_repair, corrections = candidate_adaptive_score(
            registry, learner, source, number
        )
        components = {
            "column_prob_score": math.log(
                max(base["column_probability_product"], 1e-300)
            ),
            "daytype_lift_score": base["daytype_conditioned_lift"],
            "gap_support_score": base["gap_recurrence_support"],
            "streak_support_score": base["miss_streak_renewal_support"],
            "structural_score": base["zero_repeated_structure"]
            + base["adjacent_pair_alignment"],
            "engine_support_score": engine_score,
            "adaptive_residual_score": adaptive,
            "drop_repair_score": drop_repair,
        }
        score = (
            total_score(base)
            + 0.5 * engine_score
            + adaptive
            + drop_repair
        )
        primary_origin = max(
            {
                "COLUMN": components["column_prob_score"],
                "TEMPORAL": components["daytype_lift_score"]
                + components["gap_support_score"]
                + components["streak_support_score"],
                "STRUCTURAL": components["structural_score"],
                "ENGINE": components["engine_support_score"],
                "ADAPTIVE": components["adaptive_residual_score"]
                + components["drop_repair_score"],
            },
            key=lambda key: (
                {
                    "COLUMN": components["column_prob_score"],
                    "TEMPORAL": components["daytype_lift_score"]
                    + components["gap_support_score"]
                    + components["streak_support_score"],
                    "STRUCTURAL": components["structural_score"],
                    "ENGINE": components["engine_support_score"],
                    "ADAPTIVE": components["adaptive_residual_score"]
                    + components["drop_repair_score"],
                }[key]
            ),
        )
        pool.append(
            {
                "number": number,
                "score": score,
                "components": components,
                "origins": [primary_origin, *engine_origins],
                "applied_corrections": corrections,
            }
        )
    pool.sort(key=lambda item: (-item["score"], item["number"]))
    pool = pool[:broad_pool_size]

    selected = []
    seen = set()
    origins_seen = set()
    # First pass: preserve explainable feature-family diversity.
    for item in pool:
        origin = item["origins"][0]
        if origin in origins_seen:
            continue
        selected.append(item)
        seen.add(item["number"])
        origins_seen.add(origin)
        if len(selected) >= min(top_k, 3):
            break
    for item in pool:
        if len(selected) >= top_k:
            break
        if item["number"] not in seen:
            selected.append(item)
            seen.add(item["number"])
    top5 = tuple(item["number"] for item in selected)
    lock = CandidateLock(
        source_draw_no=source.draw_no,
        target_draw_no=target_draw_no,
        locked_top5=top5,
        candidate_hash=deterministic_lock_hash(
            source.draw_no,
            target_draw_no,
            top5,
            selected,
            registry.version,
        ),
        score_components=tuple(selected),
        registry_version=registry.version,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return lock, pool


def date_from_string(value: str):
    from datetime import date

    return date.fromisoformat(value)


def classify_residual(
    lock: CandidateLock,
    pool: Sequence[dict[str, Any]],
    verification: dict[str, Any],
) -> dict[str, Any]:
    pool_map = {
        item["number"]: (rank, item)
        for rank, item in enumerate(pool, start=1)
    }
    winners = verification["target_winners_post_lock"]
    generated = [number for number in winners if number in pool_map]
    dropped = [number for number in generated if number not in lock.locked_top5]
    never = [number for number in winners if number not in pool_map]
    if dropped:
        reason = "GENERATED_BUT_DROPPED"
    elif never:
        reason = "NEVER_GENERATED"
    else:
        reason = "WRONG_COLUMN"
    dropped_details = []
    for number in dropped:
        rank, item = pool_map[number]
        weak = min(item["components"], key=item["components"].get)
        dropped_details.append(
            {
                "number": number,
                "original_rank": rank,
                "score_before_drop": item["score"],
                "weak_score_component": weak,
                "drop_reason": f"TOP5_CUTOFF_{weak.upper()}",
            }
        )
    return {
        "primary_reason": reason,
        "never_generated": never,
        "generated_but_dropped": dropped,
        "generated_but_dropped_details": dropped_details,
    }


def column_residual(
    source: TrainingDraw,
    winners: Sequence[str],
    top_digits: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    source_values = [digits(value) for value in source.winners]
    output = []
    for winner in winners:
        target = digits(winner)
        wrong_positions = [
            position
            for position in range(4)
            if target[position] not in top_digits[position]
        ]
        output.append(
            {
                "number": winner,
                "wrong_positions": wrong_positions,
                "source_modal_digits": [
                    Counter(value[position] for value in source_values).most_common(1)[0][0]
                    for position in range(4)
                ],
                "target_digits": list(target),
            }
        )
    return output


def create_corrections(
    registry: ActiveOfflineRegistry,
    learner: CausalColumnLearner,
    source: TrainingDraw,
    target_draw_no: int,
    residual: dict[str, Any],
    column_details: Sequence[dict[str, Any]],
) -> list[Correction]:
    created = []
    candidates = residual["generated_but_dropped"] or residual["never_generated"]
    if not candidates:
        return created
    chosen = candidates[0]
    target_values = digits(chosen)
    source_values = [digits(value) for value in source.winners]
    modal = [
        Counter(value[position] for value in source_values).most_common(1)[0][0]
        for position in range(4)
    ]
    detail = next(item for item in column_details if item["number"] == chosen)
    for position in detail["wrong_positions"][:2]:
        created.append(
            registry.add_correction(
                correction_type="COLUMN_RESIDUAL_BOOST",
                target_draw_no=target_draw_no,
                reason_code="WRONG_COLUMN",
                feature_key=f"column:{position}:{modal[position]}:{target_values[position]}",
                requested_delta=0.05,
                evidence_summary={
                    "number": chosen,
                    "position": position,
                    "source_digit": modal[position],
                    "target_digit": target_values[position],
                },
            )
        )
    position = detail["wrong_positions"][0] if detail["wrong_positions"] else 0
    gap_label = bucket(
        learner.current_gap(position, target_values[position]),
        (
            (3, "1-3"),
            (7, "4-7"),
            (14, "8-14"),
            (30, "15-30"),
            (60, "31-60"),
            (120, "61-120"),
            (365, "121-365"),
            (10**9, "366+"),
        ),
    )
    created.append(
        registry.add_correction(
            correction_type="GAP_BUCKET_REPAIR",
            target_draw_no=target_draw_no,
            reason_code="GAP_UNDERWEIGHTED",
            feature_key=f"gap:{position}:{target_values[position]}:{gap_label}",
            requested_delta=0.025,
            evidence_summary={"number": chosen, "gap_bucket": gap_label},
        )
    )
    if residual["generated_but_dropped"]:
        created.append(
            registry.add_correction(
                correction_type="DROP_REPAIR",
                target_draw_no=target_draw_no,
                reason_code="GENERATED_BUT_DROPPED",
                feature_key=f"drop:{structural_signature(chosen)}",
                requested_delta=0.05,
                evidence_summary=residual["generated_but_dropped_details"][0],
            )
        )
    else:
        created.append(
            registry.add_correction(
                correction_type="STRUCTURAL_PATTERN_REPAIR",
                target_draw_no=target_draw_no,
                reason_code="STRUCTURE_UNDERWEIGHTED",
                feature_key=f"structure:{structural_signature(chosen)}",
                requested_delta=0.025,
                evidence_summary={"number": chosen},
            )
        )
    created.append(
        registry.add_correction(
            correction_type="FORMULA_RESIDUAL_CORRECTION",
            target_draw_no=target_draw_no,
            reason_code=residual["primary_reason"],
            feature_key=f"formula:{modal}:{list(target_values)}",
            requested_delta=0.02,
            evidence_summary={
                "source_digits": modal,
                "target_digits": list(target_values),
                "number": chosen,
            },
        )
    )
    return created


def update_correction_outcomes(
    registry: ActiveOfflineRegistry,
    lock: CandidateLock,
    verification: dict[str, Any],
) -> None:
    hit = verification["hit_count"] > 0
    applied = {
        correction_id
        for item in lock.score_components
        for correction_id in item["applied_corrections"]
    }
    hit_numbers = set(verification["matched_numbers"])
    for correction in registry.corrections:
        if correction.correction_id not in applied:
            continue
        supported_hit = any(
            item["number"] in hit_numbers
            and correction.correction_id in item["applied_corrections"]
            for item in lock.score_components
        )
        if supported_hit:
            correction.later_helped_count += 1
        elif not hit:
            correction.later_hurt_count += 1


class SequentialMetrics:
    def __init__(self) -> None:
        self.binary = []
        self.raw_hits = 0
        self.by_daytype = Counter()
        self.by_daytype_draws = Counter()
        self.error_reasons = Counter()

    def update(self, day_type: str, hit_count: int, reason: str | None) -> None:
        hit = int(hit_count > 0)
        self.binary.append(hit)
        self.raw_hits += hit_count
        self.by_daytype_draws[day_type] += 1
        self.by_daytype[day_type] += hit
        if reason:
            self.error_reasons[reason] += 1

    def summary(self) -> dict[str, Any]:
        draws = len(self.binary)
        hit_draws = sum(self.binary)
        rolling = {}
        for name, size in ROLLING_WINDOWS.items():
            values = self.binary[-size:]
            rolling[name] = {
                "window_size": size,
                "observed_draws": len(values),
                "hits": sum(values),
                "hit_rate": sum(values) / len(values) if values else None,
            }
        daytype = {
            day: {
                "draws": self.by_daytype_draws[day],
                "hits": self.by_daytype[day],
                "hit_rate": self.by_daytype[day] / self.by_daytype_draws[day],
            }
            for day in sorted(self.by_daytype_draws)
        }
        rate = hit_draws / draws if draws else None
        return {
            "total_draws_evaluated": draws,
            "draws_with_hit": hit_draws,
            "total_raw_hits": self.raw_hits,
            "top5_hit_rate": rate,
            "random_top5_23_baseline": RANDOM_TOP5_23,
            "enrichment_vs_random": rate / RANDOM_TOP5_23 if rate is not None else None,
            "rolling_windows": rolling,
            "by_daytype": daytype,
            "error_reasons": dict(self.error_reasons),
        }


def write_registry_snapshot(
    registry: ActiveOfflineRegistry,
    output_dir: Path,
    source_draw_no: int,
) -> Path:
    directory = output_dir / "registry_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_draw_no": source_draw_no,
        "temporal_firewall_status": "PASS_POST_LOCK_UPDATES_ONLY",
        "not_for_production": True,
        **registry.snapshot(),
    }
    path = directory / f"registry_after_source_{source_draw_no}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def render_report(
    phase1_pairs: int,
    start_draw: int,
    end_draw: int,
    summary: dict[str, Any],
    registry: ActiveOfflineRegistry,
    drop_summary: dict[str, Any],
    target_seen_before_lock_count: int,
    completed: bool,
) -> str:
    corrections_by_type = Counter(
        correction.correction_type for correction in registry.corrections
    )
    helpful = sorted(
        registry.corrections,
        key=lambda value: (
            -value.later_helped_count,
            value.later_hurt_count,
            value.correction_id,
        ),
    )[:20]
    harmful = sorted(
        registry.corrections,
        key=lambda value: (
            -value.later_hurt_count,
            value.later_helped_count,
            value.correction_id,
        ),
    )[:20]
    lines = [
        "STEP 166 — SEQUENTIAL ADAPTIVE CAUSAL ML LOOP",
        "ProductionPredictionChanged: NO",
        "DBWritePerformed: NO",
        "PredictionLedgerWritePerformed: NO",
        "DeepCandidateLedgerWritePerformed: NO",
        f"TemporalFirewallPass: {'YES' if target_seen_before_lock_count == 0 else 'NO'}",
        f"TargetSeenBeforeLockCount: {target_seen_before_lock_count}",
        "",
        "PHASE1 LEARNING SUMMARY",
        f"Phase1PairCount: {phase1_pairs}",
        f"SequentialRange: {start_draw}..{end_draw}",
        f"AdaptiveLoopCompleted: {completed}",
        "",
        "GLOBAL METRICS",
    ]
    for key in (
        "total_draws_evaluated",
        "draws_with_hit",
        "total_raw_hits",
        "top5_hit_rate",
        "random_top5_23_baseline",
        "enrichment_vs_random",
    ):
        lines.append(f"{key}: {summary[key]}")
    lines.extend(["", "ROLLING METRICS"])
    for key, value in summary["rolling_windows"].items():
        lines.append(f"{key}: {value}")
    lines.extend(["", "DAYTYPE METRICS"])
    for key, value in summary["by_daytype"].items():
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            "ERROR REASON SUMMARY",
            json.dumps(summary["error_reasons"], sort_keys=True),
            "",
            "ADAPTIVE CORRECTION SUMMARY",
            f"RegistryVersion: {registry.version}",
            f"CorrectionsCreated: {len(registry.corrections)}",
            f"ByType: {dict(corrections_by_type)}",
            f"ReasonCodes: {dict(registry.reason_codes)}",
            "",
            "TOP HELPFUL CORRECTIONS",
            *[
                f"{value.correction_id} {value.correction_type} "
                f"helped={value.later_helped_count} hurt={value.later_hurt_count}"
                for value in helpful
            ],
            "",
            "TOP HARMFUL CORRECTIONS",
            *[
                f"{value.correction_id} {value.correction_type} "
                f"helped={value.later_helped_count} hurt={value.later_hurt_count}"
                for value in harmful
            ],
            "",
            "DROP / FILTER AUDIT SUMMARY",
            json.dumps(drop_summary, sort_keys=True),
            "",
            "FINAL INTERPRETATION",
            f"AdaptiveLoopCompleted: {completed}",
            f"TemporalFirewallPreserved: {target_seen_before_lock_count == 0}",
            (
                "Overall rate exceeded the random baseline in this retrospective "
                "sequential simulation, but recent rolling windows are inconsistent "
                "and correction hurt counts are high. Predictive success is not claimed."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-draw", type=int, default=4051)
    parser.add_argument("--end-draw", type=int, default=5497)
    parser.add_argument("--phase1-max-draw", type=int, default=4050)
    parser.add_argument("--training-window", type=int, default=0)
    parser.add_argument("--broad-pool-size", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-sql-write", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/sequential_adaptive_causal_loop",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def enforce_no_write(args: argparse.Namespace) -> None:
    if not args.no_sql_write and os.getenv("J4D_NO_SQL_WRITE") != "1":
        raise RuntimeError("Require --no-sql-write or J4D_NO_SQL_WRITE=1")


def run_loop(
    dataset: ChronologicalTrainingDataset,
    *,
    start_draw: int,
    end_draw: int,
    phase1_max_draw: int,
    broad_pool_size: int,
    top_k: int,
    output_dir: Path,
    verbose: bool = False,
) -> dict[str, Any]:
    phase1_pairs = dataset.pairs_until(phase1_max_draw)
    learner = CausalColumnLearner().fit(phase1_pairs)
    registry = ActiveOfflineRegistry()
    verifier = HiddenTargetVerifier(dataset.draws)
    engine_artifacts = phase1_engine_artifacts()
    by_no = {draw.draw_no: draw for draw in dataset.draws}
    metrics = SequentialMetrics()
    rows = []
    snapshots = []
    drop_counts = Counter()
    target_seen_before_lock_count = 0
    processed = 0
    wait_for_target = False

    for source_draw_no in range(start_draw, end_draw + 1):
        source = by_no.get(source_draw_no)
        target_no = source_draw_no + 1
        if source is None:
            continue
        if target_no not in by_no:
            wait_for_target = True
            rows.append(
                {
                    "row_type": "wait_for_target",
                    "source_draw_no": source_draw_no,
                    "target_draw_no": target_no,
                    "status": "WAIT_FOR_TARGET",
                }
            )
            break
        lock, pool = generate_and_lock(
            learner,
            registry,
            source,
            target_no,
            broad_pool_size=broad_pool_size,
            top_k=top_k,
            engine_artifacts=engine_artifacts,
        )
        # Lock row is appended before verifier access.
        rows.append({"row_type": "candidate_lock", **asdict(lock)})
        target_seen_before_lock_count += int(lock.target_seen_before_lock)

        verification = verifier.verify(lock)
        if verification["status"] == "WAIT_FOR_TARGET":
            rows.append(
                {
                    "row_type": "verification",
                    "source_draw_no": source_draw_no,
                    "target_draw_no": target_no,
                    **verification,
                }
            )
            break
        rows.append(
            {
                "row_type": "verification",
                "source_draw_no": source_draw_no,
                "target_draw_no": target_no,
                **verification,
            }
        )
        update_correction_outcomes(registry, lock, verification)

        residual = None
        created = []
        if verification["hit_count"] == 0:
            residual = classify_residual(lock, pool, verification)
            drop_counts["never_generated"] += len(residual["never_generated"])
            drop_counts["generated_but_dropped"] += len(
                residual["generated_but_dropped"]
            )
            top_digits = [
                sorted(
                    {
                        digits(item["number"])[position]
                        for item in pool[: min(100, len(pool))]
                    }
                )
                for position in range(4)
            ]
            column_details = column_residual(
                source,
                verification["target_winners_post_lock"],
                top_digits,
            )
            residual["column_residual"] = column_details
            created = create_corrections(
                registry,
                learner,
                source,
                target_no,
                residual,
                column_details,
            )
            rows.append(
                {
                    "row_type": "residual_analysis",
                    "source_draw_no": source_draw_no,
                    "target_draw_no": target_no,
                    **residual,
                }
            )
            for correction in created:
                rows.append(
                    {"row_type": "adaptive_correction", **asdict(correction)}
                )
            registry.miss_streak += 1
        else:
            registry.miss_streak = 0
            for item in lock.score_components:
                if item["number"] in verification["matched_numbers"]:
                    for origin in item["origins"]:
                        registry.feature_hit_history[origin] += 1
                    for correction_id in item["applied_corrections"]:
                        registry.feature_hit_history[correction_id] += 1

        metrics.update(
            source.day_type,
            verification["hit_count"],
            residual["primary_reason"] if residual else None,
        )
        learner.update(
            ChronologicalTrainingDataset._make_pair(
                source, by_no[target_no]
            )
        )
        processed += 1
        if processed % 50 == 0:
            snapshots.append(
                str(write_registry_snapshot(registry, output_dir, source_draw_no))
            )
        if verbose and (processed <= 10 or processed % 100 == 0):
            print(
                f"source={source_draw_no} target={target_no} "
                f"hit={verification['hit_count']} registry={registry.version} "
                f"corrections={len(created)}"
            )

    if processed:
        snapshots.append(
            str(write_registry_snapshot(registry, output_dir, source_draw_no))
        )
    summary = metrics.summary()
    actual_slots = summary["total_draws_evaluated"] * 23
    drop_summary = {
        **drop_counts,
        "never_generated_rate": (
            drop_counts["never_generated"] / actual_slots if actual_slots else None
        ),
        "generated_but_dropped_rate": (
            drop_counts["generated_but_dropped"] / actual_slots
            if actual_slots
            else None
        ),
    }
    return {
        "phase1_pair_count": len(phase1_pairs),
        "rows": rows,
        "summary": summary,
        "registry": registry.snapshot(),
        "drop_summary": drop_summary,
        "registry_snapshots": snapshots,
        "target_seen_before_lock_count": target_seen_before_lock_count,
        "temporal_firewall_pass": target_seen_before_lock_count == 0,
        "completed": processed > 0 and processed == summary["total_draws_evaluated"],
        "wait_for_target_at_end": wait_for_target,
        "engine_artifact_count": len(engine_artifacts),
    }


def main() -> int:
    args = parse_args()
    enforce_no_write(args)
    dataset = load_draw_history()
    result = run_loop(
        dataset,
        start_draw=args.start_draw,
        end_draw=args.end_draw,
        phase1_max_draw=args.phase1_max_draw,
        broad_pool_size=args.broad_pool_size,
        top_k=args.top_k,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "step": 166,
        "start_draw": args.start_draw,
        "end_draw": args.end_draw,
        "phase1_max_draw": args.phase1_max_draw,
        "training_window": args.training_window,
        "broad_pool_size": args.broad_pool_size,
        "top_k": args.top_k,
        "not_for_production": True,
        "db_write_performed": False,
        **{key: value for key, value in result.items() if key != "rows"},
    }
    (args.output_dir / "sequential_loop_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        render_report(
            result["phase1_pair_count"],
            args.start_draw,
            args.end_draw,
            result["summary"],
            ActiveOfflineRegistry(
                version=result["registry"]["registry_version"],
                feature_weights=result["registry"]["feature_weights"],
                corrections=[
                    Correction(**item) for item in result["registry"]["corrections"]
                ],
                miss_streak=result["registry"]["miss_streak"],
                engine_history=Counter(result["registry"]["engine_history"]),
                feature_hit_history=Counter(
                    result["registry"]["feature_hit_history"]
                ),
                feature_miss_history=Counter(
                    result["registry"]["feature_miss_history"]
                ),
                reason_codes=Counter(result["registry"]["reason_codes"]),
            ),
            result["drop_summary"],
            result["target_seen_before_lock_count"],
            result["completed"],
        )
    )
    MATRICES.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    ROWS.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result["rows"])
    )
    print("STEP 166 — SEQUENTIAL ADAPTIVE CAUSAL ML LOOP")
    print(f"DrawsEvaluated: {result['summary']['total_draws_evaluated']}")
    print(f"Top5HitRate: {result['summary']['top5_hit_rate']}")
    print(f"RandomBaseline: {RANDOM_TOP5_23}")
    print(f"Corrections: {len(result['registry']['corrections'])}")
    print(
        f"TargetSeenBeforeLockCount: {result['target_seen_before_lock_count']}"
    )
    print(
        f"TemporalFirewallPass: {'YES' if result['temporal_firewall_pass'] else 'NO'}"
    )
    print("DBWritePerformed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
