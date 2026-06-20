from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyodbc
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
REPORT_PATH = PROJECT_ROOT / "reports" / "step_153c_forward_top50_persistence_report.txt"
BATCH_PATH = PROJECT_ROOT / "reports" / "step_153c_forward_top50_persistence_batch.json"
ROWS_PATH = PROJECT_ROOT / "reports" / "step_153c_forward_top50_persistence_rows.jsonl"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings
from scripts.shadow_hybrid_engine_v2 import generate_shadow_hybrid_v2_top5


V2_ENGINE = "E_SHADOW_HYBRID_GUARD_V2_BASELINE_PROTECTED"
MODE = "ForwardTop50"
TOP50_DEPTH = 50
MAX_SYNTHETIC = 15
ENGINE_SPECS = (
    {
        "engine_source": V2_ENGINE,
        "engine_version": "v2.0",
        "persisted_alias": None,
        "required_depth": 50,
    },
    {
        "engine_source": "E1_TEMPORAL_CONTEXT_MATCH",
        "engine_version": "persisted-ledger-anchor",
        "persisted_alias": "E1_TEMPORAL_CONTEXT_MATCH",
        "required_depth": None,
    },
    {
        "engine_source": "E1_DELTA",
        "engine_version": "safe-reconstructed-preview",
        "persisted_alias": "E1_DELTA_ROTATION_LSTS",
        "required_depth": None,
    },
    {
        "engine_source": "E1_WLS",
        "engine_version": "safe-reconstructed-preview",
        "persisted_alias": "E1_WLS_DECAY_0.98",
        "required_depth": None,
    },
    {
        "engine_source": "E1_LINEAR",
        "engine_version": "safe-reconstructed-preview",
        "persisted_alias": "E1_CROSS_PAIR_LINEAR",
        "required_depth": None,
    },
)


@dataclass(frozen=True)
class Candidate:
    number: str
    score: float
    family: str
    generation_method: str
    reason: str
    supports: tuple[str, ...]
    source_draw_no: int
    source_engine: str | None
    source_rank: int | None
    synthetic: bool = False


def get_conn():
    return pyodbc.connect(
        get_settings().sql_connection_string(),
        timeout=120,
        autocommit=False,
    )


def z4(value: str | int) -> str:
    return str(value).strip().zfill(4)


def parse_numbers(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(z4(part) for part in str(value).replace(" ", "").split(",") if part)


def digits(number: str) -> tuple[int, ...]:
    return tuple(int(ch) for ch in z4(number))


def digit_sum(number: str) -> int:
    return sum(digits(number))


def mirror_signature(number: str) -> str:
    return "".join(str(value % 5) for value in digits(number))


def box_signature(number: str) -> str:
    return "".join(sorted(z4(number)))


def repeated_digits(number: str) -> int:
    number = z4(number)
    return len(number) - len(set(number))


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def fetch_latest_source(cursor) -> dict:
    row = cursor.execute(
        """
        SELECT TOP (1)
            DrawNo,
            CONVERT(varchar(10), DrawDate, 120) AS DrawDateText,
            DayType
        FROM dbo.DrawHistory
        WHERE WinningNumbers IS NOT NULL
        ORDER BY DrawNo DESC;
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("DrawHistory is empty")
    source = int(row.DrawNo)
    target_row = cursor.execute(
        """
        SELECT CASE
            WHEN EXISTS (
                SELECT 1 FROM dbo.DrawHistory WHERE DrawNo = ?
            ) THEN 1 ELSE 0
        END AS TargetExists;
        """,
        source + 1,
    ).fetchone()
    return {
        "source_draw_no": source,
        "target_draw_no": source + 1,
        "source_date": str(row.DrawDateText) if row.DrawDateText else None,
        "source_day_type": str(row.DayType or "Unknown"),
        "target_available_at_generation": bool(target_row and int(target_row.TargetExists)),
    }


def fetch_draws_through_source(cursor, source: int) -> dict[int, dict]:
    rows = cursor.execute(
        """
        SELECT
            DrawNo,
            WinningNumbers
        FROM dbo.DrawHistory
        WHERE DrawNo <= ?
          AND WinningNumbers IS NOT NULL
        ORDER BY DrawNo;
        """,
        source,
    ).fetchall()
    return {
        int(row.DrawNo): {
            "draw_no": int(row.DrawNo),
            "winners": parse_numbers(row.WinningNumbers),
        }
        for row in rows
    }


def fetch_ledger_through_source(cursor, source: int) -> list[dict]:
    rows = cursor.execute(
        """
        SELECT
            EngineSource,
            Mode,
            SourceDrawNo,
            TargetDrawNo,
            RankNo,
            PredictedNumber,
            Score
        FROM dbo.PredictionLedger
        WHERE SourceDrawNo <= ?
          AND TargetDrawNo <= ?
        ORDER BY SourceDrawNo, TargetDrawNo, EngineSource, Mode, RankNo;
        """,
        source,
        source + 1,
    ).fetchall()
    return [
        {
            "engine_source": str(row.EngineSource or "UNKNOWN"),
            "mode": str(row.Mode),
            "source_draw_no": int(row.SourceDrawNo),
            "target_draw_no": int(row.TargetDrawNo),
            "rank": int(row.RankNo),
            "number": z4(row.PredictedNumber),
            "score": float(row.Score) if row.Score is not None else 0.0,
        }
        for row in rows
    ]


def prediction_ledger_count(cursor) -> int:
    return int(
        cursor.execute(
            """
            SELECT COUNT_BIG(*) AS TotalRows
            FROM dbo.PredictionLedger;
            """
        ).fetchone().TotalRows
    )


def deep_ledger_count(cursor) -> int:
    return int(
        cursor.execute(
            """
            SELECT COUNT_BIG(*) AS TotalRows
            FROM dbo.DeepCandidateLedger;
            """
        ).fetchone().TotalRows
    )


def pair_cloud(draws: dict[int, dict], source: int, depth: int) -> set[str]:
    output: set[str] = set()
    for draw_no in sorted(no for no in draws if no <= source)[-depth:]:
        for number in draws[draw_no]["winners"]:
            output.update((number[:2], number[2:]))
    return output


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def recent_ranges(draws: dict[int, dict], source: int) -> dict[str, tuple[float, float]]:
    values = []
    for draw_no in sorted(no for no in draws if no <= source)[-5:]:
        values.extend(draws[draw_no]["winners"])
    sums = [digit_sum(number) for number in values]
    odd = [sum(value % 2 for value in digits(number)) / 4 for number in values]
    high = [sum(value >= 5 for value in digits(number)) / 4 for number in values]
    return {
        "sum": (max(4.0, percentile(sums, 0.10) - 2), min(32.0, percentile(sums, 0.90) + 2)),
        "odd": (max(0.0, percentile(odd, 0.10) - 0.25), min(1.0, percentile(odd, 0.90) + 0.25)),
        "high": (max(0.0, percentile(high, 0.10) - 0.25), min(1.0, percentile(high, 0.90) + 0.25)),
    }


def prior_hit_shapes(
    ledger: list[dict],
    draws: dict[int, dict],
    source: int,
) -> tuple[Counter[int], Counter[str], Counter[str]]:
    sums: Counter[int] = Counter()
    mirrors: Counter[str] = Counter()
    boxes: Counter[str] = Counter()
    for item in ledger:
        if item["target_draw_no"] > source:
            continue
        target = draws.get(item["target_draw_no"])
        if target is None or item["number"] not in set(target["winners"]):
            continue
        sums[digit_sum(item["number"])] += 1
        mirrors[mirror_signature(item["number"])] += 1
        boxes[box_signature(item["number"])] += 1
    return sums, mirrors, boxes


def controlled_repairs(top5: list[str], source: int) -> list[Candidate]:
    output: dict[str, Candidate] = {}
    for base_rank, base in enumerate(top5, start=1):
        for position in range(4):
            chars = list(base)
            chars[position] = str((int(chars[position]) + 5) % 10)
            number = "".join(chars)
            output.setdefault(
                number,
                Candidate(
                    number=number,
                    score=54.0 - base_rank - position * 0.1,
                    family="MIRROR_BOX_REPAIR",
                    generation_method="CONTROLLED_SINGLE_MIRROR_FLIP",
                    reason=f"single mirror flip position {position + 1} of locked V2 rank {base_rank}",
                    supports=("MIRROR_BOX_REPAIR",),
                    source_draw_no=source,
                    source_engine=V2_ENGINE,
                    source_rank=base_rank,
                ),
            )
        reversed_number = base[::-1]
        if reversed_number != base:
            output.setdefault(
                reversed_number,
                Candidate(
                    number=reversed_number,
                    score=50.0 - base_rank,
                    family="MIRROR_BOX_REPAIR",
                    generation_method="CONTROLLED_BOX_REVERSAL",
                    reason=f"bounded reversal of locked V2 rank {base_rank}",
                    supports=("MIRROR_BOX_REPAIR",),
                    source_draw_no=source,
                    source_engine=V2_ENGINE,
                    source_rank=base_rank,
                ),
            )
    return sorted(output.values(), key=lambda item: (-item.score, item.number))[:20]


def synthetic_candidates(
    source: int,
    source_numbers: Iterable[str],
    excluded: set[str],
) -> list[Candidate]:
    seed_material = f"{source}|" + ",".join(sorted(source_numbers))
    output = []
    counter = 0
    while len(output) < MAX_SYNTHETIC and counter < 1000:
        digest = hashlib.sha256(f"{seed_material}|{counter}".encode("utf-8")).hexdigest()
        number = f"{int(digest[:12], 16) % 10000:04d}"
        counter += 1
        if number in excluded or any(item.number == number for item in output):
            continue
        output.append(
            Candidate(
                number=number,
                score=10.0 - len(output) * 0.1,
                family="FALLBACK_SAFE_SYNTHETIC",
                generation_method="SOURCE_HASH_DETERMINISTIC_FILL",
                reason="deterministic SHA256 fill derived from source draw winners only",
                supports=("SOURCE_ONLY_DETERMINISTIC",),
                source_draw_no=source,
                source_engine=None,
                source_rank=None,
                synthetic=True,
            )
        )
    return output


def build_v2_top50(
    source: int,
    v2_result: dict,
    draws: dict[int, dict],
    ledger: list[dict],
) -> list[Candidate]:
    locked_top5 = list(v2_result["top5"])
    if len(locked_top5) != 5 or len(set(locked_top5)) != 5:
        raise RuntimeError("Committed V2 engine did not return five unique candidates")

    output = [
        Candidate(
            number=item["number"],
            score=1000.0 - item["rank"],
            family=item["family"],
            generation_method="LOCKED_V2_TOP5",
            reason=item["reason"],
            supports=tuple(item.get("supports", [])),
            source_draw_no=source,
            source_engine=V2_ENGINE,
            source_rank=item["rank"],
        )
        for item in v2_result["candidate_details"]
    ]
    selected = set(locked_top5)
    clouds = {depth: pair_cloud(draws, source, depth) for depth in (1, 3, 5)}
    ranges = recent_ranges(draws, source)
    hit_sums, hit_mirrors, hit_boxes = prior_hit_shapes(ledger, draws, source)

    history: dict[str, dict] = defaultdict(
        lambda: {
            "appearances": 0,
            "latest_source": 0,
            "best_rank": 999,
            "max_score": 0.0,
            "engines": set(),
            "current_source": False,
            "current_rank": None,
        }
    )
    for item in ledger:
        number = item["number"]
        stats = history[number]
        stats["appearances"] += 1
        stats["latest_source"] = max(stats["latest_source"], item["source_draw_no"])
        stats["best_rank"] = min(stats["best_rank"], item["rank"])
        stats["max_score"] = max(stats["max_score"], item["score"])
        stats["engines"].add(item["engine_source"])
        if item["source_draw_no"] == source:
            stats["current_source"] = True
            stats["current_rank"] = (
                item["rank"]
                if stats["current_rank"] is None
                else min(stats["current_rank"], item["rank"])
            )

    ranked_pool: list[Candidate] = []
    for number, stats in history.items():
        if number in selected:
            continue
        values = digits(number)
        total = sum(values)
        first, last = number[:2], number[2:]
        supports = []
        pair_score = 0.0
        if first in clouds[1] or last in clouds[1]:
            supports.append("PAIR_RECURRENCE_1")
            pair_score += 8.0
        if first in clouds[3] or last in clouds[3]:
            supports.append("PAIR_RECURRENCE_3")
            pair_score += 4.0
        if first in clouds[5] or last in clouds[5]:
            supports.append("PAIR_RECURRENCE_5")
            pair_score += 2.0
        structure_ok = (
            ranges["sum"][0] <= total <= ranges["sum"][1]
            and ranges["odd"][0] <= sum(value % 2 for value in values) / 4 <= ranges["odd"][1]
            and ranges["high"][0] <= sum(value >= 5 for value in values) / 4 <= ranges["high"][1]
            and repeated_digits(number) <= 1
        )
        if structure_ok:
            supports.append("RECENT_STRUCTURE")
        shape_support = (
            hit_sums[total]
            + hit_mirrors[mirror_signature(number)]
            + hit_boxes[box_signature(number)]
        )
        if hit_sums[total]:
            supports.append("RISK_SUM_BAND")
        if hit_mirrors[mirror_signature(number)] or hit_boxes[box_signature(number)]:
            supports.append("MIRROR_BOX_HISTORY")

        recency = 1.0 / (1.0 + source - stats["latest_source"])
        current_bonus = (
            35.0 - float(stats["current_rank"] or 5)
            if stats["current_source"]
            else 0.0
        )
        score = (
            60.0
            + current_bonus
            + min(12.0, math.log1p(stats["appearances"]) * 3.0)
            + min(10.0, shape_support * 0.8)
            + pair_score
            + (5.0 if structure_ok else -2.0)
            + recency * 4.0
            + min(3.0, stats["max_score"])
        )
        if stats["current_source"]:
            family = "SAFE_LEDGER_CANDIDATE"
            method = "CURRENT_SOURCE_PERSISTED_ENGINE_CANDIDATE"
        elif pair_score >= 8:
            family = "PAIR_RECURRENCE"
            method = "PRIOR_LEDGER_CANDIDATE_PAIR_SUPPORTED"
        elif hit_sums[total]:
            family = "RISK_SUM_BAND"
            method = "PRIOR_LEDGER_CANDIDATE_SUM_SUPPORTED"
        elif structure_ok:
            family = "RECENT_STRUCTURE"
            method = "PRIOR_LEDGER_CANDIDATE_STRUCTURE_SUPPORTED"
        else:
            family = "SAFE_LEDGER_CANDIDATE"
            method = "PRIOR_PERSISTED_LEDGER_CANDIDATE"
        ranked_pool.append(
            Candidate(
                number=number,
                score=score,
                family=family,
                generation_method=method,
                reason=(
                    f"persisted ledger candidate; appearances={stats['appearances']}; "
                    f"latest_source={stats['latest_source']}; best_rank={stats['best_rank']}"
                ),
                supports=tuple(sorted(set(supports))),
                source_draw_no=int(stats["latest_source"]),
                source_engine=",".join(sorted(stats["engines"])),
                source_rank=(
                    int(stats["current_rank"])
                    if stats["current_rank"] is not None
                    else int(stats["best_rank"])
                ),
            )
        )

    for candidate in sorted(ranked_pool, key=lambda item: (-item.score, item.number)):
        if candidate.number in selected:
            continue
        output.append(candidate)
        selected.add(candidate.number)
        if len(output) == TOP50_DEPTH:
            break

    if len(output) < TOP50_DEPTH:
        for candidate in controlled_repairs(locked_top5, source):
            if candidate.number in selected:
                continue
            output.append(candidate)
            selected.add(candidate.number)
            if len(output) == TOP50_DEPTH:
                break

    if len(output) < TOP50_DEPTH:
        source_numbers = draws[source]["winners"]
        for candidate in synthetic_candidates(source, source_numbers, selected):
            output.append(candidate)
            selected.add(candidate.number)
            if len(output) == TOP50_DEPTH:
                break

    if len(output) != TOP50_DEPTH:
        raise RuntimeError(
            f"V2 cannot safely produce exactly 50 unique candidates; produced {len(output)}"
        )
    synthetic_count = sum(item.synthetic for item in output)
    top20_non_synthetic = sum(not item.synthetic for item in output[:20])
    if synthetic_count > MAX_SYNTHETIC:
        raise RuntimeError(f"Synthetic cap exceeded: {synthetic_count}")
    if top20_non_synthetic < 10:
        raise RuntimeError(
            f"Top20 non-synthetic minimum failed: {top20_non_synthetic}"
        )
    if [item.number for item in output[:5]] != locked_top5:
        raise RuntimeError("V2 Top5 lock was not preserved")
    return output


def current_persisted_batches(
    source: int,
    target: int,
    ledger: list[dict],
) -> dict[str, list[Candidate]]:
    output: dict[str, list[Candidate]] = {}
    for spec in ENGINE_SPECS:
        alias = spec["persisted_alias"]
        if alias is None:
            continue
        items = [
            item
            for item in ledger
            if item["source_draw_no"] == source
            and item["target_draw_no"] == target
            and item["engine_source"] == alias
        ]
        if not items:
            output[spec["engine_source"]] = []
            continue
        mode_priority = {
            "Current": 0,
            "Temporal_Global_Loop": 1,
            "Historical": 2,
            "Engine_Grand_Loop": 3,
        }
        chosen_mode = min(
            {item["mode"] for item in items},
            key=lambda value: mode_priority.get(value, 99),
        )
        ranked = sorted(
            [item for item in items if item["mode"] == chosen_mode],
            key=lambda item: item["rank"],
        )
        output[spec["engine_source"]] = [
            Candidate(
                number=item["number"],
                score=item["score"],
                family="SAFE_LEDGER_CANDIDATE",
                generation_method="COPIED_FROM_LOCKED_PREDICTION_LEDGER_TOP5",
                reason=f"persisted {alias} rank {item['rank']} mode {chosen_mode}",
                supports=("PERSISTED_LEDGER_LOCK",),
                source_draw_no=source,
                source_engine=alias,
                source_rank=item["rank"],
            )
            for item in ranked
        ]
    return output


def feature_json(candidate: Candidate, source: int) -> dict:
    return {
        "reason": candidate.reason,
        "supports": list(candidate.supports),
        "source_candidate_draw_no": candidate.source_draw_no,
        "source_engine": candidate.source_engine,
        "source_rank": candidate.source_rank,
        "synthetic": candidate.synthetic,
        "temporal_cutoff_draw_no": source,
    }


def build_batch(
    *,
    engine_source: str,
    engine_version: str,
    source: int,
    target: int,
    target_available: bool,
    candidates: list[Candidate],
) -> dict:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        rows.append(
            {
                "candidate_rank": rank,
                "candidate_number": candidate.number,
                "candidate_score": round(float(candidate.score), 12),
                "candidate_family": candidate.family,
                "generation_method": candidate.generation_method,
                "feature_json": feature_json(candidate, source),
            }
        )
    payload = {
        "engine_source": engine_source,
        "engine_version": engine_version,
        "mode": MODE,
        "source_draw_no": source,
        "target_draw_no": target,
        "temporal_cutoff_draw_no": source,
        "rows": rows,
    }
    batch_hash = sha256(payload)
    output_rows = []
    for row in rows:
        hash_payload = {
            "engine_source": engine_source,
            "engine_version": engine_version,
            "mode": MODE,
            "source_draw_no": source,
            "target_draw_no": target,
            **row,
            "candidate_batch_hash": batch_hash,
            "temporal_cutoff_draw_no": source,
            "target_available_at_generation": target_available,
            "verification_status": "Unverified",
            "hit_count": None,
        }
        output_rows.append(
            {
                **hash_payload,
                "candidate_row_hash": sha256(hash_payload),
            }
        )
    return {
        "engine_source": engine_source,
        "engine_version": engine_version,
        "mode": MODE,
        "source_draw_no": source,
        "target_draw_no": target,
        "candidate_batch_hash": batch_hash,
        "canonical_batch_payload": payload,
        "rows": output_rows,
    }


def fetch_existing_batch(cursor, batch: dict) -> list[dict]:
    rows = cursor.execute(
        """
        SELECT
            EngineSource,
            EngineVersion,
            Mode,
            SourceDrawNo,
            TargetDrawNo,
            CandidateRank,
            CandidateNumber,
            CandidateScore,
            CandidateFamily,
            GenerationMethod,
            FeatureJson,
            CandidateBatchHash,
            CandidateRowHash,
            TemporalCutoffDrawNo,
            TargetAvailableAtGeneration,
            VerificationStatus,
            HitCount
        FROM dbo.DeepCandidateLedger
        WHERE EngineSource = ?
          AND EngineVersion = ?
          AND Mode = ?
          AND SourceDrawNo = ?
          AND TargetDrawNo = ?
        ORDER BY CandidateRank;
        """,
        batch["engine_source"],
        batch["engine_version"],
        batch["mode"],
        batch["source_draw_no"],
        batch["target_draw_no"],
    ).fetchall()
    return [
        {
            "engine_source": str(row.EngineSource),
            "engine_version": str(row.EngineVersion),
            "mode": str(row.Mode),
            "source_draw_no": int(row.SourceDrawNo),
            "target_draw_no": int(row.TargetDrawNo),
            "candidate_rank": int(row.CandidateRank),
            "candidate_number": str(row.CandidateNumber),
            "candidate_score": float(row.CandidateScore)
            if row.CandidateScore is not None
            else None,
            "candidate_family": str(row.CandidateFamily)
            if row.CandidateFamily is not None
            else None,
            "generation_method": str(row.GenerationMethod)
            if row.GenerationMethod is not None
            else None,
            "feature_json": json.loads(row.FeatureJson) if row.FeatureJson else None,
            "candidate_batch_hash": str(row.CandidateBatchHash),
            "candidate_row_hash": str(row.CandidateRowHash),
            "temporal_cutoff_draw_no": int(row.TemporalCutoffDrawNo),
            "target_available_at_generation": bool(row.TargetAvailableAtGeneration),
            "verification_status": str(row.VerificationStatus),
            "hit_count": int(row.HitCount) if row.HitCount is not None else None,
        }
        for row in rows
    ]


def insert_batch(cursor, batch: dict) -> None:
    parameters = [
        (
            row["engine_source"],
            row["engine_version"],
            row["mode"],
            row["source_draw_no"],
            row["target_draw_no"],
            row["candidate_rank"],
            row["candidate_number"],
            row["candidate_score"],
            row["candidate_family"],
            row["generation_method"],
            canonical_json(row["feature_json"]),
            row["candidate_batch_hash"],
            row["candidate_row_hash"],
            row["temporal_cutoff_draw_no"],
            int(row["target_available_at_generation"]),
            row["verification_status"],
            row["hit_count"],
        )
        for row in batch["rows"]
    ]
    cursor.executemany(
        """
        INSERT INTO dbo.DeepCandidateLedger
        (
            EngineSource,
            EngineVersion,
            Mode,
            SourceDrawNo,
            TargetDrawNo,
            CandidateRank,
            CandidateNumber,
            CandidateScore,
            CandidateFamily,
            GenerationMethod,
            FeatureJson,
            CandidateBatchHash,
            CandidateRowHash,
            TemporalCutoffDrawNo,
            TargetAvailableAtGeneration,
            VerificationStatus,
            HitCount
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        parameters,
    )


def validate_rows(rows: list[dict], expected_batch: dict) -> list[str]:
    failures = []
    expected_count = len(expected_batch["rows"])
    if len(rows) != expected_count:
        failures.append(f"ROW_COUNT_EXPECTED_{expected_count}_ACTUAL_{len(rows)}")
    ranks = [row["candidate_rank"] for row in rows]
    if ranks != list(range(1, len(rows) + 1)):
        failures.append("RANKS_NOT_CONTIGUOUS")
    numbers = [row["candidate_number"] for row in rows]
    if len(numbers) != len(set(numbers)):
        failures.append("DUPLICATE_CANDIDATE_NUMBER")
    if any(len(number) != 4 or not number.isdigit() for number in numbers):
        failures.append("INVALID_CANDIDATE_NUMBER")
    if any(
        row["temporal_cutoff_draw_no"] != row["source_draw_no"] for row in rows
    ):
        failures.append("TEMPORAL_CUTOFF_MISMATCH")
    if any(row["verification_status"] != "Unverified" for row in rows):
        failures.append("VERIFICATION_STATUS_NOT_UNVERIFIED")
    if any(row["hit_count"] is not None for row in rows):
        failures.append("HITCOUNT_NOT_NULL")
    batch_hashes = {row["candidate_batch_hash"] for row in rows}
    if batch_hashes != {expected_batch["candidate_batch_hash"]}:
        failures.append("BATCH_HASH_MISMATCH")
    for row in rows:
        hash_payload = {
            key: row[key]
            for key in (
                "engine_source",
                "engine_version",
                "mode",
                "source_draw_no",
                "target_draw_no",
                "candidate_rank",
                "candidate_number",
                "candidate_score",
                "candidate_family",
                "generation_method",
                "feature_json",
                "candidate_batch_hash",
                "temporal_cutoff_draw_no",
                "target_available_at_generation",
                "verification_status",
                "hit_count",
            )
        }
        if sha256(hash_payload) != row["candidate_row_hash"]:
            failures.append(f"ROW_HASH_MISMATCH_RANK_{row['candidate_rank']}")
    return failures


def persist_batches(
    connection,
    batches: list[dict],
    prediction_count_before: int,
) -> tuple[list[dict], bool]:
    cursor = connection.cursor()
    statuses = []
    write_performed = False
    for batch in batches:
        existing = fetch_existing_batch(cursor, batch)
        if existing:
            hashes = {row["candidate_batch_hash"] for row in existing}
            if hashes != {batch["candidate_batch_hash"]}:
                raise RuntimeError(
                    f"BATCH_HASH_MISMATCH for {batch['engine_source']}: "
                    f"existing={sorted(hashes)} computed={batch['candidate_batch_hash']}"
                )
            failures = validate_rows(existing, batch)
            if failures:
                raise RuntimeError(
                    f"Existing batch validation failed for {batch['engine_source']}: {failures}"
                )
            statuses.append(
                {
                    "engine_source": batch["engine_source"],
                    "engine_version": batch["engine_version"],
                    "mode": batch["mode"],
                    "rows_intended": len(batch["rows"]),
                    "rows_inserted": 0,
                    "rows_existing": len(existing),
                    "candidate_batch_hash": batch["candidate_batch_hash"],
                    "status": "ALREADY_EXISTS_MATCH",
                    "validation_failures": [],
                }
            )
            continue

        insert_batch(cursor, batch)
        write_performed = True
        inserted = fetch_existing_batch(cursor, batch)
        failures = validate_rows(inserted, batch)
        if failures:
            raise RuntimeError(
                f"Inserted batch validation failed for {batch['engine_source']}: {failures}"
            )
        statuses.append(
            {
                "engine_source": batch["engine_source"],
                "engine_version": batch["engine_version"],
                "mode": batch["mode"],
                "rows_intended": len(batch["rows"]),
                "rows_inserted": len(inserted),
                "rows_existing": 0,
                "candidate_batch_hash": batch["candidate_batch_hash"],
                "status": "INSERTED",
                "validation_failures": [],
            }
        )

    prediction_count_after = prediction_ledger_count(cursor)
    if prediction_count_after != prediction_count_before:
        raise RuntimeError(
            f"PredictionLedger row count changed: before={prediction_count_before} "
            f"after={prediction_count_after}"
        )
    return statuses, write_performed


def build_report(
    metadata: dict,
    batches: list[dict],
    statuses: list[dict],
    unavailable: list[dict],
    v2_top5: list[str],
    validation: dict,
    write_performed: bool,
    prediction_count_before: int,
    prediction_count_after: int,
) -> str:
    width = 166
    status_by_engine = {item["engine_source"]: item for item in statuses}
    v2_batch = next(item for item in batches if item["engine_source"] == V2_ENGINE)
    lines = [
        "=" * width,
        "STEP 153C — FORWARD TOP50 CANDIDATE PERSISTENCE",
        "=" * width,
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "DeploymentChanged: NO",
        "PredictionLedgerWritePerformed: NO",
        f"DeepCandidateLedgerWritePerformed: {'YES' if write_performed else 'NO'}",
        "",
        "SOURCE / TARGET",
        "-" * width,
        f"SourceDrawNo: {metadata['source_draw_no']}",
        f"TargetDrawNo: {metadata['target_draw_no']}",
        f"SourceDate: {metadata['source_date']}",
        f"SourceDayType: {metadata['source_day_type']}",
        f"TargetAvailableAtGeneration: {'YES' if metadata['target_available_at_generation'] else 'NO'}",
        "",
        "BATCH SUMMARY",
        "-" * width,
        "EngineSource                                      Version                       Mode          Intended Inserted Existing Status                 BatchHash",
    ]
    for spec in ENGINE_SPECS:
        engine = spec["engine_source"]
        status = status_by_engine.get(engine)
        if status is None:
            unavailable_item = next(
                item for item in unavailable if item["engine_source"] == engine
            )
            lines.append(
                f"{engine:<49} {spec['engine_version']:<29} {MODE:<13} "
                f"{0:>8} {0:>8} {0:>8} {'UNAVAILABLE':<22} {unavailable_item['limitation']}"
            )
        else:
            lines.append(
                f"{engine:<49} {status['engine_version']:<29} {status['mode']:<13} "
                f"{status['rows_intended']:>8} {status['rows_inserted']:>8} "
                f"{status['rows_existing']:>8} {status['status']:<22} "
                f"{status['candidate_batch_hash']}"
            )

    lines.extend(
        (
            "",
            "V2 TOP50 PREVIEW",
            "-" * width,
            "Rank Number Family                    Score        ShortReason",
        )
    )
    for row in v2_batch["rows"]:
        reason = str(row["feature_json"]["reason"])
        lines.append(
            f"{row['candidate_rank']:>4} {row['candidate_number']:<6} "
            f"{row['candidate_family']:<25} {row['candidate_score']:>12.6f} "
            f"{reason}"
        )

    lines.extend(
        (
            "",
            "VALIDATION",
            "-" * width,
            f"Top5LockMatch: {validation['top5_lock_match']}",
            f"V2RowCount50: {validation['v2_row_count_50']}",
            f"V2UniqueCandidates: {validation['v2_unique']}",
            f"V2NumericFourDigit: {validation['v2_valid_numbers']}",
            f"Top20NonSyntheticCount: {validation['top20_non_synthetic_count']}",
            f"V2SyntheticCount: {validation['v2_synthetic_count']}",
            f"TemporalCutoffValidation: {validation['temporal_cutoff']}",
            f"BatchHashValidation: {validation['batch_hash']}",
            f"RowHashValidation: {validation['row_hash']}",
            f"IdempotencyStatus: {validation['idempotency_status']}",
            f"PredictionLedgerCountBefore: {prediction_count_before}",
            f"PredictionLedgerCountAfter: {prediction_count_after}",
            f"PredictionLedgerUnchanged: {prediction_count_before == prediction_count_after}",
            "",
            "LIMITATIONS",
            "-" * width,
            "E1_TEMPORAL_CONTEXT_MATCH has no persisted 5497→5498 ledger batch, so no Temporal rows were fabricated.",
            "E1_DELTA, E1_WLS, and E1_LINEAR persist only their locked Top5 because no deeper engine ranks exist.",
            "Only V2 extends to Top50; the extension is source-safe and contains no target-winner input.",
            "",
            "FINAL",
            "-" * width,
            f"ForwardTop50PersistenceReady: {'YES' if validation['all_pass'] else 'NO'}",
            "ProductionSwitchRecommendedNow: NO",
            "NextStep: Step 153D verify persisted Top50 after target result appears",
            "",
            f"REPORT_WRITTEN: {REPORT_PATH}",
            f"BATCH_WRITTEN: {BATCH_PATH}",
            f"ROWS_WRITTEN: {ROWS_PATH}",
        )
    )
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = get_conn()
    statuses: list[dict] = []
    write_performed = False
    try:
        cursor = connection.cursor()
        metadata = fetch_latest_source(cursor)
        source = metadata["source_draw_no"]
        target = metadata["target_draw_no"]
        prediction_count_before = prediction_ledger_count(cursor)
        deep_count_before = deep_ledger_count(cursor)
        draws = fetch_draws_through_source(cursor, source)
        ledger = fetch_ledger_through_source(cursor, source)

        v2_result = generate_shadow_hybrid_v2_top5(source)
        if v2_result["source_draw_no"] != source:
            raise RuntimeError("V2 engine returned a different source draw")
        v2_candidates = build_v2_top50(source, v2_result, draws, ledger)
        persisted = current_persisted_batches(source, target, ledger)

        batches = []
        unavailable = []
        for spec in ENGINE_SPECS:
            engine = spec["engine_source"]
            if engine == V2_ENGINE:
                candidates = v2_candidates
            else:
                candidates = persisted.get(engine, [])
            if not candidates:
                unavailable.append(
                    {
                        "engine_source": engine,
                        "engine_version": spec["engine_version"],
                        "mode": MODE,
                        "limitation": "NO_SAFE_PERSISTED_FORWARD_CANDIDATES",
                    }
                )
                continue
            if spec["required_depth"] is not None and len(candidates) != spec["required_depth"]:
                raise RuntimeError(
                    f"{engine} required depth {spec['required_depth']} but generated {len(candidates)}"
                )
            batches.append(
                build_batch(
                    engine_source=engine,
                    engine_version=spec["engine_version"],
                    source=source,
                    target=target,
                    target_available=metadata["target_available_at_generation"],
                    candidates=candidates,
                )
            )

        v2_batch = next(item for item in batches if item["engine_source"] == V2_ENGINE)
        v2_numbers = [row["candidate_number"] for row in v2_batch["rows"]]
        if v2_numbers[:5] != v2_result["top5"]:
            raise RuntimeError("V2 Top5 lock mismatch before persistence")
        if len(v2_numbers) != 50 or len(set(v2_numbers)) != 50:
            raise RuntimeError("V2 Top50 pre-insert validation failed")

        statuses, write_performed = persist_batches(
            connection,
            batches,
            prediction_count_before,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    with get_conn() as check_connection:
        check_cursor = check_connection.cursor()
        prediction_count_after = prediction_ledger_count(check_cursor)
        deep_count_after = deep_ledger_count(check_cursor)
        persisted_rows = []
        all_validation_failures = []
        for batch in batches:
            existing = fetch_existing_batch(check_cursor, batch)
            failures = validate_rows(existing, batch)
            all_validation_failures.extend(
                f"{batch['engine_source']}:{failure}" for failure in failures
            )
            status = next(
                item for item in statuses if item["engine_source"] == batch["engine_source"]
            )
            for row in existing:
                persisted_rows.append(
                    {
                        **row,
                        "persistence_status": status["status"],
                        "validation_status": "PASS" if not failures else "FAIL",
                    }
                )
        check_connection.rollback()

    v2_rows = [
        row for row in persisted_rows if row["engine_source"] == V2_ENGINE
    ]
    v2_synthetic_count = sum(
        bool((row["feature_json"] or {}).get("synthetic")) for row in v2_rows
    )
    top20_non_synthetic = sum(
        not bool((row["feature_json"] or {}).get("synthetic"))
        for row in v2_rows[:20]
    )
    validation = {
        "top5_lock_match": [row["candidate_number"] for row in v2_rows[:5]]
        == v2_result["top5"],
        "v2_row_count_50": len(v2_rows) == 50,
        "v2_unique": len({row["candidate_number"] for row in v2_rows}) == 50,
        "v2_valid_numbers": all(
            len(row["candidate_number"]) == 4
            and row["candidate_number"].isdigit()
            for row in v2_rows
        ),
        "top20_non_synthetic_count": top20_non_synthetic,
        "v2_synthetic_count": v2_synthetic_count,
        "temporal_cutoff": all(
            row["temporal_cutoff_draw_no"] == row["source_draw_no"]
            for row in persisted_rows
        ),
        "batch_hash": all(
            len({row["candidate_batch_hash"] for row in persisted_rows if row["engine_source"] == batch["engine_source"]}) == 1
            and next(
                row["candidate_batch_hash"]
                for row in persisted_rows
                if row["engine_source"] == batch["engine_source"]
            )
            == batch["candidate_batch_hash"]
            for batch in batches
        ),
        "row_hash": not any(
            "ROW_HASH_MISMATCH" in failure for failure in all_validation_failures
        ),
        "idempotency_status": {
            item["engine_source"]: item["status"] for item in statuses
        },
        "prediction_ledger_unchanged": prediction_count_before
        == prediction_count_after,
        "deep_count_before": deep_count_before,
        "deep_count_after": deep_count_after,
        "validation_failures": all_validation_failures,
    }
    validation["all_pass"] = (
        validation["top5_lock_match"]
        and validation["v2_row_count_50"]
        and validation["v2_unique"]
        and validation["v2_valid_numbers"]
        and validation["top20_non_synthetic_count"] >= 10
        and validation["v2_synthetic_count"] <= MAX_SYNTHETIC
        and validation["temporal_cutoff"]
        and validation["batch_hash"]
        and validation["row_hash"]
        and validation["prediction_ledger_unchanged"]
        and not validation["validation_failures"]
    )
    if not validation["all_pass"]:
        raise RuntimeError(f"Post-commit validation failed: {validation}")

    batch_output = {
        "source_metadata": metadata,
        "batches": batches,
        "unavailable_batches": unavailable,
        "persistence_statuses": statuses,
        "validation": validation,
        "prediction_ledger_count_before": prediction_count_before,
        "prediction_ledger_count_after": prediction_count_after,
    }
    BATCH_PATH.write_text(
        json.dumps(batch_output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with ROWS_PATH.open("w", encoding="utf-8") as handle:
        for row in sorted(
            persisted_rows,
            key=lambda item: (item["engine_source"], item["candidate_rank"]),
        ):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = build_report(
        metadata,
        batches,
        statuses,
        unavailable,
        v2_result["top5"],
        validation,
        write_performed,
        prediction_count_before,
        prediction_count_after,
    )
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
