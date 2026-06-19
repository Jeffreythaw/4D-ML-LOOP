from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import pyodbc
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"

import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings


REPORT_PATH = PROJECT_ROOT / "reports" / "step_150a_master_forensic_audit.txt"

LIVE_CASES = [
    {
        "source": 5494,
        "target": 5495,
        "predicted": ["5941", "9554", "6201", "0000", "7006"],
        "actual": [
            "5478","8081","6943","0136","0376","1350","3478","5827","6189","6812","7436","8039","8665",
            "0625","1007","1545","2122","2241","3671","5055","5315","9378","9949",
        ],
    },
    {
        "source": 5495,
        "target": 5496,
        "predicted": ["8044", "4443", "0550", "1294", "0368"],
        "actual": [
            "4048","6689","6318","0405","1210","3160","4345","4433","6088","8476","9291","9397","9750",
            "0669","0949","2086","2251","3185","4202","7003","7035","8627","9339",
        ],
    },
    {
        "source": 5496,
        "target": 5497,
        "predicted": ["4796", "5024", "9418", "8044", "7452"],
        "actual": [
            "4505","3831","8219","1111","1389","2071","2658","4249","6510","6515","6663","7412","7776",
            "0374","2255","2426","3022","3066","3334","6065","6322","8999","9964",
        ],
    },
]

CORE_ENGINES = [
    "E1_TEMPORAL_CONTEXT_MATCH",
    "E1_DELTA_ROTATION_LSTS",
    "E1_WLS_DECAY_0.98",
    "E1_CROSS_PAIR_LINEAR",
    "E1_MIRROR_BASE5_LSTS",
    "E2_META_ENSEMBLE_RANKER",
    "E2_WEIGHTED_META_ENSEMBLE_RANKER",
]


@dataclass(frozen=True)
class DistanceResult:
    predicted: str
    actual: str
    circular_distance: int
    hamming_distance: int
    digit_sum_delta: int
    first_pair_match: bool
    last_pair_match: bool
    swapped_pair_match: bool
    mirror_exact: bool
    mirror_hamming: int
    box_exact: bool
    box_overlap: int
    box_distance: int


def get_conn():
    settings = get_settings()
    return pyodbc.connect(settings.sql_connection_string(), timeout=120)


def digits(number: str) -> list[int]:
    return [int(ch) for ch in str(number).zfill(4)]


def digit_sum(number: str) -> int:
    return sum(digits(number))


def circular_digit_distance(a: int, b: int) -> int:
    diff = abs(a - b)
    return min(diff, 10 - diff)


def circular_distance(a: str, b: str) -> int:
    return sum(circular_digit_distance(x, y) for x, y in zip(digits(a), digits(b)))


def hamming_distance(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(str(a).zfill(4), str(b).zfill(4)))


def mirror_signature(number: str) -> str:
    return "".join(str(int(ch) % 5) for ch in str(number).zfill(4))


def mirror_hamming_distance(a: str, b: str) -> int:
    return hamming_distance(mirror_signature(a), mirror_signature(b))


def box_overlap(a: str, b: str) -> int:
    ca = Counter(str(a).zfill(4))
    cb = Counter(str(b).zfill(4))
    return sum((ca & cb).values())


def box_distance(a: str, b: str) -> int:
    return 4 - box_overlap(a, b)


def pair_info(a: str, b: str) -> tuple[bool, bool, bool]:
    a = str(a).zfill(4)
    b = str(b).zfill(4)
    first = a[:2] == b[:2]
    last = a[2:] == b[2:]
    swapped = a[:2] == b[2:] or a[2:] == b[:2]
    return first, last, swapped


def compare_pair(predicted: str, actual: str) -> DistanceResult:
    first, last, swapped = pair_info(predicted, actual)
    return DistanceResult(
        predicted=predicted,
        actual=actual,
        circular_distance=circular_distance(predicted, actual),
        hamming_distance=hamming_distance(predicted, actual),
        digit_sum_delta=abs(digit_sum(predicted) - digit_sum(actual)),
        first_pair_match=first,
        last_pair_match=last,
        swapped_pair_match=swapped,
        mirror_exact=mirror_signature(predicted) == mirror_signature(actual),
        mirror_hamming=mirror_hamming_distance(predicted, actual),
        box_exact=sorted(predicted) == sorted(actual),
        box_overlap=box_overlap(predicted, actual),
        box_distance=box_distance(predicted, actual),
    )


def closest_by_metric(predicted: str, actuals: Sequence[str], key) -> DistanceResult:
    results = [compare_pair(predicted, actual) for actual in actuals]
    return sorted(results, key=key)[0]


def fetch_drawhistory_state(cursor) -> list[tuple]:
    rows = cursor.execute("""
        SELECT TOP (10)
            DrawNo,
            DrawDate,
            DayType,
            WinningNumbers
        FROM dbo.DrawHistory
        ORDER BY DrawNo DESC;
    """).fetchall()
    return [tuple(row) for row in rows]


def fetch_draw_range(cursor) -> tuple[int, int, int]:
    row = cursor.execute("""
        SELECT
            MIN(DrawNo) AS MinDrawNo,
            MAX(DrawNo) AS MaxDrawNo,
            COUNT(*) AS TotalRows
        FROM dbo.DrawHistory;
    """).fetchone()
    return int(row.MinDrawNo), int(row.MaxDrawNo), int(row.TotalRows)


def fetch_ledger_range(cursor) -> list[tuple]:
    rows = cursor.execute("""
        SELECT
            Mode,
            EngineSource,
            MIN(SourceDrawNo) AS MinSource,
            MAX(SourceDrawNo) AS MaxSource,
            COUNT(*) AS Rows
        FROM dbo.PredictionLedger
        GROUP BY Mode, EngineSource
        ORDER BY Mode, EngineSource;
    """).fetchall()
    return [tuple(row) for row in rows]


def fetch_live_ledger_group(cursor, source: int, target: int) -> list[tuple]:
    rows = cursor.execute("""
        SELECT
            Mode,
            SourceDrawNo,
            TargetDrawNo,
            RankNo,
            PredictedNumber,
            EngineSource,
            VerificationStatus,
            HitCount
        FROM dbo.PredictionLedger
        WHERE Mode = 'Current'
          AND SourceDrawNo = ?
          AND TargetDrawNo = ?
        ORDER BY RankNo;
    """, source, target).fetchall()
    return [tuple(row) for row in rows]


def fetch_candidate_depth(cursor, source: int, target: int, actuals: Sequence[str]) -> dict[str, str]:
    """
    Uses persisted PredictionLedger rows only. If Top10/25/50/100 pools are not persisted,
    this report will explicitly mark coverage as unavailable rather than inventing data.
    """
    rows = cursor.execute("""
        SELECT
            EngineSource,
            RankNo,
            PredictedNumber,
            Score
        FROM dbo.PredictionLedger
        WHERE SourceDrawNo = ?
          AND TargetDrawNo = ?
        ORDER BY EngineSource, RankNo;
    """, source, target).fetchall()

    by_engine: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        try:
            by_engine[str(row.EngineSource)].append((int(row.RankNo), str(row.PredictedNumber).zfill(4)))
        except Exception:
            continue

    coverage: dict[str, str] = {}
    actual_set = set(actuals)

    if not by_engine:
        return {"status": "NO_PERSISTED_CANDIDATES_FOUND"}

    for engine, items in sorted(by_engine.items()):
        ranked = [number for _, number in sorted(items)]
        max_rank = len(ranked)
        hits = sorted(actual_set.intersection(ranked))
        if not hits:
            coverage[engine] = f"no actuals in persisted depth {max_rank}"
        else:
            hit_ranks = [(num, ranked.index(num) + 1) for num in hits]
            coverage[engine] = f"actuals found: {hit_ranks}; persisted depth={max_rank}"

    return coverage


def build_engine_streaks(cursor) -> list[dict]:
    rows = cursor.execute("""
        SELECT
            p.Mode,
            p.EngineSource,
            p.SourceDrawNo,
            p.TargetDrawNo,
            MAX(ISNULL(p.HitCount, 0)) AS HitCount,
            MAX(p.VerificationStatus) AS VerificationStatus,
            MAX(d.DrawDate) AS TargetDrawDate,
            MAX(d.DayType) AS TargetDayType
        FROM dbo.PredictionLedger p
        LEFT JOIN dbo.DrawHistory d
            ON d.DrawNo = p.TargetDrawNo
        WHERE p.VerificationStatus = 'Verified'
          AND p.EngineSource IS NOT NULL
        GROUP BY
            p.Mode,
            p.EngineSource,
            p.SourceDrawNo,
            p.TargetDrawNo
        ORDER BY
            p.EngineSource,
            p.Mode,
            p.SourceDrawNo;
    """).fetchall()

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for row in rows:
        grouped[(str(row.Mode), str(row.EngineSource))].append(row)

    streak_events: list[dict] = []

    for (mode, engine), engine_rows in grouped.items():
        current_sign = None
        current_len = 0
        start_source = None
        end_source = None

        for row in engine_rows:
            hit = int(row.HitCount or 0) > 0
            sign = 1 if hit else -1

            if current_sign is None:
                current_sign = sign
                current_len = 1
                start_source = int(row.SourceDrawNo)
                end_source = int(row.SourceDrawNo)
            elif sign == current_sign:
                current_len += 1
                end_source = int(row.SourceDrawNo)
            else:
                streak_events.append({
                    "mode": mode,
                    "engine": engine,
                    "sign": current_sign,
                    "length": current_len,
                    "start_source": start_source,
                    "end_source": end_source,
                })
                current_sign = sign
                current_len = 1
                start_source = int(row.SourceDrawNo)
                end_source = int(row.SourceDrawNo)

        if current_sign is not None:
            streak_events.append({
                "mode": mode,
                "engine": engine,
                "sign": current_sign,
                "length": current_len,
                "start_source": start_source,
                "end_source": end_source,
            })

    return streak_events


def renewal_probability_table(cursor) -> list[tuple]:
    """
    For each verified group, compute the miss streak immediately before that draw
    within the same Mode/EngineSource. Then estimate P(next draw hit | prior miss streak length).
    """
    rows = cursor.execute("""
        SELECT
            p.Mode,
            p.EngineSource,
            p.SourceDrawNo,
            p.TargetDrawNo,
            MAX(ISNULL(p.HitCount, 0)) AS HitCount,
            MAX(d.DrawDate) AS TargetDrawDate,
            MAX(d.DayType) AS TargetDayType
        FROM dbo.PredictionLedger p
        LEFT JOIN dbo.DrawHistory d
            ON d.DrawNo = p.TargetDrawNo
        WHERE p.VerificationStatus = 'Verified'
          AND p.EngineSource IS NOT NULL
        GROUP BY
            p.Mode,
            p.EngineSource,
            p.SourceDrawNo,
            p.TargetDrawNo
        ORDER BY
            p.Mode,
            p.EngineSource,
            p.SourceDrawNo;
    """).fetchall()

    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row.Mode), str(row.EngineSource))].append(row)

    stats = defaultdict(lambda: {"samples": 0, "hits": 0})

    for (mode, engine), engine_rows in grouped.items():
        miss_streak = 0
        for row in engine_rows:
            hit = int(row.HitCount or 0) > 0
            target_date = row.TargetDrawDate
            month = getattr(target_date, "month", None) if target_date else None
            year = getattr(target_date, "year", None) if target_date else None
            day_type = str(row.TargetDayType) if row.TargetDayType else "Unknown"

            if miss_streak > 0:
                keys = [
                    ("GLOBAL", mode, engine, miss_streak, "ALL", "ALL"),
                    ("DAYTYPE", mode, engine, miss_streak, day_type, "ALL"),
                    ("MONTH", mode, engine, miss_streak, f"M{month}", "ALL"),
                    ("YEAR", mode, engine, miss_streak, f"Y{year}", "ALL"),
                ]
                for key in keys:
                    stats[key]["samples"] += 1
                    stats[key]["hits"] += int(hit)

            if hit:
                miss_streak = 0
            else:
                miss_streak += 1

    output = []
    for key, value in stats.items():
        scope, mode, engine, streak_len, coord, _ = key
        samples = value["samples"]
        hits = value["hits"]
        rate = hits / samples * 100.0 if samples else 0.0
        output.append((scope, mode, engine, streak_len, coord, samples, hits, rate))

    output.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return output


def fmt_bool(value: bool) -> str:
    return "YES" if value else "NO"


def add_section(lines: list[str], title: str):
    lines.append("")
    lines.append("=" * 110)
    lines.append(title)
    lines.append("=" * 110)


def main() -> int:
    lines: list[str] = []
    now = datetime.now().isoformat(timespec="seconds")

    add_section(lines, "STEP 150A — MASTER FORENSIC & STREAK ANALYTICS AUDIT")
    lines.append(f"GeneratedAt: {now}")
    lines.append("Mode: REPORT ONLY")
    lines.append("ProductionMathChanged: NO")
    lines.append("FrontendChanged: NO")
    lines.append("SQLSchemaChanged: NO")

    with get_conn() as conn:
        cursor = conn.cursor()

        min_draw, max_draw, row_count = fetch_draw_range(cursor)
        ledger_ranges = fetch_ledger_range(cursor)
        latest_rows = fetch_drawhistory_state(cursor)

        add_section(lines, "1. SQL DRAW HISTORY STATE")
        lines.append(f"DrawHistoryRange: {min_draw}..{max_draw}")
        lines.append(f"DrawHistoryRows: {row_count}")
        lines.append("")
        lines.append("Latest DrawHistory Rows:")
        for row in latest_rows:
            draw_no, draw_date, day_type, winners = row
            preview = str(winners)[:90] + ("..." if len(str(winners)) > 90 else "")
            lines.append(f"  DrawNo={draw_no} | DrawDate={draw_date} | DayType={day_type} | Winners={preview}")

        add_section(lines, "2. PREDICTION LEDGER AVAILABLE RANGE")
        if not ledger_ranges:
            lines.append("PredictionLedger: EMPTY")
        else:
            for mode, engine, min_source, max_source, rows in ledger_ranges:
                lines.append(f"  Mode={mode:<22} Engine={engine:<36} SourceRange={min_source}..{max_source} Rows={rows}")
        lines.append("")
        lines.append("Note: Streak analytics use available PredictionLedger only. No fake 40-year ledger is assumed.")

        add_section(lines, "3. THREE LIVE MISSES — LEDGER CONFIRMATION")
        for case in LIVE_CASES:
            rows = fetch_live_ledger_group(cursor, case["source"], case["target"])
            lines.append("")
            lines.append(f"Current {case['source']} -> {case['target']}")
            if not rows:
                lines.append("  LedgerRows: NONE")
                continue
            for row in rows:
                lines.append(f"  {row}")
            statuses = {str(row[6]) for row in rows}
            hit_counts = {row[7] for row in rows}
            engines = {str(row[5]) for row in rows}
            lines.append(f"  RowCount={len(rows)} Engines={sorted(engines)} Statuses={sorted(statuses)} HitCounts={sorted(str(x) for x in hit_counts)}")

        add_section(lines, "4. DEEP FORENSIC & CRYPTO NEAR-MISS METRICS")
        total_mirror_exact = 0
        total_box_exact = 0
        total_hamming1 = 0
        total_pair_match = 0

        for case in LIVE_CASES:
            lines.append("")
            lines.append("-" * 110)
            lines.append(f"CASE: Source {case['source']} -> Target {case['target']}")
            lines.append("-" * 110)

            actual_sums = Counter(digit_sum(num) for num in case["actual"])
            lines.append(f"ActualDigitSumDistribution: {dict(sorted(actual_sums.items()))}")

            for pred in case["predicted"]:
                best_normal = closest_by_metric(
                    pred,
                    case["actual"],
                    key=lambda r: (r.circular_distance, r.hamming_distance, r.digit_sum_delta, r.actual),
                )
                best_mirror = closest_by_metric(
                    pred,
                    case["actual"],
                    key=lambda r: (r.mirror_hamming, r.circular_distance, r.hamming_distance, r.actual),
                )
                best_box = closest_by_metric(
                    pred,
                    case["actual"],
                    key=lambda r: (r.box_distance, r.hamming_distance, r.circular_distance, r.actual),
                )

                total_mirror_exact += int(best_mirror.mirror_exact)
                total_box_exact += int(best_box.box_exact)
                total_hamming1 += int(best_normal.hamming_distance <= 1)
                total_pair_match += int(
                    best_normal.first_pair_match
                    or best_normal.last_pair_match
                    or best_normal.swapped_pair_match
                    or best_mirror.first_pair_match
                    or best_mirror.last_pair_match
                    or best_mirror.swapped_pair_match
                )

                lines.append("")
                lines.append(f"Prediction {pred} | DigitSum={digit_sum(pred)} | MirrorSig={mirror_signature(pred)}")
                lines.append(
                    "  ClosestNormal: "
                    f"Actual={best_normal.actual} CircDist={best_normal.circular_distance} "
                    f"Hamming={best_normal.hamming_distance} SumDelta={best_normal.digit_sum_delta} "
                    f"FirstPair={fmt_bool(best_normal.first_pair_match)} "
                    f"LastPair={fmt_bool(best_normal.last_pair_match)} "
                    f"SwappedPair={fmt_bool(best_normal.swapped_pair_match)}"
                )
                lines.append(
                    "  ClosestMirror: "
                    f"Actual={best_mirror.actual} ActualMirror={mirror_signature(best_mirror.actual)} "
                    f"MirrorExact={fmt_bool(best_mirror.mirror_exact)} "
                    f"MirrorHamming={best_mirror.mirror_hamming} "
                    f"NormalCirc={best_mirror.circular_distance}"
                )
                lines.append(
                    "  ClosestBox: "
                    f"Actual={best_box.actual} BoxExact={fmt_bool(best_box.box_exact)} "
                    f"BoxOverlap={best_box.box_overlap}/4 BoxDistance={best_box.box_distance} "
                    f"Hamming={best_box.hamming_distance}"
                )

        lines.append("")
        lines.append("FORENSIC AGGREGATE:")
        lines.append(f"  MirrorExactHiddenHits={total_mirror_exact}")
        lines.append(f"  BoxExactHits={total_box_exact}")
        lines.append(f"  HammingDistance<=1 NearMisses={total_hamming1}")
        lines.append(f"  PairStructureNearMisses={total_pair_match}")

        add_section(lines, "5. CANDIDATE POOL CUTOFF & DEPTH AUDIT")
        lines.append("Method: uses persisted PredictionLedger candidate rows. If full Top100 pools were not persisted, this section is marked as limited.")
        for case in LIVE_CASES:
            coverage = fetch_candidate_depth(cursor, case["source"], case["target"], case["actual"])
            lines.append("")
            lines.append(f"CandidateDepthCoverage Source {case['source']} -> Target {case['target']}:")
            for engine, detail in coverage.items():
                lines.append(f"  {engine}: {detail}")
            if all("no actuals" in detail or "NO_PERSISTED" in detail for detail in coverage.values()):
                lines.append("  Diagnosis: generation/repair issue likely OR pool depth not persisted. Requires reconstructed Top100 instrumentation.")
            else:
                lines.append("  Diagnosis: ranking/cutoff issue possible where actuals appear below Top5.")

        add_section(lines, "6. HIT-MISS STREAK DURATION DATABASE")
        streaks = build_engine_streaks(cursor)
        miss_lengths = Counter()
        hit_lengths = Counter()
        for item in streaks:
            if item["sign"] < 0:
                miss_lengths[(item["mode"], item["engine"], item["length"])] += 1
            else:
                hit_lengths[(item["mode"], item["engine"], item["length"])] += 1

        lines.append(f"TotalStreakBlocks={len(streaks)}")
        lines.append("")
        lines.append("Top Miss Streak Blocks:")
        top_miss = sorted(
            miss_lengths.items(),
            key=lambda kv: (-kv[0][2], -kv[1], kv[0][0], kv[0][1]),
        )[:30]
        for (mode, engine, length), count in top_miss:
            lines.append(f"  Mode={mode:<22} Engine={engine:<36} MissLength={length:<4} Blocks={count}")

        lines.append("")
        lines.append("Top Hit Streak Blocks:")
        top_hit = sorted(
            hit_lengths.items(),
            key=lambda kv: (-kv[0][2], -kv[1], kv[0][0], kv[0][1]),
        )[:20]
        for (mode, engine, length), count in top_hit:
            lines.append(f"  Mode={mode:<22} Engine={engine:<36} HitLength={length:<4} Blocks={count}")

        add_section(lines, "7. HIT-MISS STREAK RECOVERY FREQUENCY")
        renewal_rows = renewal_probability_table(cursor)

        interesting = []
        for row in renewal_rows:
            scope, mode, engine, streak_len, coord, samples, hits, rate = row
            if samples >= 3 and streak_len in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}:
                interesting.append(row)

        interesting.sort(key=lambda r: (-r[7], -r[5], r[0], r[1], r[2], r[3], r[4]))
        lines.append("Top Recovery Patterns, sample>=3:")
        for scope, mode, engine, streak_len, coord, samples, hits, rate in interesting[:50]:
            warning = " LOW_SAMPLE" if samples < 10 else ""
            lines.append(
                f"  Scope={scope:<8} Mode={mode:<22} Engine={engine:<36} "
                f"PriorMissStreak={streak_len:<3} Coord={coord:<12} "
                f"Samples={samples:<4} NextHits={hits:<4} NextHitRate={rate:6.2f}%{warning}"
            )

        lines.append("")
        lines.append("Current Live Miss Streak Context:")
        lines.append("  E1_TEMPORAL_CONTEXT_MATCH Current live streak: 3 misses observed.")
        lines.append("  Use recovery table above to evaluate whether miss-streak momentum has enough samples.")
        lines.append("  Low-sample contexts are not promotion-grade.")

        add_section(lines, "8. DIAGNOSIS MATRIX")
        lines.append("Generation Issue:")
        lines.append("  If actual prizes are absent from persisted or reconstructed Top100 pools.")
        lines.append("Ranking/Cutoff Issue:")
        lines.append("  If actual prizes appear below Top5 but inside Top25/Top50/Top100.")
        lines.append("Mirror Expansion Issue:")
        lines.append("  If mirror class exact/near matches are frequent while exact hits fail.")
        lines.append("Box Repair Issue:")
        lines.append("  If digit multiset overlap is high but positional exact hits fail.")
        lines.append("")
        lines.append("Initial Evidence:")
        lines.append("  - 0000 vs 5055 is a Base-5 Mirror exact hidden hit.")
        lines.append("  - 4443 vs 4433 is a pair/box/hamming near miss.")
        lines.append("  - Three live exact misses justify residual audit, not blind production switch.")

        add_section(lines, "9. E2_LIVE_RESIDUAL_META_RANKER — DRAFT BLUEPRINT")
        lines.append("Proposed streams:")
        lines.append("  1. Temporal Base Stream: existing E1_TEMPORAL_CONTEXT_MATCH candidates.")
        lines.append("  2. Mirror Expansion Stream: expand candidates across Base-5 mirror equivalence classes.")
        lines.append("  3. Box Repair Stream: reorder/repair high-overlap digit baskets.")
        lines.append("  4. Residual Delta Stream: apply common digit-wise circular deltas from recent misses.")
        lines.append("  5. Pair Repair Stream: preserve strong first/last-pair signals and repair the opposite pair.")
        lines.append("  6. Streak Momentum Stream: weight engines after historically recoverable miss streak lengths.")
        lines.append("  7. Diversity Guard: final Top5 must not be monopolized by one signal family.")
        lines.append("")
        lines.append("Promotion criteria:")
        lines.append("  - Must beat baseline on honest rolling backtest.")
        lines.append("  - Must preserve temporal firewall.")
        lines.append("  - Must not use target winner before lock.")
        lines.append("  - Must pass ledger integrity audit.")
        lines.append("  - Must not change production Current mode until approved.")

        add_section(lines, "10. FINAL STEP 150A CONCLUSION")
        lines.append("ProductionMathChanged: NO")
        lines.append("ProductionSwitchRecommendedNow: NO")
        lines.append("ResidualPrototypeRecommended: YES")
        lines.append("CandidateTop100InstrumentationRecommended: YES")
        lines.append("MirrorClassExpansionRecommended: YES")
        lines.append("BoxRepairLayerRecommended: YES")
        lines.append("StreakMomentumMatrixRecommended: YES")
        lines.append("NextStep: STEP 150B — reconstructed candidate-pool instrumentation and residual prototype backtest.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("")
    print(f"REPORT_WRITTEN: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
