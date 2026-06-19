from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from shadow_hybrid_engine_v2 import (
    ENGINE_NAME,
    TOP_K,
    V2HistoricalState,
    fetch_draws,
    fetch_ledger_pairs,
    generate_with_context,
    get_conn,
    risk_bucket,
    runtime_bucket,
)


REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "step_151b_baseline_protected_shadow_hybrid_v2_report.txt"
)
JSONL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "step_151b_baseline_protected_shadow_hybrid_v2_rows.jsonl"
)
V1_JSONL_PATH = PROJECT_ROOT / "reports" / "step_151a_shadow_hybrid_engine_rows.jsonl"
NUMBER_SPACE = 10_000
LIVE_SOURCES = (5494, 5495, 5496, 5497)
LOCKED_150F_HYBRID_5497 = ("7006", "2689", "7776", "3193", "3509")


def load_v1_rows() -> dict[int, dict]:
    if not V1_JSONL_PATH.exists():
        return {}
    output = {}
    with V1_JSONL_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[int(row["source_draw_no"])] = row
    return output


def make_row(result, pair, target_draw, prior_max_target, v1_row) -> dict:
    actuals = target_draw.winners if target_draw else ()
    actual_set = set(actuals)
    target_available = target_draw is not None
    exact_hits = (
        sum(number in actual_set for number in result["top5"])
        if target_available
        else None
    )
    baseline_hits = (
        sum(number in actual_set for number in result["baseline_top5"])
        if target_available
        else None
    )
    v1_hits = None
    if target_available and v1_row is not None:
        v1_hits = sum(number in actual_set for number in v1_row["top5"])
    prize_count = len(actual_set)
    random_rate = (
        1 - ((NUMBER_SPACE - prize_count) / NUMBER_SPACE) ** TOP_K
        if target_available
        else None
    )
    return {
        "source_draw_no": result["source_draw_no"],
        "target_draw_no": result["target_draw_no"],
        "target_available": target_available,
        "target_month": target_draw.month if target_draw else None,
        "target_day_type": target_draw.day_type if target_draw else None,
        "runtime_bucket": runtime_bucket(target_draw.day_type if target_draw else None),
        "engine_name": result["engine_name"],
        "risk_bucket": result["risk_bucket"],
        "full_ledger_pre_miss_streak": result["full_ledger_pre_miss_streak"],
        "baseline_top5": result["baseline_top5"],
        "top5": result["top5"],
        "baseline_kept_count": result["baseline_kept_count"],
        "replacement_attempts": result["replacement_attempts"],
        "replacement_used": result["replacement_used"],
        "replacement_detail": result["replacement_detail"],
        "candidate_details": result["candidate_details"],
        "exact_hit_count": exact_hits,
        "baseline_hit_count": baseline_hits,
        "v1_hit_count": v1_hits,
        "verification_status": "VERIFIED" if target_available else "UNVERIFIED",
        "warnings": result["warnings"],
        "temporal_firewall_ok": result["temporal_firewall_ok"],
        "candidate_history_max_draw_no": max(
            item["history_max_draw_no"] for item in result["candidate_details"]
        ),
        "prior_training_max_target_draw_no": prior_max_target,
        "random_expected_top5_rate": random_rate,
        "comparison_step_150f_hybrid": (
            list(LOCKED_150F_HYBRID_5497)
            if result["source_draw_no"] == 5497
            else None
        ),
    }


def run_backtest(draws, pairs, v1_rows):
    rows = []
    state = V2HistoricalState()
    streak = 0
    pairs_by_source = {pair.source: pair for pair in pairs}
    for pair in pairs:
        target = draws.get(pair.target)
        if target is None:
            continue
        result = generate_with_context(
            pair.source,
            draws,
            pairs_by_source,
            state,
            streak,
        )
        rows.append(
            make_row(
                result,
                pair,
                target,
                state.observed_max_target,
                v1_rows.get(pair.source),
            )
        )
        state.observe(risk_bucket(streak), pair, target.winners)
        streak = 0 if pair.ledger_hit_count > 0 else streak + 1
    return rows, state, streak


def summarize(rows: list[dict], predicate: Callable[[dict], bool]) -> dict:
    selected = [row for row in rows if row["target_available"] and predicate(row)]
    count = len(selected)
    hit_draws = sum((row["exact_hit_count"] or 0) > 0 for row in selected)
    raw_hits = sum(row["exact_hit_count"] or 0 for row in selected)
    baseline_hits = sum((row["baseline_hit_count"] or 0) > 0 for row in selected)
    v1_available = [row for row in selected if row["v1_hit_count"] is not None]
    v1_hits = sum((row["v1_hit_count"] or 0) > 0 for row in v1_available)
    hit_rate = hit_draws / count if count else 0.0
    baseline_rate = baseline_hits / count if count else 0.0
    v1_rate = v1_hits / len(v1_available) if v1_available else None
    random_rate = (
        statistics.mean(row["random_expected_top5_rate"] or 0 for row in selected)
        if selected
        else 0.0
    )
    replacement_rows = [row for row in selected if row["replacement_used"]]
    return {
        "rows": count,
        "hit_draws": hit_draws,
        "raw_hits": raw_hits,
        "hit_rate": hit_rate,
        "baseline_hit_rate": baseline_rate,
        "delta_vs_baseline": hit_rate - baseline_rate,
        "v1_hit_rate": v1_rate,
        "delta_vs_v1": hit_rate - v1_rate if v1_rate is not None else None,
        "random_rate": random_rate,
        "enrichment": hit_rate / random_rate if random_rate else 0.0,
        "replacement_attempts": sum(row["replacement_attempts"] for row in selected),
        "replacement_used_count": len(replacement_rows),
        "replacement_hit_count": sum(
            (row["exact_hit_count"] or 0) > (row["baseline_hit_count"] or 0)
            for row in replacement_rows
        ),
        "baseline_kept_average": (
            statistics.mean(row["baseline_kept_count"] for row in selected)
            if selected
            else 0.0
        ),
        "warning_count": sum(len(row["warnings"]) for row in selected),
        "temporal_violations": sum(not row["temporal_firewall_ok"] for row in selected),
        "invalid_top5_count": sum(
            len(row["top5"]) != TOP_K
            or any(len(number) != 4 or not number.isdigit() for number in row["top5"])
            for row in selected
        ),
        "duplicate_top5_count": sum(
            len(set(row["top5"])) != TOP_K for row in selected
        ),
    }


def non_inferiority(summaries: dict[str, dict]) -> tuple[bool, list[str]]:
    thresholds = {
        "FULL_VERIFIED": -0.0005,
        "RECENT_365": -0.0010,
        "RECENT_90": 0.0,
        "RECENT_47": 0.0,
        "HIGH_RISK_ONLY": -0.0010,
    }
    failures = []
    for name, threshold in thresholds.items():
        delta = summaries[name]["delta_vs_baseline"]
        if delta < threshold:
            failures.append(
                f"{name} delta {delta * 100:.3f}pp below {threshold * 100:.3f}pp"
            )
    return not failures, failures


def add_table(lines, title, items):
    width = 164
    lines.extend(("", title, "-" * width))
    lines.append(
        "Window                               Rows HitDraws RawHits HitRate Baseline DeltaBase V1Rate DeltaV1 Random Enrich Attempts Used ReplHits KeptAvg Warnings"
    )
    for name, item in items:
        v1_rate = "NULL" if item["v1_hit_rate"] is None else f"{item['v1_hit_rate'] * 100:.3f}%"
        delta_v1 = "NULL" if item["delta_vs_v1"] is None else f"{item['delta_vs_v1'] * 100:+.3f}pp"
        lines.append(
            f"{name:<36} {item['rows']:>5} {item['hit_draws']:>8} {item['raw_hits']:>7} "
            f"{item['hit_rate'] * 100:>6.3f}% {item['baseline_hit_rate'] * 100:>7.3f}% "
            f"{item['delta_vs_baseline'] * 100:>+8.3f}pp {v1_rate:>7} {delta_v1:>9} "
            f"{item['random_rate'] * 100:>6.3f}% {item['enrichment']:>6.3f} "
            f"{item['replacement_attempts']:>8} {item['replacement_used_count']:>4} "
            f"{item['replacement_hit_count']:>8} {item['baseline_kept_average']:>7.3f} "
            f"{item['warning_count']:>8}"
        )


def build_report(rows, live_row, summaries):
    width = 164
    noninferior, failures = non_inferiority(summaries)
    verified = [row for row in rows if row["target_available"]]
    used = [row for row in verified if row["replacement_used"]]
    support_counts = Counter(
        support
        for row in used
        for support in row["replacement_detail"]["supports"]
    )
    warning_counts = Counter(
        warning for row in verified for warning in row["warnings"]
    )
    candidate_violations = sum(
        row["candidate_history_max_draw_no"] > row["source_draw_no"] for row in rows
    )
    target_leakage = sum(
        row["prior_training_max_target_draw_no"] > row["source_draw_no"] for row in rows
    )
    invalid = sum(
        len(row["top5"]) != TOP_K
        or any(len(number) != 4 or not number.isdigit() for number in row["top5"])
        for row in rows
    )
    duplicates = sum(len(set(row["top5"])) != TOP_K for row in rows)
    meaningful_wins = sum(
        summaries[name]["delta_vs_baseline"] > 0
        for name in (
            "FULL_VERIFIED",
            "RECENT_365",
            "RECENT_90",
            "RECENT_47",
            "HIGH_RISK_ONLY",
        )
    )

    lines = [
        "=" * width,
        "STEP 151B — BASELINE-PROTECTED SHADOW HYBRID V2 — REPORT ONLY",
        "=" * width,
        f"EngineName: {ENGINE_NAME}",
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "SQLSchemaChanged: NO",
        "DeploymentChanged: NO",
        "ProductionIntegration: NONE",
        "",
        "STEP 151A FAILURE SUMMARY",
        "-" * width,
        "V1 full verified: 1.037% vs baseline 1.451% (-0.415pp).",
        "V1 high-risk: 0.973% vs baseline 1.503% (-0.531pp).",
        "V1 recent 90 and recent 47: 0 hits.",
        "Diagnosis: excessive historical-winner fallback diluted the baseline.",
        "V2 response: keep at least four baseline candidates, permit at most one strictly guarded replacement, and disable historical-winner fallback.",
        "",
        "CURRENT LIVE SOURCE / TARGET",
        "-" * width,
        f"LatestSourceDrawNo: {live_row['source_draw_no']}",
        f"TargetDrawNo: {live_row['target_draw_no']}",
        f"TargetAvailable: {'YES' if live_row['target_available'] else 'NO'}",
        f"VerificationStatus: {live_row['verification_status']}",
        f"FullLedgerPreMissStreak: {live_row['full_ledger_pre_miss_streak']}",
        f"RiskBucket: {live_row['risk_bucket']}",
        f"BaselineTop5: {live_row['baseline_top5']}",
        f"V2Top5: {live_row['top5']}",
        f"BaselineKeptCount: {live_row['baseline_kept_count']}",
        f"ReplacementUsed: {'YES' if live_row['replacement_used'] else 'NO'}",
        f"ReplacementDetail: {live_row['replacement_detail']}",
        f"Step150FHybridComparison: {live_row['comparison_step_150f_hybrid']}",
        f"Warnings: {live_row['warnings'] or ['NONE']}",
        "CandidateDetails:",
    ]
    for item in live_row["candidate_details"]:
        lines.append(
            f"  Rank={item['rank']} Number={item['number']} Family={item['family']} "
            f"Score={item['score']:.6f} Supports={item['supports']} "
            f"HistoryMax={item['history_max_draw_no']} Reason={item['reason']}"
        )

    add_table(
        lines,
        "HISTORICAL BACKTEST SUMMARY",
        [
            ("FULL_VERIFIED", summaries["FULL_VERIFIED"]),
            ("HIGH_RISK_ONLY", summaries["HIGH_RISK_ONLY"]),
        ],
    )
    add_table(
        lines,
        "WINDOW BREAKDOWN",
        [
            ("RECENT_365", summaries["RECENT_365"]),
            ("RECENT_90", summaries["RECENT_90"]),
            ("RECENT_47", summaries["RECENT_47"]),
            ("HIGH_RISK_MONTH_JUNE", summaries["HIGH_RISK_MONTH_JUNE"]),
            ("HIGH_RISK_WEEKEND_SPACE", summaries["HIGH_RISK_WEEKEND_SPACE"]),
            (
                "HIGH_RISK_MIDWEEK_SPECIAL",
                summaries["HIGH_RISK_MIDWEEK_SPECIAL"],
            ),
        ],
    )

    lines.extend(
        (
            "",
            "REPLACEMENT DIAGNOSTICS",
            "-" * width,
            f"ReplacementAttempts: {sum(row['replacement_attempts'] for row in verified)}",
            f"ReplacementsUsed: {len(used)}",
            f"ReplacementsThatImprovedExactHit: {sum((row['exact_hit_count'] or 0) > (row['baseline_hit_count'] or 0) for row in used)}",
            f"ReplacementsThatRemovedBaselineHit: {sum((row['exact_hit_count'] or 0) < (row['baseline_hit_count'] or 0) for row in used)}",
            f"MostCommonSupports: {support_counts.most_common()}",
            f"MostCommonWarnings: {warning_counts.most_common()}",
        )
    )

    lines.extend(
        (
            "",
            "LIVE 5494→5497 AUDIT",
            "-" * width,
            "Source Target Risk        PreMiss BaselineTop5                            V2Top5                                  Kept Repl V2Hits BaseHits V1Hits Status Warnings",
        )
    )
    by_source = {row["source_draw_no"]: row for row in rows}
    by_source[live_row["source_draw_no"]] = live_row
    for source in LIVE_SOURCES:
        row = by_source.get(source)
        if row is None:
            lines.append(f"{source}: NO_ROW")
            continue
        values = [
            "NULL" if row[key] is None else str(row[key])
            for key in ("exact_hit_count", "baseline_hit_count", "v1_hit_count")
        ]
        lines.append(
            f"{source:>6} {row['target_draw_no']:>6} {row['risk_bucket']:<11} "
            f"{row['full_ledger_pre_miss_streak']:>7} {str(row['baseline_top5']):<39} "
            f"{str(row['top5']):<39} {row['baseline_kept_count']:>4} "
            f"{str(row['replacement_used']):>4} {values[0]:>6} {values[1]:>8} "
            f"{values[2]:>6} {row['verification_status']:<10} "
            f"{row['warnings'] or ['NONE']}"
        )

    lines.extend(
        (
            "",
            "TEMPORAL FIREWALL VALIDATION",
            "-" * width,
            f"CandidateHistoryViolations: {candidate_violations}",
            f"TargetLeakageChecksFailed: {target_leakage}",
            f"InvalidTop5Checks: {invalid}",
            f"DuplicateTop5Checks: {duplicates}",
            f"TemporalFirewallValidation: {'PASS' if not candidate_violations and not target_leakage else 'FAIL'}",
            "",
            "FINAL RECOMMENDATION",
            "-" * width,
            f"NonInferiorityVsBaseline: {'PASS' if noninferior else 'FAIL'}",
            f"NonInferiorityFailures: {failures or ['NONE']}",
            f"MeaningfulWindowsBeatingBaseline: {meaningful_wins}",
            "ProductionSwitchRecommendedNow: NO",
            "NextStep: Review Step 151B before deciding Step 151C locked live prediction generation",
            "",
            f"REPORT_WRITTEN: {REPORT_PATH}",
            f"JSONL_WRITTEN: {JSONL_PATH}",
        )
    )
    return "\n".join(lines)


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    v1_rows = load_v1_rows()
    with get_conn() as connection:
        cursor = connection.cursor()
        draws = fetch_draws(cursor)
        all_pairs = fetch_ledger_pairs(cursor)
    verified_pairs = [pair for pair in all_pairs if pair.target in draws]
    rows, state, streak = run_backtest(draws, verified_pairs, v1_rows)
    latest_source = max(draws)
    pairs_by_source = {pair.source: pair for pair in all_pairs}
    live_result = generate_with_context(
        latest_source,
        draws,
        pairs_by_source,
        state,
        streak,
    )
    live_row = make_row(
        live_result,
        pairs_by_source.get(latest_source),
        draws.get(latest_source + 1),
        state.observed_max_target,
        v1_rows.get(latest_source),
    )

    sources = sorted(row["source_draw_no"] for row in rows)
    recent_365 = set(sources[-365:])
    recent_90 = set(sources[-90:])
    recent_47 = set(sources[-47:])
    filters: dict[str, Callable[[dict], bool]] = {
        "FULL_VERIFIED": lambda row: True,
        "RECENT_365": lambda row: row["source_draw_no"] in recent_365,
        "RECENT_90": lambda row: row["source_draw_no"] in recent_90,
        "RECENT_47": lambda row: row["source_draw_no"] in recent_47,
        "HIGH_RISK_ONLY": lambda row: row["risk_bucket"]
        in {"HIGH_RISK", "EXTREME_RISK"},
        "HIGH_RISK_MONTH_JUNE": lambda row: row["risk_bucket"]
        in {"HIGH_RISK", "EXTREME_RISK"}
        and row["target_month"] == 6,
        "HIGH_RISK_WEEKEND_SPACE": lambda row: row["risk_bucket"]
        in {"HIGH_RISK", "EXTREME_RISK"}
        and row["runtime_bucket"] == "WEEKEND_SPACE",
        "HIGH_RISK_MIDWEEK_SPECIAL": lambda row: row["risk_bucket"]
        in {"HIGH_RISK", "EXTREME_RISK"}
        and row["runtime_bucket"] == "MIDWEEK_SPECIAL_SPACE",
    }
    summaries = {name: summarize(rows, predicate) for name, predicate in filters.items()}
    report = build_report(rows, live_row, summaries)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    with JSONL_PATH.open("w", encoding="utf-8") as handle:
        for row in [*rows, live_row]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(report)


if __name__ == "__main__":
    main()
