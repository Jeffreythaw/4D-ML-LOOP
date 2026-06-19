from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pyodbc
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))
load_dotenv(BACKEND_ROOT / ".env")

from app.core.config import get_settings


SOURCE_DRAW_NO = 5497
TARGET_DRAW_NO = 5498
CURRENT_FULL_LEDGER_PRE_MISS_STREAK = 18
CURRENT_RISK_BUCKET = "HIGH_RISK"
LIVE_DEPLOYMENT_LOCAL_MISS_STREAK = 3

REPORT_PATH = PROJECT_ROOT / "reports" / "step_150g_live_shadow_board_verification.txt"
JSONL_PATH = PROJECT_ROOT / "reports" / "step_150g_live_shadow_board_verification_rows.jsonl"

LOCKED_BOARD = {
    "BASELINE_E1": ("7006", "4723", "3193", "9098", "2698"),
    "RISK_SUM_BAND_SHADOW": ("3193", "4723", "7006", "6065", "6515"),
    "PAIR_RECURRENCE_SHADOW": ("7776", "0374", "1111", "1389", "2071"),
    "MIRROR_BOX_REPAIR_SHADOW": ("2689", "2869", "2896", "2968", "2986"),
    "STREAK_RISK_SUPPRESSION_SHADOW": ("3509", "8059", "9106", "1111", "2086"),
    "HYBRID_GUARD_SHADOW": ("7006", "2689", "7776", "3193", "3509"),
}


@dataclass(frozen=True)
class PairMetrics:
    candidate: str
    actual: str
    circular_distance: int
    hamming_distance: int
    box_overlap: int
    mirror_exact: bool
    box_exact: bool


@dataclass(frozen=True)
class VerificationRow:
    source_draw_no: int
    target_draw_no: int
    target_available: bool
    policy: str
    top5: tuple[str, ...]
    exact_hit_count: int | None
    sp_hit_count: int | None
    sp_check_status: str
    min_circular_digit_distance: int | None
    min_hamming_distance: int | None
    max_box_overlap: int | None
    mirror_exact_hidden_hits: int | None
    box_exact_hidden_hits: int | None
    best_candidate: str | None
    best_actual: str | None
    best_near_miss_metrics: dict | None
    final_status: str


def get_conn():
    return pyodbc.connect(get_settings().sql_connection_string(), timeout=120)


def z4(value: str | int) -> str:
    return str(value).strip().zfill(4)


def parse_numbers(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(z4(part) for part in str(value).replace(" ", "").split(",") if part)


def mirror_signature(number: str) -> str:
    return "".join(str(int(digit) % 5) for digit in z4(number))


def box_signature(number: str) -> str:
    return "".join(sorted(z4(number)))


def circular_digit_distance(left: str, right: str) -> int:
    total = 0
    for left_digit, right_digit in zip(z4(left), z4(right)):
        delta = abs(int(left_digit) - int(right_digit))
        total += min(delta, 10 - delta)
    return total


def hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(z4(left), z4(right)))


def box_overlap(left: str, right: str) -> int:
    return sum((Counter(z4(left)) & Counter(z4(right))).values())


def pair_metrics(candidate: str, actual: str) -> PairMetrics:
    normalized_candidate = z4(candidate)
    normalized_actual = z4(actual)
    return PairMetrics(
        candidate=normalized_candidate,
        actual=normalized_actual,
        circular_distance=circular_digit_distance(normalized_candidate, normalized_actual),
        hamming_distance=hamming_distance(normalized_candidate, normalized_actual),
        box_overlap=box_overlap(normalized_candidate, normalized_actual),
        mirror_exact=(
            normalized_candidate != normalized_actual
            and mirror_signature(normalized_candidate) == mirror_signature(normalized_actual)
        ),
        box_exact=(
            normalized_candidate != normalized_actual
            and box_signature(normalized_candidate) == box_signature(normalized_actual)
        ),
    )


def fetch_target_winners(cursor) -> tuple[str, ...] | None:
    row = cursor.execute(
        """
        SELECT WinningNumbers
        FROM dbo.DrawHistory
        WHERE DrawNo = ?;
        """,
        TARGET_DRAW_NO,
    ).fetchone()
    if row is None or not row.WinningNumbers:
        return None
    return parse_numbers(row.WinningNumbers)


def stored_procedure_available(cursor) -> bool:
    row = cursor.execute(
        """
        SELECT CASE
            WHEN OBJECT_ID(?, 'P') IS NULL THEN 0
            ELSE 1
        END AS ProcedureExists;
        """,
        "dbo.SP_Verify_Predictions",
    ).fetchone()
    return bool(row and int(row.ProcedureExists))


def call_sql_verifier(cursor, top5: tuple[str, ...]) -> int:
    row = cursor.execute(
        """
        EXEC dbo.SP_Verify_Predictions
            @TargetDrawNo = ?,
            @Top5Predictions = ?;
        """,
        TARGET_DRAW_NO,
        ",".join(top5),
    ).fetchone()
    if row is None:
        raise RuntimeError("SP_Verify_Predictions returned no row")
    return int(row[0])


def best_pair(metrics: list[PairMetrics]) -> PairMetrics:
    return min(
        metrics,
        key=lambda item: (
            item.circular_distance,
            item.hamming_distance,
            -item.box_overlap,
            -int(item.mirror_exact),
            -int(item.box_exact),
            item.candidate,
            item.actual,
        ),
    )


def verify_policy(
    policy: str,
    top5: tuple[str, ...],
    actuals: tuple[str, ...] | None,
    sp_hit_count: int | None,
    sp_status: str,
) -> VerificationRow:
    if actuals is None:
        return VerificationRow(
            source_draw_no=SOURCE_DRAW_NO,
            target_draw_no=TARGET_DRAW_NO,
            target_available=False,
            policy=policy,
            top5=top5,
            exact_hit_count=None,
            sp_hit_count=None,
            sp_check_status=sp_status,
            min_circular_digit_distance=None,
            min_hamming_distance=None,
            max_box_overlap=None,
            mirror_exact_hidden_hits=None,
            box_exact_hidden_hits=None,
            best_candidate=None,
            best_actual=None,
            best_near_miss_metrics=None,
            final_status="UNVERIFIED",
        )

    actual_set = set(actuals)
    metrics = [pair_metrics(candidate, actual) for candidate in top5 for actual in actuals]
    best = best_pair(metrics)
    exact_hits = sum(candidate in actual_set for candidate in top5)
    status = "EXACT_HIT" if exact_hits else "VERIFIED_MISS"
    return VerificationRow(
        source_draw_no=SOURCE_DRAW_NO,
        target_draw_no=TARGET_DRAW_NO,
        target_available=True,
        policy=policy,
        top5=top5,
        exact_hit_count=exact_hits,
        sp_hit_count=sp_hit_count,
        sp_check_status=sp_status,
        min_circular_digit_distance=min(item.circular_distance for item in metrics),
        min_hamming_distance=min(item.hamming_distance for item in metrics),
        max_box_overlap=max(item.box_overlap for item in metrics),
        mirror_exact_hidden_hits=sum(item.mirror_exact for item in metrics),
        box_exact_hidden_hits=sum(item.box_exact for item in metrics),
        best_candidate=best.candidate,
        best_actual=best.actual,
        best_near_miss_metrics={
            "circular_distance": best.circular_distance,
            "hamming_distance": best.hamming_distance,
            "box_overlap": best.box_overlap,
            "mirror_exact": best.mirror_exact,
            "box_exact": best.box_exact,
        },
        final_status=status,
    )


def policy_rank(row: VerificationRow) -> tuple:
    return (
        -(row.exact_hit_count or 0),
        row.min_circular_digit_distance if row.min_circular_digit_distance is not None else 999,
        row.min_hamming_distance if row.min_hamming_distance is not None else 999,
        -(row.max_box_overlap or 0),
        -(row.mirror_exact_hidden_hits or 0),
        row.policy,
    )


def final_decision(rows: list[VerificationRow]) -> tuple[str, str, str]:
    if not rows[0].target_available:
        return (
            "WAIT_FOR_5498_RESULT",
            "Target 5498 is not available in DrawHistory.",
            "NO",
        )

    baseline = next(row for row in rows if row.policy == "BASELINE_E1")
    shadow_hits = [
        row
        for row in rows
        if row.policy != "BASELINE_E1" and (row.exact_hit_count or 0) > 0
    ]
    if (baseline.exact_hit_count or 0) > 0:
        return (
            "BASELINE_STILL_ALIVE",
            "Baseline recorded an exact hit; no shadow promotion is supported.",
            "NO",
        )
    if shadow_hits:
        return (
            "LIVE_SIGNAL_OBSERVED",
            "At least one shadow policy hit while baseline missed; rolling confirmation is required.",
            "NO",
        )
    return (
        "NO_LIVE_BREAKTHROUGH",
        "All locked policies missed target 5498.",
        "NO",
    )


def build_report(
    rows: list[VerificationRow],
    actuals: tuple[str, ...] | None,
    procedure_available: bool,
) -> str:
    width = 142
    decision, reason, production_switch = final_decision(rows)
    ranked = sorted(rows, key=policy_rank)
    lines = [
        "=" * width,
        "STEP 150G — LIVE SHADOW BOARD VERIFICATION HARNESS — REPORT ONLY",
        "=" * width,
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "SQLSchemaChanged: NO",
        "DeploymentChanged: NO",
        f"SourceDrawNo: {SOURCE_DRAW_NO}",
        f"TargetDrawNo: {TARGET_DRAW_NO}",
        f"CurrentFullLedgerPreMissStreak: {CURRENT_FULL_LEDGER_PRE_MISS_STREAK}",
        f"CurrentRiskBucket: {CURRENT_RISK_BUCKET}",
        f"LiveDeploymentLocalMissStreak: {LIVE_DEPLOYMENT_LOCAL_MISS_STREAK}",
        "RiskUse: suppression/caution only; no streak recovery boost",
        "CandidateLock: Step 150F constants only; target winners never alter candidate sets.",
        "",
        "LOCKED SHADOW BOARD",
        "-" * width,
        "Policy                              Locked Top5",
    ]
    for policy, top5 in LOCKED_BOARD.items():
        lines.append(f"{policy:<35} {','.join(top5)}")

    lines.extend(
        (
            "",
            "TARGET AVAILABILITY CHECK",
            "-" * width,
            f"TargetAvailableInDrawHistory: {'YES' if actuals is not None else 'NO'}",
        )
    )
    if actuals is None:
        lines.extend(
            (
                "Every policy status: UNVERIFIED",
                "SQLStoredProcedureCheck: NOT_RUN_TARGET_UNAVAILABLE",
                "Local deterministic comparison: NOT_RUN_TARGET_UNAVAILABLE",
                "",
                "WAIT_FOR_5498_RESULT",
                "-" * width,
                "Decision: WAIT_FOR_5498_RESULT",
                "No verification or near-miss claim is made before target 5498 exists.",
                "ProductionSwitchRecommendedNow: NO",
            )
        )
    else:
        lines.extend(
            (
                f"Actual5498Winners: {','.join(actuals)}",
                f"ActualPrizeCount: {len(actuals)}",
                "",
                "POLICY VERIFICATION TABLE",
                "-" * width,
                "Rank Policy                              ExactHits MinCircular MinHamming MaxBox MirrorHidden BoxHidden Status",
            )
        )
        for rank, row in enumerate(ranked, start=1):
            lines.append(
                f"{rank:>4} {row.policy:<35} {row.exact_hit_count:>9} "
                f"{row.min_circular_digit_distance:>11} {row.min_hamming_distance:>10} "
                f"{row.max_box_overlap:>6} {row.mirror_exact_hidden_hits:>12} "
                f"{row.box_exact_hidden_hits:>9} {row.final_status}"
            )

        lines.extend(
            (
                "",
                "SQL SP CROSS-CHECK TABLE",
                "-" * width,
                "Local deterministic comparison is secondary to dbo.SP_Verify_Predictions when the procedure is available.",
                f"SQLStoredProcedureAvailable: {'YES' if procedure_available else 'NO'}",
                "Policy                              LocalHits SPHits SPCheckStatus",
            )
        )
        for row in rows:
            sp_value = "NULL" if row.sp_hit_count is None else str(row.sp_hit_count)
            lines.append(
                f"{row.policy:<35} {row.exact_hit_count:>9} {sp_value:>6} {row.sp_check_status}"
            )

        lines.extend(
            (
                "",
                "NEAR-MISS TABLE",
                "-" * width,
                "Policy                              Candidate Actual Circular Hamming BoxOverlap MirrorExact BoxExact",
            )
        )
        for row in rows:
            metrics = row.best_near_miss_metrics or {}
            lines.append(
                f"{row.policy:<35} {row.best_candidate:>9} {row.best_actual:>6} "
                f"{metrics.get('circular_distance'):>8} {metrics.get('hamming_distance'):>7} "
                f"{metrics.get('box_overlap'):>10} {str(metrics.get('mirror_exact')):>11} "
                f"{str(metrics.get('box_exact')):>8}"
            )

        exact_best = max(row.exact_hit_count or 0 for row in rows)
        exact_leaders = [row.policy for row in ranked if (row.exact_hit_count or 0) == exact_best]
        lines.extend(
            (
                "",
                f"BestPolicyByExactHit: {','.join(exact_leaders)} (HitCount={exact_best})",
            )
        )
        if exact_best == 0:
            near_best = ranked[0]
            lines.append(
                f"BestPolicyByNearMiss: {near_best.policy} "
                f"(Circular={near_best.min_circular_digit_distance}, "
                f"Hamming={near_best.min_hamming_distance}, Box={near_best.max_box_overlap})"
            )

        lines.extend(
            (
                "",
                "FINAL DECISION",
                "-" * width,
                f"Decision: {decision}",
                f"Reason: {reason}",
                "PromotionDecision: DO NOT PROMOTE",
                f"ProductionSwitchRecommendedNow: {production_switch}",
            )
        )

    lines.extend(
        (
            "",
            f"REPORT_WRITTEN: {REPORT_PATH}",
            f"JSONL_WRITTEN: {JSONL_PATH}",
        )
    )
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    actuals: tuple[str, ...] | None = None
    procedure_available = False
    sp_results: dict[str, tuple[int | None, str]] = {}

    with get_conn() as connection:
        cursor = connection.cursor()
        actuals = fetch_target_winners(cursor)
        try:
            procedure_available = stored_procedure_available(cursor)
        except Exception:
            procedure_available = False

        for policy, top5 in LOCKED_BOARD.items():
            if actuals is None:
                sp_results[policy] = (None, "NOT_RUN_TARGET_UNAVAILABLE")
            elif not procedure_available:
                sp_results[policy] = (None, "UNAVAILABLE")
            else:
                try:
                    hit_count = call_sql_verifier(cursor, top5)
                    sp_results[policy] = (hit_count, "AVAILABLE")
                except Exception as exc:
                    sp_results[policy] = (
                        None,
                        f"UNAVAILABLE:{type(exc).__name__}",
                    )

    rows = [
        verify_policy(
            policy,
            top5,
            actuals,
            sp_results[policy][0],
            sp_results[policy][1],
        )
        for policy, top5 in LOCKED_BOARD.items()
    ]

    report = build_report(rows, actuals, procedure_available)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    with JSONL_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    print(report)


if __name__ == "__main__":
    main()
