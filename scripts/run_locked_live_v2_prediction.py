from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from shadow_hybrid_engine_v2 import (
    ENGINE_NAME,
    box_overlap,
    generate_shadow_hybrid_v2_top5,
    get_conn,
    hamming_distance,
    mirror_signature,
    parse_numbers,
    z4,
)


REPORT_PATH = PROJECT_ROOT / "reports" / "step_151c_locked_live_v2_prediction.txt"
JSON_PATH = PROJECT_ROOT / "reports" / "step_151c_locked_live_v2_prediction.json"
LOCAL_TIMEZONE = ZoneInfo("Asia/Singapore")
TOP_K = 5


def fetch_latest_source_metadata(cursor) -> dict:
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
        raise RuntimeError("DrawHistory has no source draw")
    return {
        "draw_no": int(row.DrawNo),
        "draw_date": str(row.DrawDateText) if row.DrawDateText else None,
        "day_type": str(row.DayType or "Unknown"),
    }


def fetch_target_winners(cursor, target_draw_no: int) -> tuple[str, ...] | None:
    row = cursor.execute(
        """
        SELECT WinningNumbers
        FROM dbo.DrawHistory
        WHERE DrawNo = ?;
        """,
        target_draw_no,
    ).fetchone()
    if row is None or not row.WinningNumbers:
        return None
    return parse_numbers(row.WinningNumbers)


def circular_digit_distance(left: str, right: str) -> int:
    total = 0
    for left_digit, right_digit in zip(z4(left), z4(right)):
        delta = abs(int(left_digit) - int(right_digit))
        total += min(delta, 10 - delta)
    return total


def box_signature(number: str) -> str:
    return "".join(sorted(z4(number)))


def near_miss_diagnostics(top5: list[str], actuals: tuple[str, ...]) -> dict:
    pairs = [(candidate, actual) for candidate in top5 for actual in actuals]
    best_candidate, best_actual = min(
        pairs,
        key=lambda item: (
            circular_digit_distance(item[0], item[1]),
            hamming_distance(item[0], item[1]),
            -box_overlap(item[0], item[1]),
            item[0],
            item[1],
        ),
    )
    return {
        "min_circular_digit_distance": min(
            circular_digit_distance(candidate, actual)
            for candidate, actual in pairs
        ),
        "min_hamming_distance": min(
            hamming_distance(candidate, actual) for candidate, actual in pairs
        ),
        "max_box_overlap": max(
            box_overlap(candidate, actual) for candidate, actual in pairs
        ),
        "mirror_exact_hidden_hits": sum(
            candidate != actual
            and mirror_signature(candidate) == mirror_signature(actual)
            for candidate, actual in pairs
        ),
        "box_exact_hidden_hits": sum(
            candidate != actual
            and box_signature(candidate) == box_signature(actual)
            for candidate, actual in pairs
        ),
        "best_candidate": best_candidate,
        "best_actual": best_actual,
        "best_pair": {
            "circular_digit_distance": circular_digit_distance(
                best_candidate, best_actual
            ),
            "hamming_distance": hamming_distance(best_candidate, best_actual),
            "box_overlap": box_overlap(best_candidate, best_actual),
            "mirror_exact": (
                best_candidate != best_actual
                and mirror_signature(best_candidate) == mirror_signature(best_actual)
            ),
            "box_exact": (
                best_candidate != best_actual
                and box_signature(best_candidate) == box_signature(best_actual)
            ),
        },
    }


def canonical_lock(result: dict) -> tuple[dict, str, str]:
    payload = {
        "engine_name": result["engine_name"],
        "source_draw_no": result["source_draw_no"],
        "target_draw_no": result["target_draw_no"],
        "top5": result["top5"],
        "candidate_details": result["candidate_details"],
    }
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    lock_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return payload, canonical_json, lock_hash


def validate_lock(result: dict) -> dict:
    top5 = result["top5"]
    invalid_count = sum(
        len(number) != 4 or not number.isdigit() or not 0 <= int(number) <= 9999
        for number in top5
    )
    duplicate_count = len(top5) - len(set(top5))
    candidate_history_violations = sum(
        int(item.get("history_max_draw_no", 0)) > result["source_draw_no"]
        for item in result["candidate_details"]
    )
    return {
        "candidate_history_violations": candidate_history_violations,
        "invalid_candidate_count": invalid_count + int(len(top5) != TOP_K),
        "duplicate_candidate_count": duplicate_count,
        "target_leakage_status": "PASS_GENERATED_BEFORE_TARGET_QUERY",
        "temporal_firewall_ok": (
            bool(result["temporal_firewall_ok"])
            and candidate_history_violations == 0
            and invalid_count == 0
            and duplicate_count == 0
            and len(top5) == TOP_K
        ),
    }


def build_report(payload: dict) -> str:
    width = 156
    lines = [
        "=" * width,
        "STEP 151C — LOCKED LIVE V2 PREDICTION — REPORT ONLY",
        "=" * width,
        f"EngineName: {payload['engine_name']}",
        "ProductionMathChanged: NO",
        "APIChanged: NO",
        "FrontendChanged: NO",
        "SQLSchemaChanged: NO",
        "DeploymentChanged: NO",
        "PredictionLedgerWritePerformed: NO",
        "ProductionWritePerformed: NO",
        "",
        "SOURCE / TARGET",
        "-" * width,
        f"SourceDrawNo: {payload['source_draw_no']}",
        f"TargetDrawNo: {payload['target_draw_no']}",
        f"SourceDrawDate: {payload['source_draw_metadata']['draw_date']}",
        f"SourceDayType: {payload['source_draw_metadata']['day_type']}",
        f"TargetAvailable: {'YES' if payload['target_available'] else 'NO'}",
        "",
        "V2 MACRO STATE",
        "-" * width,
        f"FullLedgerPreMissStreak: {payload['full_ledger_pre_miss_streak']}",
        f"RiskBucket: {payload['risk_bucket']}",
        f"BaselineTop5: {payload['baseline_top5']}",
        f"V2Top5: {payload['top5']}",
        f"BaselineKeptCount: {payload['baseline_kept_count']}",
        f"ReplacementUsed: {'YES' if payload['replacement_used'] else 'NO'}",
        f"ReplacementDetail: {payload['replacement_detail']}",
        f"Warnings: {payload['warnings'] or ['NONE']}",
        "",
        "LOCKED TOP5",
        "-" * width,
        "Rank Number Family           Score      Supports                                             Reason",
    ]
    for item in payload["candidate_details"]:
        lines.append(
            f"{item['rank']:>4} {item['number']:<6} {item['family']:<16} "
            f"{item['score']:>10.6f} {str(item['supports']):<52} {item['reason']}"
        )

    lines.extend(
        (
            "",
            "CANDIDATE LOCK",
            "-" * width,
            f"CandidateLockHash: {payload['candidate_lock_hash']}",
            f"CandidateLockCreatedAtLocal: {payload['candidate_lock_created_at_local']}",
            f"CandidateLockCreatedAtUTC: {payload['candidate_lock_created_at_utc']}",
            f"CandidateSource: {payload['engine_name']}",
            f"CanonicalPayloadFilePath: {JSON_PATH}",
            "CanonicalPayloadEncoding: sorted keys, compact separators, UTF-8",
            "ProductionWritePerformed: NO",
            "PredictionLedgerWritePerformed: NO",
            "",
            "VERIFICATION STATUS",
            "-" * width,
            f"VerificationStatus: {payload['verification_status']}",
        )
    )
    if payload["target_available"]:
        lines.extend(
            (
                f"ExactHitCount: {payload['exact_hit_count']}",
                f"NearMissDiagnostics: {payload['near_miss_diagnostics']}",
                "VerificationOrdering: candidate lock created before target winner query",
                "Decision: LOCKED_RESULT_VERIFIED_REPORT_ONLY",
            )
        )
    else:
        lines.extend(
            (
                "ExactHitCount: NULL",
                "NearMissDiagnostics: NULL",
                "Decision: WAIT_FOR_TARGET_RESULT",
            )
        )
    lines.extend(
        (
            "ProductionSwitchRecommendedNow: NO",
            "",
            "TEMPORAL FIREWALL VALIDATION",
            "-" * width,
            f"CandidateHistoryViolations: {payload['firewall_validation']['candidate_history_violations']}",
            f"InvalidCandidateCount: {payload['firewall_validation']['invalid_candidate_count']}",
            f"DuplicateCandidateCount: {payload['firewall_validation']['duplicate_candidate_count']}",
            f"TargetLeakageStatus: {payload['firewall_validation']['target_leakage_status']}",
            f"TemporalFirewallOK: {'YES' if payload['temporal_firewall_ok'] else 'NO'}",
            "",
            "FINAL",
            "-" * width,
            "LOCKED_LIVE_V2_PREDICTION_READY",
            "Do not publish as production until human approval.",
            "ProductionSwitchRecommendedNow: NO",
            "",
            f"REPORT_WRITTEN: {REPORT_PATH}",
            f"LOCK_PAYLOAD_WRITTEN: {JSON_PATH}",
        )
    )
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_conn() as connection:
        source_metadata = fetch_latest_source_metadata(connection.cursor())

    source_draw_no = int(source_metadata["draw_no"])
    result = generate_shadow_hybrid_v2_top5(source_draw_no)
    if result["engine_name"] != ENGINE_NAME:
        raise RuntimeError("Unexpected shadow engine name")
    if result["source_draw_no"] != source_draw_no:
        raise RuntimeError("Engine returned a different source draw")

    lock_payload, canonical_json, lock_hash = canonical_lock(result)
    firewall = validate_lock(result)
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(LOCAL_TIMEZONE)

    locked_output = {
        "engine_name": result["engine_name"],
        "source_draw_no": result["source_draw_no"],
        "target_draw_no": result["target_draw_no"],
        "target_available": False,
        "source_draw_metadata": source_metadata,
        "full_ledger_pre_miss_streak": result["full_ledger_pre_miss_streak"],
        "risk_bucket": result["risk_bucket"],
        "baseline_top5": result["baseline_top5"],
        "top5": result["top5"],
        "baseline_kept_count": result["baseline_kept_count"],
        "candidate_details": result["candidate_details"],
        "replacement_used": result["replacement_used"],
        "replacement_detail": result["replacement_detail"],
        "warnings": result["warnings"],
        "candidate_lock_hash": lock_hash,
        "candidate_lock_created_at_local": now_local.isoformat(),
        "candidate_lock_created_at_utc": now_utc.isoformat(),
        "candidate_lock_payload": lock_payload,
        "candidate_lock_canonical_json": canonical_json,
        "verification_status": "UNVERIFIED",
        "exact_hit_count": None,
        "near_miss_diagnostics": None,
        "production_write_performed": False,
        "prediction_ledger_write_performed": False,
        "temporal_firewall_ok": firewall["temporal_firewall_ok"],
        "firewall_validation": firewall,
    }

    # Candidate generation and lock are complete before the target query below.
    with get_conn() as connection:
        actuals = fetch_target_winners(
            connection.cursor(), result["target_draw_no"]
        )

    if actuals is not None:
        actual_set = set(actuals)
        locked_output["target_available"] = True
        locked_output["verification_status"] = "VERIFIED_AFTER_LOCK"
        locked_output["exact_hit_count"] = sum(
            number in actual_set for number in result["top5"]
        )
        locked_output["near_miss_diagnostics"] = near_miss_diagnostics(
            result["top5"], actuals
        )

    JSON_PATH.write_text(
        json.dumps(locked_output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = build_report(locked_output)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
