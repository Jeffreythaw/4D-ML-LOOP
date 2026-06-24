#!/usr/bin/env python3
"""Read-only chronological causal column backtest for Jeffrey Quad Engine V2."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
REPORT_DIR = ROOT / "reports"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

POSITIONS = ("Pos_1", "Pos_2", "Pos_3", "Pos_4")
DIGITS = tuple(range(10))
PAIR_SPACE = 100

DRAW_SQL = """
    SELECT
        DrawNo,
        CONVERT(varchar(10), DrawDate, 120) AS DrawDateText,
        CASE
            WHEN DATENAME(WEEKDAY, DrawDate) IN ('Wednesday', 'Saturday', 'Sunday')
                THEN DATENAME(WEEKDAY, DrawDate)
            ELSE 'Special'
        END AS DayType,
        WinningNumbers
    FROM dbo.DrawHistory
    WHERE DrawNo <= ?
      AND WinningNumbers IS NOT NULL
    ORDER BY DrawNo;
"""


@dataclass(frozen=True)
class Draw:
    draw_no: int
    draw_date: str
    day_type: str
    winners: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only chronological column-by-column causal backtest. "
            "No PredictionLedger, FormulaRegistry, or schema writes."
        )
    )
    parser.add_argument("--start-draw", type=int, default=4051)
    parser.add_argument("--end-draw", type=int, default=4070)
    parser.add_argument("--top-k-digits", type=int, choices=range(1, 10), default=3)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--laplace-alpha", type=float, default=1.0)
    parser.add_argument("--day-type-min-pairs", type=int, default=25)
    parser.add_argument(
        "--output-prefix",
        default="step_163_offline_causal_column_backtest",
        help="Safe report filename prefix written under reports/.",
    )
    return parser.parse_args()


def parse_numbers(raw: object) -> tuple[str, ...]:
    values = []
    for item in str(raw or "").replace(" ", "").split(","):
        if not item:
            continue
        number = item.zfill(4)
        if len(number) == 4 and number.isdigit():
            values.append(number)
    return tuple(values)


def digits(number: str) -> tuple[int, int, int, int]:
    value = str(number).zfill(4)
    if len(value) != 4 or not value.isdigit():
        raise ValueError(f"Expected 4D number, got {number!r}")
    return tuple(int(char) for char in value)  # type: ignore[return-value]


def fetch_draws(max_draw_no: int) -> list[Draw]:
    from app.core.config import get_settings
    import pyodbc  # type: ignore

    settings = get_settings()
    connection = pyodbc.connect(
        settings.sql_connection_string(),
        timeout=120,
        autocommit=True,
    )
    cursor = connection.cursor()
    try:
        cursor.execute(DRAW_SQL, int(max_draw_no))
        rows = cursor.fetchall()
    finally:
        cursor.close()
        connection.close()

    output = []
    for row in rows:
        winners = parse_numbers(row.WinningNumbers)
        if not winners:
            continue
        output.append(
            Draw(
                draw_no=int(row.DrawNo),
                draw_date=str(row.DrawDateText),
                day_type=str(row.DayType),
                winners=winners,
            )
        )
    return output


def empty_transitions() -> list[list[list[int]]]:
    return [[[0 for _ in DIGITS] for _ in DIGITS] for _ in POSITIONS]


def empty_pair_counts() -> list[list[int]]:
    return [[0 for _ in range(PAIR_SPACE)] for _ in range(3)]


class CausalColumnState:
    def __init__(self) -> None:
        self.global_transitions = empty_transitions()
        self.day_transitions: dict[str, list[list[list[int]]]] = defaultdict(
            empty_transitions
        )
        self.global_pairs = empty_pair_counts()
        self.day_pairs: dict[str, list[list[int]]] = defaultdict(empty_pair_counts)
        self.global_training_pairs = 0
        self.day_training_pairs: Counter[str] = Counter()

    def update(self, source: Draw, target: Draw) -> None:
        source_digits = [digits(number) for number in source.winners]
        target_digits = [digits(number) for number in target.winners]
        day_matrix = self.day_transitions[source.day_type]

        for position in range(4):
            source_frequency = Counter(item[position] for item in source_digits)
            target_frequency = Counter(item[position] for item in target_digits)
            for source_digit, source_count in source_frequency.items():
                for target_digit, target_count in target_frequency.items():
                    increment = source_count * target_count
                    self.global_transitions[position][source_digit][target_digit] += increment
                    day_matrix[position][source_digit][target_digit] += increment

        day_pairs = self.day_pairs[source.day_type]
        for target_value in target_digits:
            for position in range(3):
                pair_index = target_value[position] * 10 + target_value[position + 1]
                self.global_pairs[position][pair_index] += 1
                day_pairs[position][pair_index] += 1

        self.global_training_pairs += 1
        self.day_training_pairs[source.day_type] += 1

    def distribution(
        self,
        source_numbers: Sequence[str],
        day_type: str,
        position: int,
        *,
        alpha: float,
        day_type_min_pairs: int,
    ) -> tuple[list[float], str]:
        use_day = self.day_training_pairs[day_type] >= day_type_min_pairs
        matrix = (
            self.day_transitions[day_type][position]
            if use_day
            else self.global_transitions[position]
        )
        source_frequency = Counter(digits(number)[position] for number in source_numbers)
        total_source = sum(source_frequency.values())
        probabilities = [0.0 for _ in DIGITS]

        for source_digit, source_count in source_frequency.items():
            row = matrix[source_digit]
            denominator = sum(row) + alpha * len(DIGITS)
            for target_digit in DIGITS:
                probabilities[target_digit] += (
                    source_count / total_source
                    * (row[target_digit] + alpha)
                    / denominator
                )
        return probabilities, ("DAY_TYPE" if use_day else "GLOBAL_FALLBACK")

    def pair_probability(
        self,
        left: int,
        right: int,
        day_type: str,
        position: int,
        *,
        alpha: float,
        day_type_min_pairs: int,
    ) -> float:
        use_day = self.day_training_pairs[day_type] >= day_type_min_pairs
        counts = (
            self.day_pairs[day_type][position]
            if use_day
            else self.global_pairs[position]
        )
        return (counts[left * 10 + right] + alpha) / (
            sum(counts) + alpha * PAIR_SPACE
        )


def generate_candidates(
    state: CausalColumnState,
    source: Draw,
    *,
    top_k_digits: int,
    top_n: int,
    alpha: float,
    day_type_min_pairs: int,
) -> dict:
    distributions = []
    distribution_sources = []
    ranked_digits = []
    for position in range(4):
        probabilities, source_kind = state.distribution(
            source.winners,
            source.day_type,
            position,
            alpha=alpha,
            day_type_min_pairs=day_type_min_pairs,
        )
        distributions.append(probabilities)
        distribution_sources.append(source_kind)
        ranked_digits.append(
            sorted(DIGITS, key=lambda digit: (-probabilities[digit], digit))[
                :top_k_digits
            ]
        )

    generated = []
    for values in itertools.product(*ranked_digits):
        number = "".join(str(value) for value in values)
        column_log_score = sum(
            math.log(max(distributions[position][values[position]], 1e-300))
            for position in range(4)
        )
        pair_log_score = sum(
            math.log(
                max(
                    state.pair_probability(
                        values[position],
                        values[position + 1],
                        source.day_type,
                        position,
                        alpha=alpha,
                        day_type_min_pairs=day_type_min_pairs,
                    ),
                    1e-300,
                )
            )
            for position in range(3)
        )
        total_score = column_log_score + 0.25 * pair_log_score
        generated.append(
            {
                "number": number,
                "score": total_score,
                "column_log_score": column_log_score,
                "alignment_log_score": pair_log_score,
            }
        )

    generated.sort(
        key=lambda item: (
            -item["score"],
            -item["column_log_score"],
            -item["alignment_log_score"],
            item["number"],
        )
    )
    locked = generated[:top_n]
    return {
        "locked": locked,
        "generated": generated,
        "ranked_digits": ranked_digits,
        "distributions": distributions,
        "distribution_sources": distribution_sources,
        "candidate_count": len(generated),
    }


def evaluate_locked(prediction: dict, target: Draw) -> dict:
    locked_numbers = {item["number"] for item in prediction["locked"]}
    generated_numbers = {item["number"] for item in prediction["generated"]}
    actuals = set(target.winners)
    hits = sorted(locked_numbers & actuals)
    generated_hits = sorted(generated_numbers & actuals)
    never_generated = sorted(actuals - generated_numbers)
    generated_but_dropped = sorted((actuals & generated_numbers) - locked_numbers)
    return {
        "hits": hits,
        "hit_count": len(hits),
        "generated_hits": generated_hits,
        "never_generated": never_generated,
        "generated_but_dropped": generated_but_dropped,
        "actual_count": len(actuals),
    }


def build_summary(events: Sequence[dict]) -> dict:
    draws = len(events)
    draws_with_hit = sum(item["evaluation"]["hit_count"] > 0 for item in events)
    actual_slots = sum(item["evaluation"]["actual_count"] for item in events)
    never_generated = sum(
        len(item["evaluation"]["never_generated"]) for item in events
    )
    generated_but_dropped = sum(
        len(item["evaluation"]["generated_but_dropped"]) for item in events
    )

    streak = 0
    streak_before_hits: Counter[int] = Counter()
    longest_miss_streak = 0
    for item in events:
        if item["evaluation"]["hit_count"] > 0:
            streak_before_hits[streak] += 1
            streak = 0
        else:
            streak += 1
            longest_miss_streak = max(longest_miss_streak, streak)

    return {
        "draws": draws,
        "draws_with_hit": draws_with_hit,
        "binary_hit_rate": draws_with_hit / draws if draws else None,
        "raw_hits": sum(item["evaluation"]["hit_count"] for item in events),
        "actual_slots": actual_slots,
        "never_generated_count": never_generated,
        "never_generated_rate": never_generated / actual_slots if actual_slots else None,
        "generated_but_dropped_count": generated_but_dropped,
        "generated_but_dropped_rate": (
            generated_but_dropped / actual_slots if actual_slots else None
        ),
        "longest_miss_streak": longest_miss_streak,
        "hit_after_miss_streak_frequency": {
            str(key): value for key, value in sorted(streak_before_hits.items())
        },
    }


def run_backtest(
    draws: Sequence[Draw],
    *,
    start_draw: int,
    end_draw: int,
    top_k_digits: int,
    top_n: int,
    alpha: float,
    day_type_min_pairs: int,
) -> tuple[list[dict], dict]:
    by_no = {draw.draw_no: draw for draw in draws}
    ordered = sorted(draws, key=lambda draw: draw.draw_no)
    state = CausalColumnState()

    for source, target in zip(ordered, ordered[1:]):
        if target.draw_no > start_draw:
            break
        state.update(source, target)

    events = []
    for source_draw_no in range(start_draw, end_draw + 1):
        source = by_no.get(source_draw_no)
        target = by_no.get(source_draw_no + 1)
        if source is None or target is None:
            continue

        prediction = generate_candidates(
            state,
            source,
            top_k_digits=top_k_digits,
            top_n=top_n,
            alpha=alpha,
            day_type_min_pairs=day_type_min_pairs,
        )

        locked_snapshot = tuple(item["number"] for item in prediction["locked"])
        evaluation = evaluate_locked(prediction, target)
        events.append(
            {
                "source_draw_no": source.draw_no,
                "target_draw_no": target.draw_no,
                "source_day_type": source.day_type,
                "target_day_type": target.day_type,
                "training_pairs_before_prediction": state.global_training_pairs,
                "day_training_pairs_before_prediction": state.day_training_pairs[
                    source.day_type
                ],
                "candidate_count": prediction["candidate_count"],
                "ranked_digits": prediction["ranked_digits"],
                "distribution_sources": prediction["distribution_sources"],
                "locked_top5": list(locked_snapshot),
                "locked_scores": prediction["locked"],
                "evaluation": evaluation,
                "temporal_firewall": {
                    "target_accessed_after_lock": True,
                    "training_max_target_draw_no": source.draw_no,
                },
            }
        )

        # The source->target pair enters training only after the locked prediction
        # has been evaluated. It can affect source_draw_no+1 and later, never itself.
        state.update(source, target)

    return events, build_summary(events)


def render_report(metadata: dict, summary: dict, events: Sequence[dict]) -> str:
    lines = [
        "STEP 163 — OFFLINE CAUSAL COLUMN BACKTEST — READ ONLY",
        f"GeneratedUTC: {metadata['generated_utc']}",
        "ProductionPredictionChanged: NO",
        "DBWritePerformed: NO",
        "PredictionLedgerUsed: NO",
        "DeepCandidateLedgerUsed: NO",
        f"SourceRange: {metadata['start_draw']}..{metadata['end_draw']}",
        f"TopKDigitsPerColumn: {metadata['top_k_digits']}",
        f"GeneratedCandidatesPerDraw: {metadata['top_k_digits'] ** 4}",
        f"LaplaceAlpha: {metadata['laplace_alpha']}",
        "",
        "SUMMARY",
    ]
    for key, value in summary.items():
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            "DRAW RESULTS",
            "Source Target DayType   TrainPairs Candidates Top5                         Hits NeverGenerated GeneratedButDropped",
            "------ ------ --------- ---------- ---------- ---------------------------- ---- -------------- -------------------",
        ]
    )
    for event in events:
        evaluation = event["evaluation"]
        lines.append(
            f"{event['source_draw_no']:<6} {event['target_draw_no']:<6} "
            f"{event['source_day_type']:<9} "
            f"{event['training_pairs_before_prediction']:<10} "
            f"{event['candidate_count']:<10} "
            f"{','.join(event['locked_top5']):<28} "
            f"{','.join(evaluation['hits']) or '-':<4} "
            f"{','.join(evaluation['never_generated']) or '-':<14} "
            f"{','.join(evaluation['generated_but_dropped']) or '-'}"
        )
    lines.extend(
        [
            "",
            "DIRECTION CHECK",
            "- Each of four positions is learned separately.",
            "- DayType-conditioned transitions are used when support is sufficient.",
            "- Laplace smoothing prevents zero-probability collapse.",
            "- Cartesian recombination is checked with adjacent-pair alignment.",
            "- Top5 is locked before target winners are accessed.",
            "- Never-generated and generated-but-dropped outcomes are separated.",
            "- Hit-after-miss-streak frequencies are reported; they are diagnostics, not live guarantees.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.start_draw < 1 or args.end_draw < args.start_draw:
        raise ValueError("Invalid draw range")
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")
    if args.laplace_alpha <= 0:
        raise ValueError("--laplace-alpha must be positive")
    if not re_fullmatch_safe_prefix(args.output_prefix):
        raise ValueError("--output-prefix may contain only letters, digits, underscores, and hyphens")

    started = time.perf_counter()
    draws = fetch_draws(args.end_draw + 1)
    events, summary = run_backtest(
        draws,
        start_draw=args.start_draw,
        end_draw=args.end_draw,
        top_k_digits=args.top_k_digits,
        top_n=args.top_n,
        alpha=args.laplace_alpha,
        day_type_min_pairs=args.day_type_min_pairs,
    )
    runtime = time.perf_counter() - started

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "start_draw": args.start_draw,
        "end_draw": args.end_draw,
        "top_k_digits": args.top_k_digits,
        "top_n": args.top_n,
        "laplace_alpha": args.laplace_alpha,
        "day_type_min_pairs": args.day_type_min_pairs,
        "draw_rows_cached": len(draws),
        "runtime_seconds": runtime,
        "read_only": True,
        "target_access_after_lock": True,
    }
    payload = {"metadata": metadata, "summary": summary, "events": events}
    rows = [
        {
            "row_type": "draw_result",
            **event,
        }
        for event in events
    ]
    rows.extend(
        {"row_type": "summary_metric", "metric": key, "value": value}
        for key, value in summary.items()
    )

    report_path = REPORT_DIR / f"{args.output_prefix}.txt"
    matrices_path = REPORT_DIR / f"{args.output_prefix}_matrices.json"
    rows_path = REPORT_DIR / f"{args.output_prefix}_rows.jsonl"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(metadata, summary, events), encoding="utf-8")
    matrices_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    print("STEP 163 — OFFLINE CAUSAL COLUMN BACKTEST — READ ONLY")
    print(f"DrawsEvaluated: {summary['draws']}")
    print(f"BinaryHitRate: {summary['binary_hit_rate']}")
    print(f"NeverGeneratedRate: {summary['never_generated_rate']}")
    print(f"GeneratedButDroppedRate: {summary['generated_but_dropped_rate']}")
    print(f"RuntimeSeconds: {runtime:.6f}")
    print("DBWritePerformed: NO")
    print(f"Report: {report_path}")
    return 0


def re_fullmatch_safe_prefix(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in "_-" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
