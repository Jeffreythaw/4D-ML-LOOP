from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for value in (ROOT, BACKEND):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.run_full_history_engine_training import (
    ChronologicalTrainingDataset,
    TrainingDraw,
)
from scripts.run_sequential_adaptive_causal_loop import (
    MAX_SINGLE_EVIDENCE_DELTA,
    ActiveOfflineRegistry,
    CandidateLock,
    HiddenTargetVerifier,
    SequentialMetrics,
    classify_residual,
    column_residual,
    deterministic_lock_hash,
)
from scripts.run_causal_40year_column_learning import RANDOM_TOP5_23


def draws(count: int = 12) -> list[TrainingDraw]:
    start = date(2020, 1, 1)
    output = []
    for draw_no in range(1, count + 1):
        winners = tuple(
            f"{(draw_no * 101 + index * 17) % 10000:04d}"
            for index in range(23)
        )
        output.append(
            TrainingDraw(
                draw_no=draw_no,
                draw_date=(start + timedelta(days=draw_no)).isoformat(),
                day_type="Wednesday",
                winners=winners,
            )
        )
    return output


class SequentialAdaptiveCausalLoopTests(unittest.TestCase):
    def test_target_not_accessible_before_lock(self):
        verifier = HiddenTargetVerifier(draws())
        invalid = CandidateLock(
            source_draw_no=1,
            target_draw_no=2,
            locked_top5=("0000",),
            candidate_hash="x",
            score_components=(),
            registry_version=1,
            created_at="now",
            target_seen_before_lock=True,
        )
        with self.assertRaises(RuntimeError):
            verifier.verify(invalid)

    def test_candidate_lock_hash_deterministic(self):
        scores = [{"number": "0001", "score": 1.25}]
        first = deterministic_lock_hash(1, 2, ["0001"], scores, 1)
        second = deterministic_lock_hash(1, 2, ["0001"], scores, 1)
        self.assertEqual(first, second)

    def test_verifier_only_after_lock(self):
        verifier = HiddenTargetVerifier(draws())
        lock = CandidateLock(
            source_draw_no=1,
            target_draw_no=2,
            locked_top5=draws()[1].winners[:5],
            candidate_hash="x",
            score_components=(),
            registry_version=1,
            created_at="now",
        )
        result = verifier.verify(lock)
        self.assertEqual(verifier.verify_calls, 1)
        self.assertTrue(result["target_seen_after_lock"])

    def test_correction_availability_for_next_source(self):
        registry = ActiveOfflineRegistry()
        correction = registry.add_correction(
            correction_type="COLUMN_RESIDUAL_BOOST",
            target_draw_no=100,
            reason_code="WRONG_COLUMN",
            feature_key="column:0:1:2",
            requested_delta=0.05,
            evidence_summary={},
        )
        self.assertEqual(correction.available_from_source_draw_no, 100)
        self.assertNotIn(correction, registry.available_corrections(99))
        self.assertIn(correction, registry.available_corrections(100))

    def test_never_generated_classification(self):
        lock = CandidateLock(
            1,
            2,
            ("0000",),
            "x",
            (),
            1,
            "now",
        )
        pool = [{"number": "0000", "score": 1.0, "components": {"x": 1.0}}]
        result = classify_residual(
            lock,
            pool,
            {"target_winners_post_lock": ["9999"]},
        )
        self.assertEqual(result["primary_reason"], "NEVER_GENERATED")

    def test_generated_but_dropped_classification(self):
        lock = CandidateLock(
            1,
            2,
            ("0000",),
            "x",
            (),
            1,
            "now",
        )
        pool = [
            {"number": "0000", "score": 2.0, "components": {"x": 1.0}},
            {"number": "9999", "score": 1.0, "components": {"x": 0.2}},
        ]
        result = classify_residual(
            lock,
            pool,
            {"target_winners_post_lock": ["9999"]},
        )
        self.assertEqual(
            result["primary_reason"], "GENERATED_BUT_DROPPED"
        )

    def test_column_residual_computation(self):
        source = draws()[0]
        result = column_residual(
            source,
            ["9876"],
            [[0, 1], [0, 1], [0, 1], [0, 1]],
        )
        self.assertEqual(result[0]["wrong_positions"], [0, 1, 2, 3])

    def test_single_draw_correction_weight_cap(self):
        registry = ActiveOfflineRegistry()
        correction = registry.add_correction(
            correction_type="DROP_REPAIR",
            target_draw_no=2,
            reason_code="GENERATED_BUT_DROPPED",
            feature_key="drop:test",
            requested_delta=10.0,
            evidence_summary={},
            support_count=1,
        )
        self.assertLessEqual(
            abs(correction.capped_weight_delta),
            MAX_SINGLE_EVIDENCE_DELTA,
        )

    def test_rolling_metrics(self):
        metrics = SequentialMetrics()
        for hit in (0, 1, 0, 1):
            metrics.update("Wednesday", hit, None)
        summary = metrics.summary()
        self.assertEqual(summary["total_draws_evaluated"], 4)
        self.assertEqual(summary["draws_with_hit"], 2)
        self.assertAlmostEqual(summary["rolling_windows"]["1W"]["hit_rate"], 2 / 3)

    def test_random_baseline(self):
        expected = 1 - ((10000 - 23) / 10000) ** 5
        self.assertAlmostEqual(RANDOM_TOP5_23, expected)

    def test_no_database_write_path(self):
        source = Path(
            ROOT / "scripts/run_sequential_adaptive_causal_loop.py"
        ).read_text().upper()
        for token in (
            "INSERT INTO ",
            "UPDATE DBO.",
            "DELETE FROM ",
            "MERGE DBO.",
            "CREATE TABLE ",
            "ALTER TABLE ",
            "DROP TABLE ",
            "TRUNCATE TABLE ",
        ):
            self.assertNotIn(token, source)
        self.assertNotIn("CURSOR.EXECUTE", source)


if __name__ == "__main__":
    unittest.main()
