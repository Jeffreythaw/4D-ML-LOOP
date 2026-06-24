from __future__ import annotations

import json
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for value in (ROOT, BACKEND):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.run_causal_40year_column_learning import (
    CausalColumnLearner,
    LockedCandidates,
    artifact_envelope,
    entropy,
    lock_candidates,
    normalized,
    random_baseline,
    verify_after_lock,
)
from scripts.run_full_history_engine_training import (
    ChronologicalTrainingDataset,
    TrainingDraw,
)


def dataset(count: int = 30) -> ChronologicalTrainingDataset:
    start = date(2000, 1, 1)
    draws = []
    for draw_no in range(1, count + 1):
        winners = tuple(
            f"{(draw_no * 317 + index * 43) % 10000:04d}"
            for index in range(23)
        )
        draws.append(
            TrainingDraw(
                draw_no=draw_no,
                draw_date=(start + timedelta(days=draw_no)).isoformat(),
                day_type=("Wednesday", "Saturday", "Sunday")[draw_no % 3],
                winners=winners,
            )
        )
    return ChronologicalTrainingDataset(draws)


class Causal40YearColumnLearningTests(unittest.TestCase):
    def setUp(self):
        self.dataset = dataset()

    def test_dataset_chronological_order(self):
        numbers = [draw.draw_no for draw in self.dataset.draws]
        self.assertEqual(numbers, sorted(numbers))

    def test_cutoff_enforcement(self):
        pairs = self.dataset.pairs_until(12)
        self.assertTrue(all(pair.target_draw_no <= 12 for pair in pairs))

    def test_random_baseline(self):
        expected = 1 - ((10000 - 23) / 10000) ** 5
        self.assertAlmostEqual(random_baseline(), expected)

    def test_column_matrices_digits_zero_to_nine(self):
        learner = CausalColumnLearner().fit(self.dataset.pairs_until(20))
        self.assertEqual(len(learner.conditional), 4)
        for matrix in learner.conditional:
            self.assertEqual(len(matrix), 10)
            self.assertTrue(all(len(row) == 10 for row in matrix))

    def test_laplace_has_no_zero_probability(self):
        values = normalized([0] * 10, alpha=1.0)
        self.assertTrue(all(value > 0 for value in values))
        self.assertAlmostEqual(sum(values), 1.0)

    def test_entropy_finite(self):
        self.assertTrue(math.isfinite(entropy(normalized([0] * 10))))

    def test_generated_candidates_are_4d_and_lock_precedes_verify(self):
        learner = CausalColumnLearner().fit(self.dataset.pairs_until(20))
        source = self.dataset._by_no[21]
        target = self.dataset._by_no[22]
        locked = lock_candidates(learner, source, target.draw_no)
        self.assertFalse(locked.target_seen_before_lock)
        self.assertEqual(len(locked.candidates), 5)
        self.assertTrue(
            all(len(number) == 4 and number.isdigit() for number in locked.candidates)
        )
        verified = verify_after_lock(locked, target)
        self.assertTrue(verified["target_seen_after_lock"])
        self.assertFalse(verified["target_seen_before_lock"])

    def test_never_generated_vs_generated_but_dropped(self):
        target = TrainingDraw(
            draw_no=2,
            draw_date="2000-01-02",
            day_type="Wednesday",
            winners=("0001", "0002", "9999"),
        )
        pool = (
            {"number": "0000", "score": 3.0, "components": {"x": 1.0}},
            {"number": "0001", "score": 2.0, "components": {"x": 0.8}},
            {"number": "0002", "score": 1.0, "components": {"x": 0.2}},
        )
        locked = LockedCandidates(
            source_draw_no=1,
            target_draw_no=2,
            candidates=("0000", "0001"),
            candidate_hash="hash",
            target_seen_before_lock=False,
            ranked_pool=pool,
            column_top_digits=((0,), (0,), (0,), (0, 1, 2)),
        )
        verified = verify_after_lock(locked, target)
        self.assertEqual(verified["generated_but_dropped"], ["0002"])
        self.assertEqual(verified["never_generated"], ["9999"])

    def test_retrospective_artifact_not_for_live(self):
        artifact = artifact_envelope(
            name="test",
            training_mode="retrospective_full_history",
            dataset=self.dataset,
            pair_count=len(self.dataset.retrospective_pairs()),
            payload={"ok": True},
        )
        self.assertTrue(artifact["not_for_live_prediction"])
        self.assertIsNotNone(artifact["retrospective_label"])
        copied = dict(artifact)
        digest = copied.pop("sha256_hash")
        self.assertEqual(len(digest), 64)
        json.dumps(copied)


if __name__ == "__main__":
    unittest.main()
