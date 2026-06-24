from __future__ import annotations

import unittest

from scripts.run_full_history_engine_training import (
    ChronologicalTrainingDataset,
    TrainingDraw,
)
from statistical_tests import random_hit_probability
from walk_forward_validator import StrictWalkForwardValidator


class TemporalFirewallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = ChronologicalTrainingDataset(
            [
                TrainingDraw(
                    draw_no=index,
                    draw_date=f"2020-01-{index:02d}",
                    day_type="Wednesday",
                    winners=tuple(f"{value:04d}" for value in range(23)),
                )
                for index in range(1, 8)
            ]
        )

    def test_pairs_until_never_contains_future_target(self) -> None:
        for cutoff in range(2, 7):
            pairs = self.dataset.pairs_until(cutoff)
            self.assertTrue(all(pair.target_draw_no <= cutoff for pair in pairs))

    def test_random_baseline_matches_combinatorial_formula(self) -> None:
        probability = random_hit_probability(
            universe_size=10_000, winner_count=23, top_k=5
        )
        self.assertAlmostEqual(
            probability,
            1.0
            - (
                (9977 / 10000)
                * (9976 / 9999)
                * (9975 / 9998)
                * (9974 / 9997)
                * (9973 / 9996)
            ),
            places=14,
        )

    def test_validator_rejects_future_training_pair(self) -> None:
        original = self.dataset.pairs_until
        self.dataset.pairs_until = lambda cutoff: self.dataset.retrospective_pairs()  # type: ignore[method-assign]
        validator = StrictWalkForwardValidator(
            self.dataset,
            start_draw_no=3,
            end_draw_no=3,
        )
        with self.assertRaisesRegex(RuntimeError, "temporal firewall"):
            validator.predict_locked(3)
        self.dataset.pairs_until = original  # type: ignore[method-assign]


if __name__ == "__main__":
    unittest.main()
