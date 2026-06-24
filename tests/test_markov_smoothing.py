from __future__ import annotations

import unittest

from markov_smoothing import MarkovObservation, SmoothedMarkovModel


class MarkovSmoothingTests(unittest.TestCase):
    def test_unseen_transition_gets_nonzero_prior_mass(self) -> None:
        model = SmoothedMarkovModel(alpha=5, state_space_size=100).fit(
            [
                MarkovObservation("0001", "0002", "Wednesday"),
                MarkovObservation("0001", "0002", "Wednesday"),
                MarkovObservation("0003", "0004", "Saturday"),
            ]
        )
        self.assertGreater(
            model.probability("0001", "0099", day_type="Wednesday"), 0.0
        )
        self.assertGreater(
            model.probability("never", "0099", day_type="Wednesday"), 0.0
        )

    def test_supported_transition_beats_unseen_transition(self) -> None:
        model = SmoothedMarkovModel(alpha=1, state_space_size=100).fit(
            [MarkovObservation("0001", "0002") for _ in range(10)]
        )
        self.assertGreater(
            model.probability("0001", "0002"),
            model.probability("0001", "0003"),
        )


if __name__ == "__main__":
    unittest.main()
