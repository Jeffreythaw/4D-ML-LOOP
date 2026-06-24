from __future__ import annotations

import unittest

import numpy as np

from ensemble_calibration import IsotonicCalibrator, LogisticCalibrator


class EnsembleCalibrationTests(unittest.TestCase):
    def test_logistic_probability_increases_with_signal(self) -> None:
        x = np.linspace(-3, 3, 400).reshape(-1, 1)
        y = (x[:, 0] > 0).astype(float)
        model = LogisticCalibrator(l2=0.1, min_samples=20).fit(x, y)
        probabilities = model.predict_proba(
            np.asarray([[-2.0], [0.0], [2.0]])
        )
        self.assertTrue(np.all(np.diff(probabilities) > 0))

    def test_isotonic_output_is_monotone(self) -> None:
        scores = np.asarray([0, 1, 2, 3, 4, 5], dtype=float)
        labels = np.asarray([0, 1, 0, 1, 1, 1], dtype=float)
        model = IsotonicCalibrator().fit(scores, labels)
        probabilities = model.predict_proba(scores)
        self.assertTrue(np.all(np.diff(probabilities) >= 0))

    def test_weighted_logistic_preserves_rare_event_base_rate(self) -> None:
        rng = np.random.default_rng(19)
        features = rng.normal(size=(200, 2))
        labels = np.concatenate([np.ones(20), np.zeros(180)])
        weights = np.concatenate(
            [np.ones(20), np.full(180, 9980 / 180)]
        )
        model = LogisticCalibrator(l2=5.0, min_samples=20).fit(
            features, labels, sample_weight=weights
        )
        mean_probability = float(np.mean(model.predict_proba(features)))
        self.assertLess(mean_probability, 0.02)


if __name__ == "__main__":
    unittest.main()
