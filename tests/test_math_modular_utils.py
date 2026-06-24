from __future__ import annotations

import unittest

import numpy as np

from circular_affine_optimizer import CircularAffineOptimizer
from math_modular_utils import (
    affine_modular_transform,
    circular_distance,
    circular_squared_loss,
    fit_lstsq_round_mod,
)


class ModularMathTests(unittest.TestCase):
    def test_circular_distance_wraps_at_modulus(self) -> None:
        actual = np.asarray([0, 9, 1, 4], dtype=np.int16)
        predicted = np.asarray([9, 0, 9, 1], dtype=np.int16)
        np.testing.assert_array_equal(
            circular_distance(actual, predicted, 10),
            np.asarray([1, 1, 2, 3], dtype=np.int16),
        )

    def test_affine_transform_is_integer_modular(self) -> None:
        vectors = np.asarray([[9, 8, 7, 6]], dtype=np.int16)
        matrix = np.eye(4, dtype=np.int16)
        bias = np.asarray([2, 3, 4, 5], dtype=np.int16)
        np.testing.assert_array_equal(
            affine_modular_transform(vectors, matrix, bias, 10),
            np.asarray([[1, 1, 1, 1]], dtype=np.int16),
        )

    def test_optimizer_recovers_low_loss_mod10_mapping(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.integers(0, 10, size=(160, 4), dtype=np.int16)
        matrix = np.eye(4, dtype=np.int16)
        bias = np.asarray([1, 3, 5, 7], dtype=np.int16)
        target = affine_modular_transform(source, matrix, bias, 10)
        model = CircularAffineOptimizer(
            modulus=10,
            min_samples=40,
            validation_fraction=0.2,
            complexity_penalty=0.0,
            random_restarts=3,
            max_passes=10,
            seed=9,
        ).fit(source, target)
        self.assertEqual(
            circular_squared_loss(target, model.predict(source), 10), 0.0
        )

    def test_circular_optimizer_not_worse_than_legacy_on_training_objective(self) -> None:
        rng = np.random.default_rng(13)
        source = rng.integers(0, 5, size=(100, 4), dtype=np.int16)
        target = np.mod(source * 3 + 4, 5).astype(np.int16)
        legacy = fit_lstsq_round_mod(source, target, 5)
        model = CircularAffineOptimizer(
            modulus=5,
            min_samples=20,
            validation_fraction=0.0,
            complexity_penalty=0.0,
            random_restarts=1,
            max_passes=8,
            seed=3,
        ).fit(source, target)
        self.assertLessEqual(
            circular_squared_loss(target, model.predict(source), 5),
            circular_squared_loss(target, legacy.predict(source), 5),
        )


if __name__ == "__main__":
    unittest.main()
