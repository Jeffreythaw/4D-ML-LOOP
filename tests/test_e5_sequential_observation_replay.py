from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest


SCRIPT_PATH = Path("scripts/e5_sequential_observation_replay.py")


def load_module():
    spec = importlib.util.spec_from_file_location("e5_sequential_observation_replay", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load e5_sequential_observation_replay.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class E5SequentialObservationReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_tier_prefers_high(self) -> None:
        self.assertEqual(
            self.module._tier({"LAST3_MATCH", "SUFFIX2_MATCH"}),
            "HIGH",
        )

    def test_tier_medium(self) -> None:
        self.assertEqual(
            self.module._tier({"PREFIX2_MATCH"}),
            "MEDIUM",
        )

    def test_tier_none_for_low_digit_bag_only(self) -> None:
        self.assertIsNone(self.module._tier({"DIGIT_BAG_2_MATCH"}))

    def test_engine_family_maps_known_prefixes(self) -> None:
        self.assertEqual(self.module._engine_family("E40_FULL_HISTORY_KNOWLEDGE"), "E40")
        self.assertEqual(self.module._engine_family("E4_MARKOV_TRANSITION_MASS"), "E4")
        self.assertEqual(self.module._engine_family("E2_SET_PROJECTOR"), "E2")


if __name__ == "__main__":
    unittest.main()
