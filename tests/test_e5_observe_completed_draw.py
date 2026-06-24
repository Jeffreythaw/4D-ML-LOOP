from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest


SCRIPT_PATH = Path("scripts/e5_observe_completed_draw.py")


def load_module():
    spec = importlib.util.spec_from_file_location("e5_observe_completed_draw", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load e5_observe_completed_draw.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class E5ObserveCompletedDrawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_parse_actuals_normalizes_4d_values(self) -> None:
        actuals = self.module._parse_actuals("1,23,456,7890")
        self.assertEqual(actuals, ("0001", "0023", "0456", "7890"))

    def test_parse_actuals_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.module._parse_actuals("0445,445")

    def test_parse_actuals_requires_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.module._parse_actuals(" , , ")

    def test_tier_prefers_high_when_high_and_medium_exist(self) -> None:
        tier = self.module._tier({"LAST3_MATCH", "SUFFIX2_MATCH"})
        self.assertEqual(tier, "HIGH")


if __name__ == "__main__":
    unittest.main()
