from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from j4d_e5.memory import update_memory_file


class E5MemoryTests(unittest.TestCase):
    def test_no_write_updates_memory_but_does_not_write_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            updated = update_memory_file(
                path,
                [
                    {
                        "segment_class": "PREFIX2_MATCH",
                        "is_exact": False,
                        "provenance": {
                            "engine_family": "E2",
                            "formula_id": "F1",
                            "method_name": "set_projector",
                            "source_prize_type": "Starter",
                            "day_type": "Saturday",
                        },
                    }
                ],
                target_draw_no=5498,
                no_write=True,
            )

            self.assertFalse(path.exists())
            self.assertEqual(len(updated["entries"]), 1)
            entry = next(iter(updated["entries"].values()))
            self.assertEqual(entry["near_success_count"], 1)
            self.assertEqual(entry["last_seen_draw_no"], 5498)

    def test_confidence_score_is_deterministic(self) -> None:
        row = {
            "segment_class": "PREFIX2_MATCH",
            "is_exact": False,
            "provenance": {
                "engine_family": "E2",
                "formula_id": "F1",
                "method_name": "set_projector",
                "source_prize_type": "Starter",
                "day_type": "Saturday",
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            first = update_memory_file(path, [row], target_draw_no=5498, no_write=True)
            second = update_memory_file(path, [row], target_draw_no=5498, no_write=True)

        first_entry = next(iter(first["entries"].values()))
        second_entry = next(iter(second["entries"].values()))
        self.assertEqual(first_entry["confidence_score"], second_entry["confidence_score"])
        self.assertEqual(first_entry["confidence_score"], 0.058333)


if __name__ == "__main__":
    unittest.main()
