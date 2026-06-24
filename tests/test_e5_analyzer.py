from __future__ import annotations

import unittest

from j4d_e5.analyzer import analyze_completed_draw_segments


PREDICTED_TOP5 = ("4445", "4640", "9917", "9373", "5335")

# Completed-draw fixture for the corrected known Draw 5498 actual 23 results.
# This fixture is post-completion evidence only and must never be used for prediction.
DRAW_5498_ACTUALS = (
    "9954",
    "2614",
    "6272",
    "0324",
    "0327",
    "1364",
    "1835",
    "3726",
    "3800",
    "5816",
    "6608",
    "6989",
    "9564",
    "0062",
    "0219",
    "0445",
    "4693",
    "6118",
    "6424",
    "7552",
    "8286",
    "8663",
    "8916",
)


class E5AnalyzerTests(unittest.TestCase):
    def test_draw_5498_top5_produces_expected_near_hit_classes(self) -> None:
        result = analyze_completed_draw_segments(
            target_draw_no=5498,
            predicted_candidates=PREDICTED_TOP5,
            actual_numbers=DRAW_5498_ACTUALS,
            no_write=True,
        )

        observed = {
            (row.candidate_number, row.actual_number, row.segment_class)
            for row in result.attribution_rows
        }
        self.assertIn(("4445", "0445", "LAST3_MATCH"), observed)
        self.assertIn(("4445", "0445", "SUFFIX2_MATCH"), observed)
        self.assertIn(("4445", "0445", "SAME_POSITION_3"), observed)
        self.assertIn(("4640", "4693", "PREFIX2_MATCH"), observed)
        self.assertIn(("4640", "4693", "SAME_POSITION_2"), observed)
        self.assertIn(("9917", "9954", "PREFIX2_MATCH"), observed)
        self.assertIn(("9917", "9954", "SAME_POSITION_2"), observed)
        self.assertIn(("5335", "1835", "SAME_POSITION_2"), observed)
        self.assertIn(("9917", "8916", "SAME_POSITION_2"), observed)

    def test_exact_hit_count_is_zero(self) -> None:
        result = analyze_completed_draw_segments(
            target_draw_no=5498,
            predicted_candidates=PREDICTED_TOP5,
            actual_numbers=DRAW_5498_ACTUALS,
            no_write=True,
        )

        self.assertEqual(result.exact_hit_count, 0)

    def test_actual_numbers_are_explicit_post_completion_input(self) -> None:
        result = analyze_completed_draw_segments(
            target_draw_no=5498,
            predicted_candidates=PREDICTED_TOP5,
            actual_numbers=DRAW_5498_ACTUALS,
            no_write=True,
        )

        self.assertTrue(result.post_completion_input_required)
        self.assertTrue(result.no_write)

        with self.assertRaisesRegex(ValueError, "after draw completion"):
            analyze_completed_draw_segments(
                target_draw_no=5498,
                predicted_candidates=PREDICTED_TOP5,
                actual_numbers=(),
                no_write=True,
            )


if __name__ == "__main__":
    unittest.main()
