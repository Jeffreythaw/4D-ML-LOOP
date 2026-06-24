from __future__ import annotations

import unittest

from j4d_e5.provenance import normalize_4d
from j4d_e5.segments import compare_segments


class E5SegmentTests(unittest.TestCase):
    def test_last3_suffix2_and_same_position3(self) -> None:
        classes = compare_segments("4445", "0445")
        self.assertIn("LAST3_MATCH", classes)
        self.assertIn("SUFFIX2_MATCH", classes)
        self.assertIn("SAME_POSITION_3", classes)

    def test_prefix2_and_same_position2_for_4640_case(self) -> None:
        classes = compare_segments("4640", "4693")
        self.assertIn("PREFIX2_MATCH", classes)
        self.assertIn("SAME_POSITION_2", classes)

    def test_prefix2_and_same_position2_for_9917_case(self) -> None:
        classes = compare_segments("9917", "9954")
        self.assertIn("PREFIX2_MATCH", classes)
        self.assertIn("SAME_POSITION_2", classes)

    def test_leading_zeros_are_preserved(self) -> None:
        self.assertEqual(normalize_4d("445"), "0445")
        classes = compare_segments("0445", "4445")
        self.assertIn("SUFFIX2_MATCH", classes)
        self.assertNotIn("EXACT4_MATCH", classes)


if __name__ == "__main__":
    unittest.main()
