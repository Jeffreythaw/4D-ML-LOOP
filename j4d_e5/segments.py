from __future__ import annotations

from collections import Counter

from .provenance import normalize_4d


SEGMENT_CLASSES = (
    "EXACT4_MATCH",
    "LAST3_MATCH",
    "PREFIX3_MATCH",
    "PREFIX2_MATCH",
    "SUFFIX2_MATCH",
    "MIDDLE2_MATCH",
    "SAME_POSITION_3",
    "SAME_POSITION_2",
    "PAIR_13_MATCH",
    "PAIR_14_MATCH",
    "PAIR_24_MATCH",
    "DIGIT_BAG_3_MATCH",
    "DIGIT_BAG_2_MATCH",
)


def _same_position_count(predicted: str, actual: str) -> int:
    return sum(left == right for left, right in zip(predicted, actual))


def _digit_bag_overlap(predicted: str, actual: str) -> int:
    return sum((Counter(predicted) & Counter(actual)).values())


def compare_segments(predicted: str | int, actual: str | int) -> set[str]:
    left = normalize_4d(predicted, field_name="predicted")
    right = normalize_4d(actual, field_name="actual")
    classes: set[str] = set()

    if left == right:
        classes.add("EXACT4_MATCH")
    if left[1:] == right[1:]:
        classes.add("LAST3_MATCH")
    if left[:3] == right[:3]:
        classes.add("PREFIX3_MATCH")
    if left[:2] == right[:2]:
        classes.add("PREFIX2_MATCH")
    if left[2:] == right[2:]:
        classes.add("SUFFIX2_MATCH")
    if left[1:3] == right[1:3]:
        classes.add("MIDDLE2_MATCH")

    same_positions = _same_position_count(left, right)
    if same_positions == 3:
        classes.add("SAME_POSITION_3")
    if same_positions == 2:
        classes.add("SAME_POSITION_2")

    if left[0] == right[0] and left[2] == right[2]:
        classes.add("PAIR_13_MATCH")
    if left[0] == right[0] and left[3] == right[3]:
        classes.add("PAIR_14_MATCH")
    if left[1] == right[1] and left[3] == right[3]:
        classes.add("PAIR_24_MATCH")

    bag_overlap = _digit_bag_overlap(left, right)
    if bag_overlap >= 3:
        classes.add("DIGIT_BAG_3_MATCH")
    if bag_overlap >= 2:
        classes.add("DIGIT_BAG_2_MATCH")

    return classes


def segment_score_classes(predicted: str | int, actual: str | int) -> dict[str, object]:
    left = normalize_4d(predicted, field_name="predicted")
    right = normalize_4d(actual, field_name="actual")
    classes = compare_segments(left, right)
    same_positions = _same_position_count(left, right)
    bag_overlap = _digit_bag_overlap(left, right)
    return {
        "predicted": left,
        "actual": right,
        "classes": sorted(classes),
        "same_position_count": same_positions,
        "digit_bag_overlap": bag_overlap,
        "is_exact": "EXACT4_MATCH" in classes,
        "is_near_hit": bool(classes - {"EXACT4_MATCH"}),
    }
