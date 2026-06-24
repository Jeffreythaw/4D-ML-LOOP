from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .provenance import normalize_4d, validate_provenance_rows
from .segments import compare_segments


@dataclass(frozen=True)
class SegmentAttributionRow:
    target_draw_no: int
    candidate_number: str
    actual_number: str
    segment_class: str
    is_exact: bool
    provenance_available: bool
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentAnalysisResult:
    target_draw_no: int
    predicted_candidates: tuple[str, ...]
    actual_numbers: tuple[str, ...]
    exact_hit_count: int
    attribution_rows: tuple[SegmentAttributionRow, ...]
    provenance_available: bool
    no_write: bool
    post_completion_input_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_draw_no": self.target_draw_no,
            "predicted_candidates": list(self.predicted_candidates),
            "actual_numbers": list(self.actual_numbers),
            "exact_hit_count": self.exact_hit_count,
            "attribution_rows": [row.to_dict() for row in self.attribution_rows],
            "provenance_available": self.provenance_available,
            "no_write": self.no_write,
            "post_completion_input_required": self.post_completion_input_required,
        }


def analyze_completed_draw_segments(
    *,
    target_draw_no: int,
    predicted_candidates: Iterable[str | int],
    actual_numbers: Iterable[str | int],
    provenance_rows: Iterable[dict[str, Any]] | None = None,
    no_write: bool = True,
) -> SegmentAnalysisResult:
    """
    Compare predictions with actual numbers after a draw has completed.

    Callers must supply completed-draw actuals explicitly. This function performs
    no target loading and no persistence by itself.
    """
    predictions = tuple(normalize_4d(value, field_name="predicted_candidate") for value in predicted_candidates)
    actuals = tuple(normalize_4d(value, field_name="actual_number") for value in actual_numbers)
    if not predictions:
        raise ValueError("predicted_candidates cannot be empty")
    if not actuals:
        raise ValueError("actual_numbers must be supplied after draw completion")

    provenance_by_candidate: dict[str, dict[str, Any]] = {}
    if provenance_rows is not None:
        for row in validate_provenance_rows(provenance_rows):
            provenance_by_candidate.setdefault(row["candidate_number"], row)

    rows: list[SegmentAttributionRow] = []
    actual_set = set(actuals)
    exact_hit_count = sum(1 for candidate in predictions if candidate in actual_set)

    for candidate in predictions:
        provenance = provenance_by_candidate.get(candidate)
        for actual in actuals:
            classes = compare_segments(candidate, actual)
            for segment_class in sorted(classes):
                rows.append(
                    SegmentAttributionRow(
                        target_draw_no=int(target_draw_no),
                        candidate_number=candidate,
                        actual_number=actual,
                        segment_class=segment_class,
                        is_exact=segment_class == "EXACT4_MATCH",
                        provenance_available=provenance is not None,
                        provenance=provenance,
                    )
                )

    return SegmentAnalysisResult(
        target_draw_no=int(target_draw_no),
        predicted_candidates=predictions,
        actual_numbers=actuals,
        exact_hit_count=exact_hit_count,
        attribution_rows=tuple(rows),
        provenance_available=bool(provenance_by_candidate),
        no_write=bool(no_write),
    )
