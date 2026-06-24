from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


REQUIRED_PROVENANCE_FIELDS = (
    "candidate_number",
    "source_draw_no",
    "target_draw_no",
    "source_prize_number",
    "source_prize_type",
    "source_prize_rank",
    "source_prize_index",
    "engine_family",
    "engine_name",
    "formula_id",
    "method_name",
    "model_name",
    "matrix_id",
    "bias_id",
    "raw_score",
    "rank_before_final",
    "rank_after_final",
    "is_final_top5",
    "day_type",
    "created_at_utc",
)


def normalize_4d(value: str | int, *, field_name: str = "number") -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    if not text.isdigit():
        raise ValueError(f"{field_name} must contain only digits")
    if len(text) > 4:
        raise ValueError(f"{field_name} must be at most four digits")
    return text.zfill(4)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CandidateProvenance:
    candidate_number: str
    source_draw_no: int
    target_draw_no: int
    source_prize_number: str | None = None
    source_prize_type: str | None = None
    source_prize_rank: int | None = None
    source_prize_index: int | None = None
    engine_family: str | None = None
    engine_name: str | None = None
    formula_id: str | None = None
    method_name: str | None = None
    model_name: str | None = None
    matrix_id: str | None = None
    bias_id: str | None = None
    raw_score: float | None = None
    rank_before_final: int | None = None
    rank_after_final: int | None = None
    is_final_top5: bool = False
    day_type: str | None = None
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_number", normalize_4d(self.candidate_number, field_name="candidate_number"))
        if self.source_prize_number is not None:
            object.__setattr__(
                self,
                "source_prize_number",
                normalize_4d(self.source_prize_number, field_name="source_prize_number"),
            )
        if not self.created_at_utc:
            object.__setattr__(self, "created_at_utc", utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_candidate_provenance(row: CandidateProvenance | dict[str, Any]) -> dict[str, Any]:
    data = row.to_dict() if isinstance(row, CandidateProvenance) else dict(row)
    missing = [field for field in REQUIRED_PROVENANCE_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Missing provenance fields: {', '.join(missing)}")

    data["candidate_number"] = normalize_4d(data["candidate_number"], field_name="candidate_number")
    if data.get("source_prize_number") is not None:
        data["source_prize_number"] = normalize_4d(data["source_prize_number"], field_name="source_prize_number")
    data["source_draw_no"] = int(data["source_draw_no"])
    data["target_draw_no"] = int(data["target_draw_no"])
    data["is_final_top5"] = bool(data["is_final_top5"])

    for field in ("source_prize_rank", "source_prize_index", "rank_before_final", "rank_after_final"):
        if data.get(field) is not None:
            data[field] = int(data[field])
    if data.get("raw_score") is not None:
        data["raw_score"] = float(data["raw_score"])

    return data


def validate_provenance_rows(rows: Iterable[CandidateProvenance | dict[str, Any]]) -> list[dict[str, Any]]:
    return [validate_candidate_provenance(row) for row in rows]


def best_effort_provenance_from_candidates(
    *,
    source_draw_no: int,
    target_draw_no: int,
    candidates: Iterable[Any],
    day_type: str | None = None,
) -> list[CandidateProvenance]:
    """Adapt final candidate objects when full source-prize provenance is unavailable."""
    rows: list[CandidateProvenance] = []
    for item in candidates:
        number = getattr(item, "number", item[0] if isinstance(item, (tuple, list)) and item else None)
        if number is None:
            continue
        rank = getattr(item, "rank", None)
        score = getattr(item, "score", None)
        source = getattr(item, "source", None)
        rows.append(
            CandidateProvenance(
                candidate_number=normalize_4d(number, field_name="candidate_number"),
                source_draw_no=int(source_draw_no),
                target_draw_no=int(target_draw_no),
                engine_family=str(source) if source is not None else None,
                engine_name=str(source) if source is not None else None,
                raw_score=float(score) if score is not None else None,
                rank_before_final=int(rank) if rank is not None else None,
                rank_after_final=int(rank) if rank is not None else None,
                is_final_top5=True,
                day_type=day_type,
            )
        )
    return rows
