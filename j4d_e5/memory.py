from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MEMORY_SCHEMA_VERSION = 1
MEMORY_DIMENSIONS = (
    "segment_class",
    "engine_family",
    "formula_id",
    "method_name",
    "source_prize_type",
    "day_type",
)


@dataclass
class SegmentMemoryEntry:
    segment_class: str
    engine_family: str | None = None
    formula_id: str | None = None
    method_name: str | None = None
    source_prize_type: str | None = None
    day_type: str | None = None
    success_count: int = 0
    near_success_count: int = 0
    exact_followup_count: int = 0
    last_seen_draw_no: int | None = None
    confidence_score: float = 0.0

    def key(self) -> str:
        values = [getattr(self, field) or "" for field in MEMORY_DIMENSIONS]
        return "|".join(values)

    def recompute_confidence(self) -> None:
        observations = self.success_count + self.near_success_count + self.exact_followup_count
        self.confidence_score = round(
            (
                self.success_count
                + 0.35 * self.near_success_count
                + 0.75 * self.exact_followup_count
            )
            / (observations + 5),
            6,
        )

    def to_dict(self) -> dict[str, Any]:
        self.recompute_confidence()
        return asdict(self)


def _empty_registry() -> dict[str, Any]:
    return {"schema_version": MEMORY_SCHEMA_VERSION, "entries": {}}


def load_memory(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return _empty_registry()
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("E5 memory payload must be a JSON object")
    payload.setdefault("schema_version", MEMORY_SCHEMA_VERSION)
    payload.setdefault("entries", {})
    return payload


def write_memory(path: str | Path, memory: dict[str, Any], *, no_write: bool = False) -> None:
    if no_write:
        return
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(memory, handle, indent=2, sort_keys=True)
        handle.write("\n")


def update_memory(
    memory: dict[str, Any],
    attribution_rows: Iterable[Any],
    *,
    target_draw_no: int,
) -> dict[str, Any]:
    memory.setdefault("schema_version", MEMORY_SCHEMA_VERSION)
    entries = memory.setdefault("entries", {})

    for row in attribution_rows:
        row_data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        provenance = row_data.get("provenance") or {}
        entry = SegmentMemoryEntry(
            segment_class=str(row_data["segment_class"]),
            engine_family=provenance.get("engine_family"),
            formula_id=provenance.get("formula_id"),
            method_name=provenance.get("method_name"),
            source_prize_type=provenance.get("source_prize_type"),
            day_type=provenance.get("day_type"),
        )
        key = entry.key()
        existing = entries.get(key, {})
        entry.success_count = int(existing.get("success_count", 0))
        entry.near_success_count = int(existing.get("near_success_count", 0))
        entry.exact_followup_count = int(existing.get("exact_followup_count", 0))

        if bool(row_data.get("is_exact")):
            entry.success_count += 1
        else:
            entry.near_success_count += 1

        entry.last_seen_draw_no = int(target_draw_no)
        entry.recompute_confidence()
        entries[key] = entry.to_dict()

    return memory


def update_memory_file(
    path: str | Path,
    attribution_rows: Iterable[Any],
    *,
    target_draw_no: int,
    no_write: bool = False,
) -> dict[str, Any]:
    memory = load_memory(path)
    updated = update_memory(memory, attribution_rows, target_draw_no=target_draw_no)
    write_memory(path, updated, no_write=no_write)
    return updated
