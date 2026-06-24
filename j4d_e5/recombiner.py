from __future__ import annotations

from typing import Any

from .provenance import normalize_4d


E5_RECOMBINATION_ENABLED_DEFAULT = False


def score_boost_from_memory(candidate: str | int, provenance: dict[str, Any] | None, memory: dict[str, Any]) -> float:
    """
    Observation-only score scaffold.

    The returned value is intended for future experiments and is not wired into
    production ranking.
    """
    normalize_4d(candidate, field_name="candidate")
    if not provenance:
        return 0.0

    entries = memory.get("entries", {}) if isinstance(memory, dict) else {}
    if not entries:
        return 0.0

    matches = []
    for entry in entries.values():
        if provenance.get("engine_family") and entry.get("engine_family") != provenance.get("engine_family"):
            continue
        if provenance.get("formula_id") and entry.get("formula_id") != provenance.get("formula_id"):
            continue
        if provenance.get("method_name") and entry.get("method_name") != provenance.get("method_name"):
            continue
        matches.append(float(entry.get("confidence_score", 0.0)))

    return round(max(matches), 6) if matches else 0.0


def recombine_prefix_suffix(prefix_source: str | int, suffix_source: str | int) -> str:
    prefix = normalize_4d(prefix_source, field_name="prefix_source")
    suffix = normalize_4d(suffix_source, field_name="suffix_source")
    return f"{prefix[:2]}{suffix[2:]}"
