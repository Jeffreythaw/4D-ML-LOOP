from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from j4d_e5.recombiner import recombine_prefix_suffix, score_boost_from_memory
from j4d_e5.provenance import normalize_4d

MEMORY_PATH = PROJECT_ROOT / "reports" / "patches" / "e5_segment_memory_observation.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "patches" / "e5_draw_5498_recombination_observation_report.txt"

PREDICTED_TOP5 = ("4445", "4640", "9917", "9373", "5335")

# Completed actuals are only used here for post-result observation labeling.
ACTUAL_5498 = {
    "9954", "2614", "6272",
    "0324", "0327", "1364", "1835", "3726", "3800", "5816", "6608", "6989", "9564",
    "0062", "0219", "0445", "4693", "6118", "6424", "7552", "8286", "8663", "8916",
}

# Candidate provenance distilled from Step 5D/5E/5F.
PROVENANCE = {
    "4445": {
        "engine_family": "E40",
        "formula_id": "E40_FULL_HISTORY_KNOWLEDGE",
        "method_name": "E40_FULL_HISTORY_KNOWLEDGE",
        "source_prize_type": "3rd",
        "day_type": "Wednesday",
    },
    "4640": {
        "engine_family": "E2",
        "formula_id": "E2_SET_PROJECTOR",
        "method_name": "E2_SET_PROJECTOR",
        "source_prize_type": "1st",
        "day_type": "Wednesday",
    },
    "9917": {
        "engine_family": "E4",
        "formula_id": "E4_MARKOV_TRANSITION_MASS",
        "method_name": "E4_MARKOV_TRANSITION_MASS",
        "source_prize_type": "Starter",
        "day_type": "Wednesday",
    },
    "5335": {
        "engine_family": "E4",
        "formula_id": "E4_FIX_5493_TO_5494",
        "method_name": "E4_ADAPTIVE_4_UNKNOWNS_AFFINE",
        "source_prize_type": "2nd",
        "day_type": "Wednesday",
    },
}

PREFIX_SOURCES = ("4640", "9917")
SUFFIX_SOURCES = ("4445", "5335", "9917")


def main() -> int:
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))

    lines = [
        "E5 Draw 5498 Recombination Observation Report",
        "Mode: observation_only",
        "ProductionRankingChanged: false",
        f"MemoryEntries: {len(memory.get('entries', {}))}",
        "",
        "MemoryBoostByOriginalCandidate:",
    ]

    for candidate in PREDICTED_TOP5:
        boost = score_boost_from_memory(candidate, PROVENANCE.get(candidate), memory)
        lines.append(f"{candidate}: boost={boost} provenance={PROVENANCE.get(candidate)}")

    lines.extend(["", "PrefixSuffixRecombinationCandidates:"])

    generated = []
    for prefix_source in PREFIX_SOURCES:
        for suffix_source in SUFFIX_SOURCES:
            number = recombine_prefix_suffix(prefix_source, suffix_source)
            number = normalize_4d(number, field_name="recombined")
            hit = number in ACTUAL_5498
            generated.append((number, prefix_source, suffix_source, hit))
            lines.append(
                f"{number}: prefix_from={prefix_source} suffix_from={suffix_source} "
                f"post_result_hit={hit}"
            )

    unique_generated = sorted({item[0] for item in generated})
    hit_generated = sorted({item[0] for item in generated if item[3]})

    lines.extend(
        [
            "",
            f"UniqueGeneratedCount: {len(unique_generated)}",
            f"GeneratedNumbers: {', '.join(unique_generated)}",
            f"PostResultHits: {', '.join(hit_generated) if hit_generated else '(none)'}",
            "",
            "FinalDecision: E5_RECOMBINATION_OBSERVATION_READY",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {REPORT_PATH}")
    print("E5_RECOMBINATION_OBSERVATION_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
