# E5 Candidate Provenance Contract

Status: PROVENANCE_CONTRACT_APPROVED

Each candidate provenance row describes how one predicted candidate was produced before final verification. Rows must preserve leading zeros and must be target-blind at generation time.

## Required Fields

- `candidate_number`: 4-digit predicted candidate string.
- `source_draw_no`: completed source draw used for prediction.
- `target_draw_no`: target draw being predicted.
- `source_prize_number`: 4-digit source prize number that contributed to the candidate.
- `source_prize_type`: source prize slot or category, such as First, Second, Third, Starter, Consolation, or Unknown.
- `source_prize_rank`: numeric rank within source prize ordering when available.
- `source_prize_index`: zero-based source prize index when available.
- `engine_family`: normalized engine family, such as E1, E2, E3, E4, or E40.
- `engine_name`: concrete engine name invoked by the runtime.
- `formula_id`: formula identifier when persisted or known.
- `method_name`: method or transform name.
- `model_name`: trained model name when applicable.
- `matrix_id`: matrix identifier when applicable.
- `bias_id`: learned bias identifier when applicable.
- `raw_score`: pre-final score emitted by the candidate source.
- `rank_before_final`: rank before final diversity-guard selection.
- `rank_after_final`: final rank if selected.
- `is_final_top5`: boolean indicating final visible Top 5 membership.
- `day_type`: Wednesday, Saturday, Sunday, or Special.
- `created_at_utc`: UTC timestamp when the provenance row was created.

## Current Implementation Note

The current runtime exposes aggregate Top 5, final source families, and engine-level rank snapshots, but not complete 23-source-prize x engine x formula provenance for every generated row. E5 therefore includes validators and best-effort adapters, while full provenance capture remains marked `PROVENANCE_MISSING_NEEDS_IMPLEMENTATION` until deep candidate rows include all required fields.
