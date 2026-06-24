# E5 Architecture Lock

Status: ARCHITECTURE_LOCKED

## Locked Production Path

For predicting target draw `N+1` from completed source draw `N`:

1. Load all 23 actual source prize numbers for Draw `N`.
2. Convert each source prize into a 4-digit vector, preserving leading zeros.
3. Run the locked 8-engine runtime over the 23 source prize vectors.
4. Aggregate candidate votes from the engine outputs.
5. Merge duplicate candidate numbers.
6. Rank with the existing diversity guard.
7. Return the visible Top 5 from the aggregate path only.

The visible production Top 5 must come from the 23-source-prize x 8-engine aggregate path. E5 segment attribution is observation-only until explicitly enabled in a later production decision.

## Temporal Firewall Rules

Prediction for target draw `N+1` may use only information available at or before completed source draw `N`.

Production prediction code must not:

- read target draw actual numbers before final Top 5 generation;
- call SQL verification before final Top 5 generation;
- expose hidden winners in API prediction metadata;
- use post-result segment attribution as an input to the same prediction.

Post-completion E5 analysis may read actual target draw data only after the draw has completed and only for verification, attribution, memory update, or replay evaluation.

## E5 Observation Boundary

E5 may capture candidate provenance, compare predicted candidates against completed actual results, and update local segment memory in no-write or artifact modes. It must not change existing candidate math, final ranking, or API predictions unless a future explicit enablement decision changes the production ranker.
