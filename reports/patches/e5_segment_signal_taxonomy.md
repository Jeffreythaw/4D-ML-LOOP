# E5 Segment Signal Taxonomy

Status: SEGMENT_TAXONOMY_APPROVED

All comparisons normalize both predicted and actual numbers to 4-digit strings, preserving leading zeros.

## Segment Classes

- `EXACT4_MATCH`: all four positions match.
- `LAST3_MATCH`: positions 2-4 match.
- `PREFIX3_MATCH`: positions 1-3 match.
- `PREFIX2_MATCH`: positions 1-2 match.
- `SUFFIX2_MATCH`: positions 3-4 match.
- `MIDDLE2_MATCH`: positions 2-3 match.
- `SAME_POSITION_3`: exactly three digit positions match.
- `SAME_POSITION_2`: exactly two digit positions match.
- `PAIR_13_MATCH`: positions 1 and 3 match.
- `PAIR_14_MATCH`: positions 1 and 4 match.
- `PAIR_24_MATCH`: positions 2 and 4 match.
- `DIGIT_BAG_3_MATCH`: multiset digit overlap count is at least 3.
- `DIGIT_BAG_2_MATCH`: multiset digit overlap count is at least 2.

## Draw 5498 Examples

Known completed-draw near-hit signals:

- `4445 -> 0445`: `LAST3_MATCH`, `SUFFIX2_MATCH`, `SAME_POSITION_3`
- `4640 -> 4693`: `PREFIX2_MATCH`, `SAME_POSITION_2`
- `9917 -> 9954`: `PREFIX2_MATCH`, `SAME_POSITION_2`
- `5335 -> 1835`: `SAME_POSITION_2`
- `9917 -> 8916`: `SAME_POSITION_2`

Exact hits remain useful but are not required for E5 to learn from segment-level evidence.
