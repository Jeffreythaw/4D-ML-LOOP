from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.temporal_context_engine import (  # noqa: E402
    OPTIONAL_UNDERLYING_ENGINES,
    REQUIRED_UNDERLYING_ENGINES,
    extract_underlying_candidates,
)
from app.schemas.prediction import PredictionCandidate  # noqa: E402


def rows(engine: str) -> list[PredictionCandidate]:
    return [
        PredictionCandidate(
            rank=rank,
            number=f"{rank:04d}",
            score=float(6 - rank),
            source=engine,
        )
        for rank in range(1, 6)
    ]


class TemporalContextFullHistoryRoutingTests(unittest.TestCase):
    def test_legacy_historical_routing_still_works_without_e40(self) -> None:
        candidates = [
            item
            for engine in REQUIRED_UNDERLYING_ENGINES
            for item in rows(engine)
        ]
        grouped = extract_underlying_candidates(candidates)
        self.assertEqual(set(grouped), set(REQUIRED_UNDERLYING_ENGINES))

    def test_e40_is_routed_when_pack_is_eligible(self) -> None:
        candidates = [
            item
            for engine in REQUIRED_UNDERLYING_ENGINES + OPTIONAL_UNDERLYING_ENGINES
            for item in rows(engine)
        ]
        grouped = extract_underlying_candidates(candidates)
        self.assertIn("E40_FULL_HISTORY_KNOWLEDGE", grouped)
        self.assertEqual(len(grouped["E40_FULL_HISTORY_KNOWLEDGE"]), 5)


if __name__ == "__main__":
    unittest.main()
