from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.ml_adapter import run_existing_engine_prediction  # noqa: E402
from app.schemas.prediction import PredictionCandidate, PredictionRequest  # noqa: E402


@dataclass(frozen=True)
class FakeRecord:
    day_type: str = "Wednesday"
    winning_numbers: tuple[str, ...] = tuple(f"{value:04d}" for value in range(23))


class FakeGateway:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def load_phase2_draw(self, draw_no: int) -> FakeRecord:
        assert draw_no == 5497
        return FakeRecord()


class FakeStep2:
    SqlServerGateway = FakeGateway


class FakeLocked:
    target_draw_no = 5498
    top5 = ("4445", "4640", "9917", "9373", "5335")
    engine_sources = (
        "E40_FULL_HISTORY_KNOWLEDGE",
        "E2_SET_PROJECTOR",
        "E4_MARKOV_TRANSITION_MASS",
        "E1_CROSS_PAIR_LINEAR",
        "E1_DELTA_ROTATION_LSTS",
    )
    candidate_scores = tuple((number, 1.0, source) for number, source in zip(top5, engine_sources))
    raw_vote_count = 658
    candidate_pool_count_before_ranking = 638
    engine_names_invoked = (
        "E1_CROSS_PAIR_LINEAR",
        "E1_WLS_DECAY_0.98",
        "E1_MIRROR_BASE5_LSTS",
        "E1_DELTA_ROTATION_LSTS",
        "E2_SET_PROJECTOR",
        "E3_POLYNOMIAL",
        "E4_MARKOV_TRANSITION_MASS",
        "E40_FULL_HISTORY_KNOWLEDGE",
    )
    engine_candidate_scores = tuple(
        (engine, rank, f"{rank:04d}", float(6 - rank))
        for engine in engine_names_invoked
        for rank in range(1, 6)
    )


class FakeOrchestrator:
    def __init__(self, **_kwargs):
        pass

    def predict_one_step_locked(self, source_draw_no: int) -> FakeLocked:
        assert source_draw_no == 5497
        return FakeLocked()


class FakeStep3:
    Step3AdaptiveOrchestrator = FakeOrchestrator


class FakeSettings:
    def sql_connection_string(self) -> str:
        return "fake"


def fake_temporal_context_prediction(**_kwargs):
    return type(
        "TemporalResult",
        (),
        {
            "source_draw_number": 5497,
            "target_draw_number": 5498,
            "day_type": "Saturday",
            "predictions": [
                PredictionCandidate(rank=idx, number=number, score=99.0, source="E1_TEMPORAL_CONTEXT_MATCH")
                for idx, number in enumerate(("7006", "4723", "3193", "9098", "2698"), start=1)
            ],
        },
    )()


class CurrentModeAggregateMetadataTests(unittest.TestCase):
    def test_current_mode_returns_aggregate_top5_and_keeps_overlay_metadata(self):
        with (
            patch("app.core.ml_adapter._load_existing_engine_modules", return_value=(FakeStep2, FakeStep3)),
            patch("app.core.ml_adapter.get_settings", return_value=FakeSettings()),
            patch(
                "app.core.temporal_context_engine.run_temporal_context_prediction",
                side_effect=fake_temporal_context_prediction,
            ),
        ):
            result = run_existing_engine_prediction(
                PredictionRequest(draw_number=5497, mode="Current"),
                allow_fallback=False,
            )

        self.assertEqual([item.number for item in result.predictions], ["4445", "4640", "9917", "9373", "5335"])
        self.assertNotEqual([item.number for item in result.predictions], result.metadata["overlay_top5"])
        self.assertEqual(result.metadata["overlay_top5"], ["7006", "4723", "3193", "9098", "2698"])
        self.assertEqual(result.metadata["source_prize_count"], 23)
        self.assertEqual(result.metadata["source_input_shape"], [23, 4])
        self.assertEqual(result.metadata["raw_vote_count"], 658)
        self.assertEqual(result.metadata["candidate_pool_count_before_ranking"], 638)
        self.assertEqual(result.metadata["engine_family_count"], 8)
        self.assertEqual(result.metadata["final_selection_mode"], "aggregate_23_source_prize_8_engine")
        self.assertFalse(result.metadata["target_winner_read"])
        self.assertFalse(result.metadata["sql_verifier_called"])


if __name__ == "__main__":
    unittest.main()
