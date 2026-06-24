from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from full_history_live_knowledge import (
    FULL_HISTORY_ENGINE_NAME,
    FullHistoryCandidate,
    FullHistoryKnowledgePack,
)
import jeffrey_quad_engine_v2_step3_adaptive_orchestrator as step3
import jeffrey_quad_engine_v2_step2_matrix_core as step2


ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "live_knowledge/full_history_engine_pack.json"


class FullHistoryLiveKnowledgeTests(unittest.TestCase):
    def test_pack_has_complete_engine_group_coverage(self) -> None:
        pack = FullHistoryKnowledgePack.load(PACK_PATH)
        self.assertEqual(len(pack.models), 26)
        self.assertEqual(
            pack.payload["source_artifact_group_counts"],
            {"A": 13, "B": 9, "C": 3, "D": 1},
        )
        self.assertEqual(pack.payload["dataset"]["maximum_training_pair_count"], 5428)
        self.assertIn("models", pack.payload)
        self.assertNotIn("artifacts", pack.payload)
        self.assertIn(
            "E4_MARKOV_TRANSITION_MASS__ALL",
            pack.models_by_engine_name,
        )

    def test_historical_replay_before_cutoff_is_prohibited(self) -> None:
        pack = FullHistoryKnowledgePack.load(PACK_PATH)
        self.assertFalse(pack.eligible_for(pack.minimum_source_draw_no - 1))
        self.assertTrue(pack.eligible_for(pack.minimum_source_draw_no))
        candidates = pack.rank_candidates(
            source_vectors=np.asarray([[1, 2, 3, 4]], dtype=np.int16),
            day_type="Saturday",
            source_draw_no=pack.minimum_source_draw_no - 1,
        )
        self.assertEqual(candidates, [])

    def test_pack_hash_tampering_is_rejected(self) -> None:
        payload = json.loads(PACK_PATH.read_text(encoding="utf-8"))
        payload["dataset"]["draw_cutoff"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                FullHistoryKnowledgePack.load(path)

    def test_candidate_ranking_is_deterministic_and_4d(self) -> None:
        pack = FullHistoryKnowledgePack.load(PACK_PATH)
        source = np.asarray(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]],
            dtype=np.int16,
        )
        first = pack.rank_candidates(
            source_vectors=source,
            day_type="Saturday",
            source_draw_no=pack.minimum_source_draw_no,
            latest_delta_vectors=np.asarray([[1, 1, 1, 1]], dtype=np.int16),
            top_n=20,
        )
        second = pack.rank_candidates(
            source_vectors=source,
            day_type="Saturday",
            source_draw_no=pack.minimum_source_draw_no,
            latest_delta_vectors=np.asarray([[1, 1, 1, 1]], dtype=np.int16),
            top_n=20,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertTrue(
            all(len(item.number) == 4 and item.number.isdigit() for item in first)
        )

    def test_step3_converts_pack_rank_to_live_votes(self) -> None:
        class FakePack:
            minimum_source_draw_no = 5497

            @staticmethod
            def eligible_for(source_draw_no: int) -> bool:
                return source_draw_no >= 5497

            @staticmethod
            def rank_candidates(**_: object) -> list[FullHistoryCandidate]:
                return [
                    FullHistoryCandidate("0123", 2.0, 4, ("A", "B")),
                    FullHistoryCandidate("4567", 1.0, 2, ("C",)),
                ]

        builder = step3.CandidatePoolBuilder(
            core=object(),
            gateway=object(),
            phase2_layer=object(),
            matrix_core=object(),
            formula_reader=object(),
            full_history_pack=FakePack(),
        )
        builder._build_delta_rotation_training_sets = lambda **_: (  # type: ignore[method-assign]
            np.zeros((1, 4), dtype=np.int16),
            np.zeros((1, 4), dtype=np.int16),
            np.zeros((1, 4), dtype=np.int16),
        )
        votes = builder._votes_from_full_history_knowledge(
            src_vectors=np.asarray([[1, 2, 3, 4]], dtype=np.int16),
            day_type="Saturday",
            source_draw_no=5497,
        )
        self.assertEqual([vote.engine_name for vote in votes], [FULL_HISTORY_ENGINE_NAME] * 2)
        self.assertGreater(votes[0].internal_score, votes[1].internal_score)

    def test_step3_emits_no_pack_votes_before_cutoff(self) -> None:
        pack = FullHistoryKnowledgePack.load(PACK_PATH)
        builder = step3.CandidatePoolBuilder(
            core=object(),
            gateway=object(),
            phase2_layer=object(),
            matrix_core=object(),
            formula_reader=object(),
            full_history_pack=pack,
        )
        votes = builder._votes_from_full_history_knowledge(
            src_vectors=np.asarray([[1, 2, 3, 4]], dtype=np.int16),
            day_type="Saturday",
            source_draw_no=pack.minimum_source_draw_no - 1,
        )
        self.assertEqual(votes, [])

    def test_trained_e2_uses_day_model_and_respects_cutoff(self) -> None:
        pack = FullHistoryKnowledgePack.load(PACK_PATH)
        source = np.asarray([[1, 2, 3, 4]], dtype=np.int16)
        self.assertEqual(
            pack.rank_e2_set_projector_candidates(
                source_vectors=source,
                day_type="Saturday",
                source_draw_no=pack.minimum_source_draw_no - 1,
            ),
            [],
        )
        ranked = pack.rank_e2_set_projector_candidates(
            source_vectors=source,
            day_type="Saturday",
            source_draw_no=pack.minimum_source_draw_no,
        )
        self.assertTrue(ranked)
        self.assertEqual(
            ranked[0].source_details,
            ("E2_SET_PROJECTOR_LEARNED_BIAS__Saturday",),
        )
        self.assertEqual(len(ranked[0].number), 4)
        self.assertTrue(ranked[0].number.isdigit())

    def test_candidate_pool_invokes_all_eight_engine_families(self) -> None:
        builder = step3.CandidatePoolBuilder(
            core=step2,
            gateway=object(),
            phase2_layer=object(),
            matrix_core=type(
                "MatrixCore",
                (),
                {"run_all_static_engines": lambda *_: {}},
            )(),
            formula_reader=object(),
        )

        def vote(number: str, engine: str) -> step3.CandidateVote:
            return step3.CandidateVote(number, engine, 1.0, 1, engine)

        static = [
            vote("0001", step2.ENGINE_1_NAME),
            vote("0002", step2.ENGINE_2_NAME),
            vote("0003", step2.ENGINE_3_NAME),
        ]
        with (
            patch.object(builder, "_votes_from_static_engine_outputs", return_value=static),
            patch.object(
                builder,
                "_votes_from_wls_decay",
                return_value=[vote("0004", step2.ENGINE_1_WLS_NAME)],
            ),
            patch.object(
                builder,
                "_votes_from_mirror_base5",
                return_value=[vote("0005", step2.ENGINE_1_MIRROR_BASE5_NAME)],
            ),
            patch.object(
                builder,
                "_votes_from_delta_rotation",
                return_value=[vote("0006", step2.ENGINE_1_DELTA_ROTATION_NAME)],
            ),
            patch.object(
                builder,
                "_votes_from_trained_e2",
                return_value=[vote("0007", step2.ENGINE_2_NAME)],
            ),
            patch.object(
                builder,
                "_votes_from_markov",
                return_value=[vote("0008", "E4_MARKOV_TRANSITION_MASS")],
            ),
            patch.object(
                builder,
                "_votes_from_full_history_knowledge",
                return_value=[vote("0009", FULL_HISTORY_ENGINE_NAME)],
            ),
            patch.object(builder, "_votes_from_adaptive_formulas", return_value=[]),
        ):
            pool = builder.build_candidate_pool(
                src_vectors=np.asarray([[1, 2, 3, 4]], dtype=np.int16),
                source_states=("1234",),
                day_type="Saturday",
                source_draw_no=5497,
            )

        engines = {
            engine
            for aggregate in pool.values()
            for engine in aggregate.engine_scores
        }
        self.assertTrue(
            {
                step2.ENGINE_1_NAME,
                step2.ENGINE_1_WLS_NAME,
                step2.ENGINE_1_MIRROR_BASE5_NAME,
                step2.ENGINE_1_DELTA_ROTATION_NAME,
                step2.ENGINE_2_NAME,
                step2.ENGINE_3_NAME,
                "E4_MARKOV_TRANSITION_MASS",
                FULL_HISTORY_ENGINE_NAME,
            }.issubset(engines)
        )
        locked = step3.DiversityGuardRanker().select_top5(
            pool,
            target_draw_no=5498,
            source_draw_no=5497,
        )
        self.assertEqual(len(locked.top5), 5)
        self.assertEqual(len(set(locked.top5)), 5)
        self.assertTrue(all(len(number) == 4 and number.isdigit() for number in locked.top5))
        self.assertGreaterEqual(len(set(locked.engine_sources)), 3)

    def test_live_orchestrator_loads_pack_without_prediction_or_db_access(self) -> None:
        orchestrator = step3.Step3AdaptiveOrchestrator(
            core=step2,
            gateway=object(),
            start_draw_no=5497,
            end_draw_no=5498,
        )
        self.assertIsNotNone(orchestrator.full_history_pack)
        self.assertEqual(len(orchestrator.full_history_pack.models), 26)


if __name__ == "__main__":
    unittest.main()
