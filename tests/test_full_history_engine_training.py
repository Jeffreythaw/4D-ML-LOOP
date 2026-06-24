from __future__ import annotations

import copy
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.merge_full_history_engine_training import find_duplicate_keys
from scripts.run_full_history_engine_training import (
    DRAW_HISTORY_SQL,
    DRAW_SCHEMA_SQL,
    LEDGER_READ_SQL,
    RETROSPECTIVE_LABEL,
    ChronologicalTrainingDataset,
    TrainingDraw,
    artifact_hash,
    assigned_groups,
    make_artifact,
    pairs_for_mode,
    validate_artifact_schema,
)


def synthetic_dataset(last_draw: int = 4060) -> ChronologicalTrainingDataset:
    start = date(1986, 1, 1)
    draws = []
    for draw_no in range(1, last_draw + 1):
        winners = tuple(
            f"{(draw_no * 37 + index * 101) % 10000:04d}"
            for index in range(23)
        )
        draws.append(
            TrainingDraw(
                draw_no=draw_no,
                draw_date=(start + timedelta(days=draw_no)).isoformat(),
                day_type=("Wednesday", "Saturday", "Sunday")[draw_no % 3],
                winners=winners,
            )
        )
    return ChronologicalTrainingDataset(draws)


def minimal_artifact(
    dataset: ChronologicalTrainingDataset,
    *,
    training_mode: str = "phase1_base",
    modulus: int = 10,
    formula_space: str = "BASE10",
) -> dict:
    cutoff = 4050 if training_mode == "phase1_base" else dataset.last_draw_no
    pairs = pairs_for_mode(dataset, training_mode, cutoff)
    return make_artifact(
        engine_group="A",
        engine_name="TEST_ENGINE",
        training_mode=training_mode,
        worker_id=1,
        draw_cutoff=cutoff,
        pairs=pairs,
        day_type="ALL",
        model_type="TEST",
        modulus=modulus,
        formula_space=formula_space,
        score_semantics="test",
    )


class FullHistoryEngineTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = synthetic_dataset()

    def test_dataset_cutoff_enforcement(self):
        pairs = self.dataset.pairs_until(100)
        self.assertTrue(pairs)
        self.assertLessEqual(max(pair.target_draw_no for pair in pairs), 100)
        self.assertEqual(pairs[-1].target_draw_no, 100)

    def test_no_target_leakage_in_rolling_origin_mode(self):
        cutoff = 4055
        pairs = self.dataset.phase2_pairs_until(cutoff)
        self.assertEqual(max(pair.target_draw_no for pair in pairs), cutoff)
        self.assertTrue(all(pair.target_draw_no <= cutoff for pair in pairs))

    def test_artifact_schema_and_deterministic_hashing(self):
        artifact = minimal_artifact(self.dataset)
        validate_artifact_schema(artifact)
        self.assertEqual(artifact_hash(artifact), artifact["sha256_hash"])
        clone = copy.deepcopy(artifact)
        self.assertEqual(artifact_hash(clone), artifact_hash(artifact))

    def test_base5_base10_semantic_separation(self):
        valid = minimal_artifact(
            self.dataset, modulus=5, formula_space="BASE5"
        )
        validate_artifact_schema(valid)
        invalid = copy.deepcopy(valid)
        invalid["modulus"] = 10
        invalid["sha256_hash"] = artifact_hash(invalid)
        with self.assertRaisesRegex(ValueError, "BASE5"):
            validate_artifact_schema(invalid)

    def test_worker_shard_determinism(self):
        first = [
            assigned_groups("all", worker, 4) for worker in range(1, 5)
        ]
        second = [
            assigned_groups("all", worker, 4) for worker in range(1, 5)
        ]
        self.assertEqual(first, second)
        self.assertEqual(first, [("A",), ("B",), ("C",), ("D",)])
        self.assertEqual(assigned_groups("C", 2, 4), ("C",))

    def test_merge_duplicate_detection(self):
        artifact = minimal_artifact(self.dataset)
        duplicates = find_duplicate_keys(
            [artifact, copy.deepcopy(artifact)]
        )
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(next(iter(duplicates.values())), 2)

    def test_phase1_base_uses_drawno_4050_or_less(self):
        pairs = self.dataset.phase1_pairs()
        self.assertEqual(pairs[-1].target_draw_no, 4050)
        self.assertTrue(
            all(pair.target_draw_no <= 4050 for pair in pairs)
        )

    def test_retrospective_mode_marked_not_for_live_prediction(self):
        artifact = minimal_artifact(
            self.dataset,
            training_mode="retrospective_full_history",
        )
        self.assertTrue(artifact["not_for_live_prediction"])
        self.assertEqual(
            artifact["retrospective_label"], RETROSPECTIVE_LABEL
        )
        validate_artifact_schema(artifact)

    def test_no_database_write_sql_path(self):
        combined = (
            f"{DRAW_HISTORY_SQL}\n{LEDGER_READ_SQL}\n{DRAW_SCHEMA_SQL}"
        ).upper()
        for forbidden in (
            "INSERT ",
            "UPDATE ",
            "DELETE ",
            "MERGE ",
            "CREATE ",
            "ALTER ",
            "DROP ",
            "TRUNCATE ",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertIn("SELECT", combined)


if __name__ == "__main__":
    unittest.main()
