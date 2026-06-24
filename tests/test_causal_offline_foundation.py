from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import jeffrey_quad_engine_v2_step2_matrix_core as core
import jeffrey_quad_engine_v2_step3_adaptive_orchestrator as step3
from scripts.run_offline_causal_column_backtest import (
    CausalColumnState,
    Draw,
    generate_candidates,
    run_backtest,
)


class NoLoadGateway:
    def load_draw_history(self, **kwargs):
        raise AssertionError("history cache should prevent gateway reload")


class CausalOfflineFoundationTests(unittest.TestCase):
    def make_records(self, count: int = 12):
        records = []
        for draw_no in range(1, count + 1):
            winners = tuple(
                f"{(draw_no * 100 + offset):04d}"
                for offset in range(10)
            )
            records.append(
                core.DrawRecord(
                    draw_no=draw_no,
                    draw_date=date(2020, 1, 1),
                    day_type="Wednesday",
                    winning_numbers=winners,
                )
            )
        return records

    def test_cache_enforces_source_cutoff(self):
        cache = step3.ChronologicalDrawCache(self.make_records())
        self.assertEqual(
            [record.draw_no for record in cache.records_through(5)],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(cache.get(6).draw_no, 6)

    def test_cache_avoids_reloading_and_preserves_default_window(self):
        cache = step3.ChronologicalDrawCache(self.make_records())
        builder = step3.CandidatePoolBuilder(
            core=core,
            gateway=NoLoadGateway(),
            phase2_layer=None,
            matrix_core=None,
            formula_reader=None,
            history_cache=cache,
        )
        source, target = builder._build_wls_training_pairs(
            source_draw_no=11,
            day_type="Wednesday",
        )
        self.assertEqual(source.shape[0], 64)
        self.assertEqual(source.shape, target.shape)

    def test_zero_window_enables_explicit_full_history(self):
        cache = step3.ChronologicalDrawCache(self.make_records())
        builder = step3.CandidatePoolBuilder(
            core=core,
            gateway=NoLoadGateway(),
            phase2_layer=None,
            matrix_core=None,
            formula_reader=None,
            history_cache=cache,
            training_window_size=0,
        )
        source, _ = builder._build_wls_training_pairs(
            source_draw_no=11,
            day_type="Wednesday",
        )
        self.assertEqual(source.shape[0], 1000)

    def test_causal_markov_excludes_future_observations(self):
        records = [
            core.DrawRecord(1, date(2020, 1, 1), "Wednesday", ("0001",)),
            core.DrawRecord(
                2,
                date(2020, 1, 2),
                "Wednesday",
                ("0001", "1000", "2000"),
            ),
            core.DrawRecord(
                3,
                date(2020, 1, 3),
                "Wednesday",
                ("1000", "3000"),
            ),
        ]
        cache = core.CausalMarkovTransitionCache(records)
        through_two = cache.load_markov_transitions_bulk(
            source_states=("0001",),
            day_type="Wednesday",
            top_n_per_source=5,
            source_draw_no=2,
        )["0001"]
        through_three = cache.load_markov_transitions_bulk(
            source_states=("0001",),
            day_type="Wednesday",
            top_n_per_source=5,
            source_draw_no=3,
        )["0001"]

        self.assertEqual(
            [(row.target_state, row.transition_count) for row in through_two],
            [("0001", 1), ("1000", 1), ("2000", 1)],
        )
        self.assertEqual(through_three[0].target_state, "1000")
        self.assertEqual(through_three[0].transition_count, 2)
        self.assertNotIn("3000", [row.target_state for row in through_two])

    def test_causal_markov_ties_sort_by_target_state(self):
        records = [
            core.DrawRecord(1, date(2020, 1, 1), "Wednesday", ("0001",)),
            core.DrawRecord(
                2,
                date(2020, 1, 2),
                "Wednesday",
                ("3000", "1000", "2000"),
            ),
        ]
        rows = core.CausalMarkovTransitionCache(
            records
        ).load_markov_transitions_bulk(
            source_states=("0001",),
            day_type="Wednesday",
            top_n_per_source=5,
            source_draw_no=2,
        )["0001"]
        self.assertEqual(
            [row.target_state for row in rows],
            ["1000", "2000", "3000"],
        )

    def test_small_causal_markov_replay_is_reproducible(self):
        records = self.make_records(8)

        def replay():
            cache = core.CausalMarkovTransitionCache(records)
            return [
                [
                    (
                        row.target_state,
                        row.transition_count,
                        row.last_seen_draw_no,
                    )
                    for row in cache.load_markov_transitions_bulk(
                        source_states=records[source_draw_no - 1].winning_numbers,
                        day_type="Wednesday",
                        top_n_per_source=5,
                        source_draw_no=source_draw_no,
                    )[records[source_draw_no - 1].winning_numbers[0]]
                ]
                for source_draw_no in range(4, 8)
            ]

        self.assertEqual(replay(), replay())

    def test_laplace_smoothed_candidate_counts(self):
        state = CausalColumnState()
        source = Draw(1, "2020-01-01", "Wednesday", ("0000", "1111"))
        prediction3 = generate_candidates(
            state,
            source,
            top_k_digits=3,
            top_n=5,
            alpha=1.0,
            day_type_min_pairs=25,
        )
        prediction5 = generate_candidates(
            state,
            source,
            top_k_digits=5,
            top_n=5,
            alpha=1.0,
            day_type_min_pairs=25,
        )
        self.assertEqual(prediction3["candidate_count"], 81)
        self.assertEqual(prediction5["candidate_count"], 625)
        self.assertEqual(len(prediction3["locked"]), 5)

    def test_target_pair_enters_training_only_after_lock(self):
        draws = []
        for draw_no in range(1, 12):
            winners = tuple(
                f"{(draw_no * 317 + offset * 43) % 10000:04d}"
                for offset in range(23)
            )
            draws.append(
                Draw(
                    draw_no,
                    f"2020-01-{draw_no:02d}",
                    "Wednesday",
                    winners,
                )
            )
        events, _ = run_backtest(
            draws,
            start_draw=5,
            end_draw=8,
            top_k_digits=3,
            top_n=5,
            alpha=1.0,
            day_type_min_pairs=2,
        )
        self.assertEqual(len(events), 4)
        for event in events:
            self.assertTrue(
                event["temporal_firewall"]["target_accessed_after_lock"]
            )
            self.assertEqual(
                event["temporal_firewall"]["training_max_target_draw_no"],
                event["source_draw_no"],
            )


if __name__ == "__main__":
    unittest.main()
