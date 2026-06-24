from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jeffrey_quad_engine_v2_step2_matrix_core as core


class _FakeCursor:
    def execute(self, sql, *args):
        return self

    def fetchone(self):
        return None

    def fetchall(self):
        return []

    def close(self):
        pass


class _FakeConn:
    def cursor(self):
        return _FakeCursor()


class NoSqlWriteFormulaRegistryTests(unittest.TestCase):
    def test_adaptive_formula_registration_stays_in_memory_when_no_write(self):
        gateway = core.SqlServerGateway("fake", no_sql_write=True)
        matrix_core = core.MatrixComputationCore(gateway)
        src = core.vectors_from_4d_strings(["0000", "1111", "2222", "3333"])
        tgt = core.vectors_from_4d_strings(["0001", "1112", "2223", "3334"])

        formula_id = matrix_core.solve_and_register_adaptive_formula(
            source_draw_no=10,
            target_draw_no=11,
            day_type="Wednesday",
            src_vectors=src,
            tgt_vectors=tgt,
            training_start_draw_no=10,
            training_end_draw_no=11,
        )

        formulas = gateway.load_in_memory_formulas(
            day_type="Wednesday",
            engine_name=core.ENGINE_4_NAME,
            max_training_end_draw_no=11,
        )
        self.assertEqual(formula_id, -1)
        self.assertEqual(len(formulas), 1)
        self.assertEqual(formulas[0].formula_version, "E4_FIX_10_TO_11")
        self.assertEqual(gateway.sql_write_statements_attempted, 0)
        self.assertEqual(gateway.sql_write_statements_executed, 0)
        self.assertEqual(gateway.sql_write_statements_blocked, 0)

    def test_no_write_cursor_guard_blocks_accidental_write_sql(self):
        gateway = core.SqlServerGateway("fake", no_sql_write=True)
        gateway._conn = core._SqlWriteGuardConnection(_FakeConn(), gateway)

        with self.assertRaises(RuntimeError):
            gateway.conn.cursor().execute("INSERT INTO dbo.FormulaRegistry VALUES (1);")

        self.assertEqual(gateway.sql_write_statements_attempted, 1)
        self.assertEqual(gateway.sql_write_statements_executed, 0)
        self.assertEqual(gateway.sql_write_statements_blocked, 1)

    def test_env_var_forces_no_write_mode(self):
        with patch.dict(os.environ, {"J4D_NO_SQL_WRITE": "1"}):
            gateway = core.SqlServerGateway("fake", no_sql_write=False)

        self.assertTrue(gateway.no_sql_write)


if __name__ == "__main__":
    unittest.main()
