from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
BACKEND_PACK = BACKEND / "live_knowledge" / "full_history_engine_pack.json"


class BackendStartupMemoryTests(unittest.TestCase):
    def test_backend_app_import_does_not_load_prediction_engine_or_pack(self):
        code = """
import json
import sys
from app.main import app

print(json.dumps({
    "route_count": len(app.routes),
    "step2_loaded": "jeffrey_quad_engine_v2_step2_matrix_core" in sys.modules,
    "step3_loaded": "jeffrey_quad_engine_v2_step3_adaptive_orchestrator" in sys.modules,
    "pack_loader_loaded": "full_history_live_knowledge" in sys.modules,
    "numpy_loaded": "numpy" in sys.modules,
}))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{BACKEND}:{ROOT}"
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])

        self.assertGreater(payload["route_count"], 0)
        self.assertFalse(payload["step2_loaded"])
        self.assertFalse(payload["step3_loaded"])
        self.assertFalse(payload["pack_loader_loaded"])
        self.assertFalse(payload["numpy_loaded"])

    def test_backend_runtime_pack_is_pruned_loadable_and_complete(self):
        self.assertLess(BACKEND_PACK.stat().st_size, 5_000_000)

        code = f"""
from pathlib import Path
from full_history_live_knowledge import FullHistoryKnowledgePack, E4_PACK_ENGINE_NAME

pack = FullHistoryKnowledgePack.load(Path({str(BACKEND_PACK)!r}))
assert len(pack.models) == 26
assert pack.payload["source_artifact_count"] == 26
assert pack.payload["source_artifact_group_counts"] == {{"A": 13, "B": 9, "C": 3, "D": 1}}
assert all(name in pack.models_by_engine_name for name in (
    "E2_SET_PROJECTOR_LEARNED_BIAS__ALL",
    "E2_SET_PROJECTOR_LEARNED_BIAS__Saturday",
    "E2_SET_PROJECTOR_LEARNED_BIAS__Special",
    "E2_SET_PROJECTOR_LEARNED_BIAS__Sunday",
    "E2_SET_PROJECTOR_LEARNED_BIAS__Wednesday",
    E4_PACK_ENGINE_NAME,
))
assert pack.models_by_engine_name[E4_PACK_ENGINE_NAME]["runtime_evidence_only"] is True
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT}:{BACKEND}"
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )


if __name__ == "__main__":
    unittest.main()
