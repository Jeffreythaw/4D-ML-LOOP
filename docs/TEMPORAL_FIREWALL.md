# Temporal Firewall

This skeleton preserves the existing temporal firewall boundary.

## Rules

- Do not rewrite the Step 2 or Step 3 engine logic.
- Do not bypass the SQL verifier stored procedure.
- Do not directly read hidden target winners from Python backend code.
- Do not verify predictions by querying result tables directly.
- Verification must go through the configured SQL stored procedure, defaulting to `dbo.SP_Verify_Predictions`.

## Backend Enforcement

`backend/app/core/db.py` contains the only verification integration point. It sends draw metadata and the prediction payload to the SQL stored procedure.

`backend/app/core/ml_adapter.py` is prediction-only and must not access hidden target winners. If the existing engine needs additional data access, that access must remain compatible with the existing temporal rules.
