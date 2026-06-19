/*
===============================================================================
STEP 140 — Weighted Grand Loop Mode Support
===============================================================================
Purpose:
  Allow Weighted_Grand_Loop in dbo.PredictionLedger.Mode.

Temporal Firewall:
  SQL remains ledger/verifier/audit only.
===============================================================================
*/

SET NOCOUNT ON;
GO

IF EXISTS (
    SELECT 1
    FROM sys.check_constraints
    WHERE name = 'CK_PredictionLedger_Mode'
      AND parent_object_id = OBJECT_ID('dbo.PredictionLedger')
)
BEGIN
    ALTER TABLE dbo.PredictionLedger
    DROP CONSTRAINT CK_PredictionLedger_Mode;
END;
GO

ALTER TABLE dbo.PredictionLedger
ADD CONSTRAINT CK_PredictionLedger_Mode
CHECK (
    Mode IN (
        'Current',
        'Historical',
        'Grand_Loop',
        'Engine_Grand_Loop',
        'Weighted_Grand_Loop'
    )
);
GO
