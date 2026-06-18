/*
===============================================================================
STEP 139 — Engine Grand Loop Performance Support
===============================================================================
Purpose:
  1. Ensure PredictionLedger.Mode allows Engine_Grand_Loop.
  2. Add rolling performance lookup index for weighted meta rankers.
  3. Keep SQL layer as ledger/verifier/stat accelerator only.
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
CHECK (Mode IN ('Current', 'Historical', 'Grand_Loop', 'Engine_Grand_Loop'));
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'IX_PredictionLedger_EngineModeSourceVerified'
      AND object_id = OBJECT_ID('dbo.PredictionLedger')
)
BEGIN
    CREATE INDEX IX_PredictionLedger_EngineModeSourceVerified
    ON dbo.PredictionLedger
    (
        EngineSource,
        Mode,
        SourceDrawNo,
        VerificationStatus
    )
    INCLUDE
    (
        TargetDrawNo,
        HitCount,
        RankNo,
        PredictedNumber,
        Score
    );
END;
GO
