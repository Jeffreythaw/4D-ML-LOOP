/*
===============================================================================
STEP 144 — Temporal Global Loop Mode Support
===============================================================================
Purpose:
  1. Allow Temporal_Global_Loop in dbo.PredictionLedger.Mode.
  2. Add a covering index for temporal DrawHistory reads.

Temporal Firewall:
  Historical temporal winner reads must always use DrawNo <= source draw N.
  Target draw N+1 may be read for calendar metadata only.
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
        'Weighted_Grand_Loop',
        'Temporal_Global_Loop'
    )
);
GO

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'IX_DrawHistory_TemporalContext'
      AND object_id = OBJECT_ID('dbo.DrawHistory')
)
BEGIN
    CREATE INDEX IX_DrawHistory_TemporalContext
    ON dbo.DrawHistory (DrawNo, DrawDate)
    INCLUDE (WinningNumbers);
END;
GO

/*
Audit commands after the production range has completed:

EXEC dbo.SP_Summarize_EngineLedger
    @Mode = 'Temporal_Global_Loop',
    @EngineSource = 'E1_TEMPORAL_CONTEXT_MATCH',
    @SourceStart = 4050,
    @SourceEnd = 5493;

EXEC dbo.SP_Audit_PredictionLedgerIntegrity
    @Mode = 'Temporal_Global_Loop',
    @EngineSource = 'E1_TEMPORAL_CONTEXT_MATCH',
    @SourceStart = 4050,
    @SourceEnd = 5493;
*/
