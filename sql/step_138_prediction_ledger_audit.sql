/*
===============================================================================
STEP 138 — PredictionLedger Performance + Audit Layer
===============================================================================
Purpose:
  1. Add/confirm Grand_Loop mode support.
  2. Add performance index for resume/summary/integrity queries.
  3. Add SP_Summarize_EngineLedger.
  4. Add SP_Audit_PredictionLedgerIntegrity.

Temporal Firewall:
  These procedures summarize/audit already-locked PredictionLedger rows only.
  They do not generate predictions.
  They do not expose winners to Python.
===============================================================================
*/

SET NOCOUNT ON;
GO

/* ---------------------------------------------------------------------------
   1. Ensure CK_PredictionLedger_Mode allows Grand_Loop
--------------------------------------------------------------------------- */

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
CHECK (Mode IN ('Current', 'Historical', 'Grand_Loop'));
GO

/* ---------------------------------------------------------------------------
   2. Performance index for resume, summary, and audit queries
--------------------------------------------------------------------------- */

IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'IX_PredictionLedger_ModeEngineSourceSourceTargetStatus'
      AND object_id = OBJECT_ID('dbo.PredictionLedger')
)
BEGIN
    CREATE INDEX IX_PredictionLedger_ModeEngineSourceSourceTargetStatus
    ON dbo.PredictionLedger
    (
        Mode,
        EngineSource,
        SourceDrawNo,
        TargetDrawNo,
        VerificationStatus
    )
    INCLUDE
    (
        HitCount,
        RankNo,
        PredictedNumber,
        Score
    );
END;
GO

/* ---------------------------------------------------------------------------
   3. Summary stored procedure
--------------------------------------------------------------------------- */

CREATE OR ALTER PROCEDURE dbo.SP_Summarize_EngineLedger
    @Mode NVARCHAR(50),
    @EngineSource NVARCHAR(100),
    @SourceStart INT,
    @SourceEnd INT
AS
BEGIN
    SET NOCOUNT ON;

    IF @SourceStart > @SourceEnd
    BEGIN
        THROW 51000, 'SourceStart must be <= SourceEnd.', 1;
    END;

    WITH EngineGroups AS (
        SELECT
            Mode,
            EngineSource,
            SourceDrawNo,
            TargetDrawNo,
            MAX(ISNULL(HitCount, 0)) AS GroupHitCount,
            COUNT(*) AS LedgerRows
        FROM dbo.PredictionLedger
        WHERE Mode = @Mode
          AND EngineSource = @EngineSource
          AND SourceDrawNo BETWEEN @SourceStart AND @SourceEnd
          AND TargetDrawNo BETWEEN @SourceStart + 1 AND @SourceEnd + 1
          AND VerificationStatus = 'Verified'
        GROUP BY
            Mode,
            EngineSource,
            SourceDrawNo,
            TargetDrawNo
    )
    SELECT
        @Mode AS Mode,
        @EngineSource AS EngineSource,
        @SourceStart AS SourceStart,
        @SourceEnd AS SourceEnd,
        COUNT(*) AS DrawsChecked,
        SUM(CASE WHEN GroupHitCount > 0 THEN 1 ELSE 0 END) AS DrawsWithHit,
        SUM(GroupHitCount) AS RawHits,
        CAST(
            CASE
                WHEN COUNT(*) = 0 THEN 0.0
                ELSE SUM(CASE WHEN GroupHitCount > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
            END
            AS DECIMAL(12, 6)
        ) AS HitRatePercent,
        SUM(LedgerRows) AS LedgerRows,
        COUNT(*) * 5 AS ExpectedLedgerRows
    FROM EngineGroups;
END;
GO

/* ---------------------------------------------------------------------------
   4. Integrity audit stored procedure
--------------------------------------------------------------------------- */

CREATE OR ALTER PROCEDURE dbo.SP_Audit_PredictionLedgerIntegrity
    @Mode NVARCHAR(50),
    @EngineSource NVARCHAR(100),
    @SourceStart INT,
    @SourceEnd INT
AS
BEGIN
    SET NOCOUNT ON;

    IF @SourceStart > @SourceEnd
    BEGIN
        THROW 51001, 'SourceStart must be <= SourceEnd.', 1;
    END;

    WITH EngineGroups AS (
        SELECT
            Mode,
            EngineSource,
            SourceDrawNo,
            TargetDrawNo,
            COUNT(*) AS LedgerRows,
            COUNT(DISTINCT RankNo) AS DistinctRanks,
            COUNT(DISTINCT PredictedNumber) AS DistinctPredictedNumbers,
            SUM(CASE WHEN RankNo BETWEEN 1 AND 5 THEN 0 ELSE 1 END) AS BadRankRows,
            SUM(CASE WHEN VerificationStatus = 'Verified' THEN 0 ELSE 1 END) AS NonVerifiedRows,
            SUM(CASE WHEN VerificationStatus = 'Verified' AND HitCount IS NULL THEN 1 ELSE 0 END) AS VerifiedNullHitRows,
            MIN(RankNo) AS MinRankNo,
            MAX(RankNo) AS MaxRankNo
        FROM dbo.PredictionLedger
        WHERE Mode = @Mode
          AND EngineSource = @EngineSource
          AND SourceDrawNo BETWEEN @SourceStart AND @SourceEnd
          AND TargetDrawNo BETWEEN @SourceStart + 1 AND @SourceEnd + 1
        GROUP BY
            Mode,
            EngineSource,
            SourceDrawNo,
            TargetDrawNo
    ),
    MissingGroups AS (
        SELECT
            d.DrawNo AS SourceDrawNo,
            d.DrawNo + 1 AS TargetDrawNo
        FROM dbo.DrawHistory d
        WHERE d.DrawNo BETWEEN @SourceStart AND @SourceEnd
          AND EXISTS (
              SELECT 1
              FROM dbo.DrawHistory t
              WHERE t.DrawNo = d.DrawNo + 1
          )
          AND NOT EXISTS (
              SELECT 1
              FROM dbo.PredictionLedger pl
              WHERE pl.Mode = @Mode
                AND pl.EngineSource = @EngineSource
                AND pl.SourceDrawNo = d.DrawNo
                AND pl.TargetDrawNo = d.DrawNo + 1
          )
    )
    SELECT
        'GROUP_AUDIT' AS AuditType,
        Mode,
        EngineSource,
        SourceDrawNo,
        TargetDrawNo,
        LedgerRows,
        DistinctRanks,
        DistinctPredictedNumbers,
        BadRankRows,
        NonVerifiedRows,
        VerifiedNullHitRows,
        MinRankNo,
        MaxRankNo,
        CASE
            WHEN LedgerRows <> 5 THEN 'BAD_LEDGER_ROW_COUNT'
            WHEN DistinctRanks <> 5 THEN 'BAD_RANK_SET'
            WHEN MinRankNo <> 1 OR MaxRankNo <> 5 THEN 'BAD_RANK_RANGE'
            WHEN DistinctPredictedNumbers <> 5 THEN 'DUPLICATE_PREDICTED_NUMBER'
            WHEN BadRankRows <> 0 THEN 'BAD_RANK_VALUE'
            WHEN NonVerifiedRows <> 0 THEN 'NON_VERIFIED_ROWS'
            WHEN VerifiedNullHitRows <> 0 THEN 'VERIFIED_NULL_HITCOUNT'
            ELSE 'OK'
        END AS AuditStatus
    FROM EngineGroups
    WHERE LedgerRows <> 5
       OR DistinctRanks <> 5
       OR MinRankNo <> 1
       OR MaxRankNo <> 5
       OR DistinctPredictedNumbers <> 5
       OR BadRankRows <> 0
       OR NonVerifiedRows <> 0
       OR VerifiedNullHitRows <> 0

    UNION ALL

    SELECT
        'MISSING_GROUP' AS AuditType,
        @Mode AS Mode,
        @EngineSource AS EngineSource,
        SourceDrawNo,
        TargetDrawNo,
        0 AS LedgerRows,
        0 AS DistinctRanks,
        0 AS DistinctPredictedNumbers,
        0 AS BadRankRows,
        0 AS NonVerifiedRows,
        0 AS VerifiedNullHitRows,
        NULL AS MinRankNo,
        NULL AS MaxRankNo,
        'MISSING_LEDGER_GROUP' AS AuditStatus
    FROM MissingGroups

    ORDER BY SourceDrawNo, TargetDrawNo, AuditType;
END;
GO
