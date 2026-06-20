/*
STEP 153 — DEEP CANDIDATE PERSISTENCE LAYER
DESIGN ONLY. DO NOT EXECUTE WITHOUT EXPLICIT HUMAN APPROVAL.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

IF OBJECT_ID(N'dbo.DeepCandidateLedger', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.DeepCandidateLedger
    (
        DeepCandidateId BIGINT IDENTITY(1,1) NOT NULL
            CONSTRAINT PK_DeepCandidateLedger PRIMARY KEY,
        EngineSource NVARCHAR(100) NOT NULL,
        EngineVersion NVARCHAR(100) NOT NULL,
        Mode NVARCHAR(100) NOT NULL,
        SourceDrawNo INT NOT NULL,
        TargetDrawNo INT NOT NULL,
        CandidateRank INT NOT NULL,
        CandidateNumber CHAR(4) NOT NULL,
        CandidateScore FLOAT NULL,
        CandidateFamily NVARCHAR(100) NULL,
        GenerationMethod NVARCHAR(200) NULL,
        FeatureJson NVARCHAR(MAX) NULL,
        CandidateBatchHash CHAR(64) NOT NULL,
        CandidateRowHash CHAR(64) NOT NULL,
        TemporalCutoffDrawNo INT NOT NULL,
        TargetAvailableAtGeneration BIT NOT NULL
            CONSTRAINT DF_DeepCandidateLedger_TargetAvailableAtGeneration DEFAULT (0),
        VerificationStatus NVARCHAR(40) NOT NULL
            CONSTRAINT DF_DeepCandidateLedger_VerificationStatus DEFAULT (N'Unverified'),
        HitCount INT NULL,
        CreatedAtUtc DATETIME2 NOT NULL
            CONSTRAINT DF_DeepCandidateLedger_CreatedAtUtc DEFAULT (SYSUTCDATETIME()),

        CONSTRAINT CK_DeepCandidateLedger_CandidateNumber
            CHECK (
                LEN(CandidateNumber) = 4
                AND CandidateNumber NOT LIKE '%[^0-9]%'
            ),
        CONSTRAINT CK_DeepCandidateLedger_CandidateRank
            CHECK (CandidateRank BETWEEN 1 AND 50),
        CONSTRAINT CK_DeepCandidateLedger_TemporalCutoff
            CHECK (TemporalCutoffDrawNo = SourceDrawNo),
        CONSTRAINT UQ_DeepCandidateLedger_BatchRank
            UNIQUE (
                EngineSource,
                EngineVersion,
                Mode,
                SourceDrawNo,
                TargetDrawNo,
                CandidateRank
            ),
        CONSTRAINT UQ_DeepCandidateLedger_BatchNumber
            UNIQUE (
                EngineSource,
                EngineVersion,
                Mode,
                SourceDrawNo,
                TargetDrawNo,
                CandidateNumber
            )
    );
END;

IF OBJECT_ID(N'dbo.DeepCandidateLedger', N'U') IS NOT NULL
   AND NOT EXISTS (
       SELECT 1
       FROM sys.indexes
       WHERE object_id = OBJECT_ID(N'dbo.DeepCandidateLedger')
         AND name = N'IX_DeepCandidateLedger_SourceTarget'
   )
BEGIN
    CREATE INDEX IX_DeepCandidateLedger_SourceTarget
        ON dbo.DeepCandidateLedger (SourceDrawNo, TargetDrawNo);
END;

IF OBJECT_ID(N'dbo.DeepCandidateLedger', N'U') IS NOT NULL
   AND NOT EXISTS (
       SELECT 1
       FROM sys.indexes
       WHERE object_id = OBJECT_ID(N'dbo.DeepCandidateLedger')
         AND name = N'IX_DeepCandidateLedger_EngineMode'
   )
BEGIN
    CREATE INDEX IX_DeepCandidateLedger_EngineMode
        ON dbo.DeepCandidateLedger (EngineSource, Mode);
END;

IF OBJECT_ID(N'dbo.DeepCandidateLedger', N'U') IS NOT NULL
   AND NOT EXISTS (
       SELECT 1
       FROM sys.indexes
       WHERE object_id = OBJECT_ID(N'dbo.DeepCandidateLedger')
         AND name = N'IX_DeepCandidateLedger_BatchHash'
   )
BEGIN
    CREATE INDEX IX_DeepCandidateLedger_BatchHash
        ON dbo.DeepCandidateLedger (CandidateBatchHash);
END;

IF OBJECT_ID(N'dbo.DeepCandidateLedger', N'U') IS NOT NULL
   AND NOT EXISTS (
       SELECT 1
       FROM sys.indexes
       WHERE object_id = OBJECT_ID(N'dbo.DeepCandidateLedger')
         AND name = N'IX_DeepCandidateLedger_VerificationStatus'
   )
BEGIN
    CREATE INDEX IX_DeepCandidateLedger_VerificationStatus
        ON dbo.DeepCandidateLedger (VerificationStatus);
END;

IF OBJECT_ID(N'dbo.DeepCandidateLedger', N'U') IS NOT NULL
   AND NOT EXISTS (
       SELECT 1
       FROM sys.indexes
       WHERE object_id = OBJECT_ID(N'dbo.DeepCandidateLedger')
         AND name = N'IX_DeepCandidateLedger_CandidateNumber'
   )
BEGIN
    CREATE INDEX IX_DeepCandidateLedger_CandidateNumber
        ON dbo.DeepCandidateLedger (CandidateNumber);
END;

/*
ROLLBACK — MANUAL APPROVAL REQUIRED

DROP TABLE dbo.DeepCandidateLedger;
*/
