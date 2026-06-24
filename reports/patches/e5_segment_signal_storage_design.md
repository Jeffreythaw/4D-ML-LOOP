# E5 Segment Signal Storage Design

Status: SEGMENT_MEMORY_REGISTRY_READY

Storage design: Hybrid.

## JSON Artifact

The default working-memory path is a JSON artifact. It supports local replay, deterministic tests, and supervisor inspection without SQL writes. JSON records are keyed by segment/provenance dimensions and aggregate counts.

## Optional SQL Table

Accepted historical attribution can later be persisted to SQL. The DDL below is proposed only and must not be executed as part of this scaffold.

```sql
CREATE TABLE dbo.E5SegmentSignalAttribution (
    E5SignalId BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    TargetDrawNo INT NOT NULL,
    CandidateNumber CHAR(4) NOT NULL,
    ActualNumber CHAR(4) NOT NULL,
    SegmentClass VARCHAR(64) NOT NULL,
    SourceDrawNo INT NULL,
    SourcePrizeNumber CHAR(4) NULL,
    SourcePrizeType VARCHAR(64) NULL,
    SourcePrizeRank INT NULL,
    SourcePrizeIndex INT NULL,
    EngineFamily VARCHAR(64) NULL,
    EngineName VARCHAR(128) NULL,
    FormulaId VARCHAR(128) NULL,
    MethodName VARCHAR(128) NULL,
    ModelName VARCHAR(128) NULL,
    MatrixId VARCHAR(128) NULL,
    BiasId VARCHAR(128) NULL,
    RawScore FLOAT NULL,
    RankBeforeFinal INT NULL,
    RankAfterFinal INT NULL,
    IsFinalTop5 BIT NOT NULL,
    DayType VARCHAR(16) NULL,
    CreatedAtUtc DATETIME2(3) NOT NULL DEFAULT SYSUTCDATETIME()
);

CREATE INDEX IX_E5SegmentSignalAttribution_Draw
ON dbo.E5SegmentSignalAttribution(TargetDrawNo, CandidateNumber, ActualNumber);

CREATE INDEX IX_E5SegmentSignalAttribution_SegmentEngine
ON dbo.E5SegmentSignalAttribution(SegmentClass, EngineFamily, FormulaId, MethodName, SourcePrizeType, DayType);
```

## In-Memory No-Write Mode

When no-write or dry-run mode is active, E5 updates only an in-memory registry object and does not write JSON or SQL artifacts. This preserves the existing no-write safety contract.
