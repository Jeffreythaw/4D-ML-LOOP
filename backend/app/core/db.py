import json
from typing import Any

from app.core.config import get_settings
from app.schemas.prediction import LatestDrawResponse, VerificationRequest, VerificationResponse


class VerificationError(RuntimeError):
    pass


def verify_predictions_with_sql(request: VerificationRequest) -> VerificationResponse:
    """
    Verify predictions only through the approved SQL Server verification layer.

    This function intentionally does not query hidden target winners. The stored
    procedure owns winner access, hit calculation, and temporal firewall rules.
    """
    if not request.predictions:
        raise ValueError("At least one prediction is required for verification.")

    settings = get_settings()

    try:
        import pyodbc  # type: ignore
    except ImportError as exc:
        raise VerificationError("pyodbc is not installed in this backend environment.") from exc

    top5_predictions = ",".join(prediction.number for prediction in request.predictions[:5])

    try:
        with pyodbc.connect(settings.sql_connection_string(), timeout=15) as connection:
            cursor = connection.cursor()
            cursor.execute(
                f"EXEC {settings.sql_verify_procedure} @TargetDrawNo=?, @Top5Predictions=?",
                request.draw_number,
                top5_predictions,
            )
            row = cursor.fetchone()
    except ValueError:
        raise
    except Exception as exc:
        raise VerificationError("SQL verification procedure failed.") from exc

    if row is None:
        raise VerificationError("SQL verification procedure returned no result.")

    result = _row_to_dict(row)

    return VerificationResponse(
        draw_number=request.draw_number,
        day_type=request.day_type,
        verification_status=str(result.get("verification_status", "verified")),
        hit_count=_safe_int(result.get("hit_count"), default=0),
        details=result,
    )


def get_latest_draw_metadata() -> LatestDrawResponse:
    """
    Return latest completed draw metadata only.

    This intentionally reads only DrawNo and DrawDate from dbo.DrawHistory.
    It does not read hidden winner columns and does not perform verification.
    """
    settings = get_settings()

    try:
        import pyodbc  # type: ignore
    except ImportError as exc:
        raise VerificationError("pyodbc is not installed in this backend environment.") from exc

    try:
        with pyodbc.connect(settings.sql_connection_string(), timeout=15) as connection:
            cursor = connection.cursor()
            row = cursor.execute(
                """
                SELECT TOP (1)
                    DrawNo,
                    DrawDate,
                    CASE
                        WHEN DATENAME(WEEKDAY, DrawDate) IN ('Wednesday', 'Saturday', 'Sunday')
                            THEN DATENAME(WEEKDAY, DrawDate)
                        ELSE 'Special'
                    END AS DayType
                FROM dbo.DrawHistory
                ORDER BY DrawNo DESC;
                """
            ).fetchone()
    except Exception as exc:
        raise VerificationError("SQL latest draw metadata query failed.") from exc

    if row is None:
        raise VerificationError("No draw metadata found in dbo.DrawHistory.")

    draw_no = int(row.DrawNo)
    draw_date = row.DrawDate.isoformat() if hasattr(row.DrawDate, "isoformat") else str(row.DrawDate)

    return LatestDrawResponse(
        draw_number=draw_no,
        target_draw_number=draw_no + 1,
        draw_date=draw_date,
        day_type=str(row.DayType),
    )


def _row_to_dict(row: Any) -> dict[str, Any]:
    columns = [column[0] for column in row.cursor_description]
    return dict(zip(columns, row))


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
