# API Contract

Base URL is configured in the frontend with `NEXT_PUBLIC_API_BASE_URL`.

## GET /health

Response:

```json
{
  "status": "ok",
  "service": "jeffrey-quad-engine-v2-api"
}
```

## POST /api/predict

Request:

```json
{
  "draw_number": 4051,
  "day_type": "Wednesday"
}
```

Response:

```json
{
  "draw_number": 4051,
  "day_type": "Wednesday",
  "predictions": [
    {
      "rank": 1,
      "number": "1234",
      "score": 0.95,
      "source": "existing-engine-wrapper"
    }
  ],
  "verification_status": "not_verified"
}
```

## POST /api/verify

Request:

```json
{
  "draw_number": 4051,
  "day_type": "Wednesday",
  "predictions": [
    {
      "rank": 1,
      "number": "1234",
      "score": 0.95,
      "source": "existing-engine-wrapper"
    }
  ]
}
```

Response:

```json
{
  "draw_number": 4051,
  "day_type": "Wednesday",
  "verification_status": "verified",
  "hit_count": 0,
  "details": {}
}
```

All verification responses come from the configured SQL stored procedure layer.
