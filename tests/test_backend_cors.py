from __future__ import annotations

import sys
import unittest
import asyncio
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for value in (ROOT, BACKEND):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from app.main import app  # noqa: E402


def options_preflight(path: str, *, origin: str, request_headers: str = "") -> tuple[int, dict[str, str]]:
    messages: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        messages.append(message)

    headers = [
        (b"origin", origin.encode()),
        (b"access-control-request-method", b"POST"),
    ]
    if request_headers:
        headers.append((b"access-control-request-headers", request_headers.encode()))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "method": "OPTIONS",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    asyncio.run(app(scope, receive, send))
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_headers = {
        key.decode().lower(): value.decode()
        for key, value in start.get("headers", [])
    }
    return int(start["status"]), response_headers


class BackendCorsTests(unittest.TestCase):
    def test_production_vercel_origin_can_preflight_predict(self):
        origin = "https://4-d-ml-loop.vercel.app"
        status_code, headers = options_preflight(
            "/api/predict",
            origin=origin,
            request_headers="content-type",
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(headers["access-control-allow-origin"], origin)
        self.assertIn("POST", headers["access-control-allow-methods"])
        self.assertIn("content-type", headers["access-control-allow-headers"].lower())

    def test_localhost_origin_still_allowed(self):
        origin = "http://localhost:3000"
        status_code, headers = options_preflight(
            "/api/predict",
            origin=origin,
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(headers["access-control-allow-origin"], origin)


if __name__ == "__main__":
    unittest.main()
