from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_health import router as health_router
from app.api.routes_predictions import router as predictions_router
from app.core.config import get_settings
from app.core.memory import log_memory


settings = get_settings()

app = FastAPI(
    title="Jeffrey Quad Engine v2 API",
    version="0.1.0",
    description="Integration boundary for the existing Jeffrey Quad Engine v2 ML scripts.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=settings.cors_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(predictions_router, prefix="/api")

log_memory("backend_app_imported")
