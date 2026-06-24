from functools import lru_cache
from pathlib import Path

from pydantic import Field
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(BACKEND_ENV_PATH, override=False)

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://4-d-ml-loop.vercel.app",
    "https://4-d-ml-loop-h0odv3i1q-jeffreythaws-projects.vercel.app",
)


class Settings(BaseSettings):
    sql_server_host: str = Field(default="", validation_alias="SQL_SERVER_HOST")
    sql_server_port: int | None = Field(default=None, validation_alias="SQL_SERVER_PORT")
    sql_server_database: str = Field(default="", validation_alias="SQL_SERVER_DATABASE")
    sql_server_username: str = Field(default="", validation_alias="SQL_SERVER_USERNAME")
    sql_server_password: str = Field(default="", validation_alias="SQL_SERVER_PASSWORD")
    sql_server_driver: str = Field(
        default="ODBC Driver 18 for SQL Server",
        validation_alias="SQL_SERVER_DRIVER",
    )
    sql_encrypt: str = Field(default="yes", validation_alias="SQL_ENCRYPT")
    sql_trust_server_certificate: str = Field(
        default="no",
        validation_alias="SQL_TRUST_SERVER_CERTIFICATE",
    )
    sql_verify_procedure: str = Field(
        default="dbo.SP_Verify_Predictions",
        validation_alias="SQL_VERIFY_PROCEDURE",
    )
    frontend_url: str = Field(default="", validation_alias="FRONTEND_URL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        origins = [*DEFAULT_CORS_ORIGINS]
        origins.extend(origin.strip() for origin in self.frontend_url.split(",") if origin.strip())
        return list(dict.fromkeys(origins))

    def sql_connection_string(self) -> str:
        required = {
            "SQL_SERVER_HOST": self.sql_server_host,
            "SQL_SERVER_DATABASE": self.sql_server_database,
            "SQL_SERVER_USERNAME": self.sql_server_username,
            "SQL_SERVER_PASSWORD": self.sql_server_password,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing SQL configuration: {', '.join(missing)}")

        server = self.sql_server_host
        if self.sql_server_port is not None and "," not in server:
            server = f"{server},{self.sql_server_port}"

        return (
            f"DRIVER={{{self.sql_server_driver}}};"
            f"SERVER={server};"
            f"DATABASE={self.sql_server_database};"
            f"UID={self.sql_server_username};"
            f"PWD={self.sql_server_password};"
            f"Encrypt={self.sql_encrypt};"
            f"TrustServerCertificate={self.sql_trust_server_certificate};"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
