import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # PostgreSQL (users + audit logs)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "duplicados"
    postgres_user: str = "duplicados"
    postgres_password: str = "duplicados_dev"

    # MySQL (activities data - read-only)
    database_host: str = ""
    database_user: str = ""
    database_password: str = ""
    database_name: str = ""

    # JWT
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    # API Zion
    api_url_api: str = ""
    api_entity_id: int = 3
    api_token: str = ""
    api_client_calls_per_second: float = 3.0
    api_client_max_attempts: int = 3
    api_client_timeout: int = 15

    # Similarity
    similarity_min_sim_global: float = 95
    similarity_min_containment: int = 55
    similarity_diff_hard_limit: int = 12000
    similarity_pre_cutoff_delta: int = 10
    similarity_min_tokens_to_match: int = 3
    similarity_stopwords_extra: str = ""
    similarity_cutoffs_por_pasta: str = ""  # JSON string

    # AI
    openai_api_key: str = ""
    ai_provider: str = ""
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o"
    openai_model: str = "gpt-4o"

    # Seed admin (optional — creates first admin on startup if no users exist)
    admin_username: str = ""
    admin_password: str = ""

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @property
    def postgres_url(self) -> str:
        from urllib.parse import quote_plus
        pwd = quote_plus(self.postgres_password)
        return f"postgresql://{self.postgres_user}:{pwd}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def mysql_url(self) -> str:
        from urllib.parse import quote_plus
        pwd = quote_plus(self.database_password)
        return f"mysql+mysqlconnector://{self.database_user}:{pwd}@{self.database_host}/{self.database_name}"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def stopwords_extra_list(self) -> List[str]:
        if not self.similarity_stopwords_extra:
            return []
        return [s.strip() for s in self.similarity_stopwords_extra.split(",") if s.strip()]

    @property
    def cutoffs_map(self) -> dict:
        import json
        if not self.similarity_cutoffs_por_pasta:
            return {}
        try:
            return json.loads(self.similarity_cutoffs_por_pasta)
        except (json.JSONDecodeError, TypeError):
            return {}

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
