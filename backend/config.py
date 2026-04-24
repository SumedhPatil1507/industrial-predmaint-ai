from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_service_key: str = ""

    openai_api_key: str = ""
    groq_api_key: str = ""
    llm_provider: str = "openai"

    slack_bot_token: str = ""
    slack_channel: str = "#maintenance-alerts"

    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_email_to: str = ""

    secret_key: str = "dev-secret-key"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
