"""
Configuration management for SmartSupport AI
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # API Keys
    groq_api_key: str
    openai_api_key: Optional[str] = None
    
    # Database
    database_url: str = "sqlite:///./smartsupport.db"
    redis_url: str = "redis://localhost:6379/0"
    
    # Application
    app_name: str = "SmartSupport AI"
    app_version: str = "1.0.0"
    debug: bool = True
    environment: str = "development"
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # LLM Configuration
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1000

    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    
    # Gradio
    gradio_share: bool = False
    gradio_server_port: int = 7860
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# For easy import
settings = get_settings()
