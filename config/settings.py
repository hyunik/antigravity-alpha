"""
Antigravity-Alpha Configuration Settings
Environment variables and global settings management
"""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class LLMConfig(BaseModel):
    """LLM Provider Configuration"""
    provider: Literal["openai", "gemini"] = Field(default="openai")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o")
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.0-flash")


class ExchangeConfig(BaseModel):
    """Exchange API Configuration"""
    binance_api_key: str = Field(default="")
    binance_api_secret: str = Field(default="")
    bybit_api_key: str = Field(default="")
    bybit_api_secret: str = Field(default="")


class DiscordConfig(BaseModel):
    """Discord Configuration"""
    webhook_url: str = Field(default="")


class SystemConfig(BaseModel):
    """System Settings"""
    database_url: str = Field(default="sqlite:///./data/antigravity.db")
    top_coins_count: int = Field(default=200)
    min_report_score: int = Field(default=60)
    report_top_n: int = Field(default=50)
    log_level: str = Field(default="INFO")


class Settings:
    """Global Settings Manager"""
    
    def __init__(self):
        self.llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        )
        
        self.exchange = ExchangeConfig(
            binance_api_key=os.getenv("BINANCE_API_KEY", ""),
            binance_api_secret=os.getenv("BINANCE_API_SECRET", ""),
            bybit_api_key=os.getenv("BYBIT_API_KEY", ""),
            bybit_api_secret=os.getenv("BYBIT_API_SECRET", ""),
        )
        
        self.discord = DiscordConfig(
            webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
        )
        
        self.system = SystemConfig(
            database_url=os.getenv("DATABASE_URL", "sqlite:///./data/antigravity.db"),
            top_coins_count=int(os.getenv("TOP_COINS_COUNT", "200")),
            min_report_score=int(os.getenv("MIN_REPORT_SCORE", "60")),
            report_top_n=int(os.getenv("REPORT_TOP_N", "50")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        
        self.base_dir = BASE_DIR
    
    def get_llm_client(self):
        """Get the configured LLM client"""
        if self.llm.provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=self.llm.openai_api_key)
        elif self.llm.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.llm.gemini_api_key)
            return genai.GenerativeModel(self.llm.gemini_model)
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm.provider}")
    
    def validate(self) -> list[str]:
        """Validate required settings and return list of errors"""
        errors = []
        
        if self.llm.provider == "openai" and not self.llm.openai_api_key:
            errors.append("OPENAI_API_KEY is required when using OpenAI provider")
        if self.llm.provider == "gemini" and not self.llm.gemini_api_key:
            errors.append("GEMINI_API_KEY is required when using Gemini provider")
        if not self.discord.webhook_url:
            errors.append("DISCORD_WEBHOOK_URL is required for report distribution")
            
        return errors


# Global settings instance
settings = Settings()
