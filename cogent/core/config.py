"""
Core Configuration Module for CogentSearch.
Utilizes Pydantic Settings for strictly typed, environment-driven configuration.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from loguru import logger
import sys

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project metadata
    PROJECT_NAME: str = "CogentSearch"
    PROJECT_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API Keys
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key for vision and embeddings")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key for Claude vision")
    LLAMA_CLOUD_API_KEY: str = Field(default="", description="Optional LlamaParse API Key")

    # Vision & Multimodal Settings
    VISION_MODEL_PROVIDER: Literal["openai", "anthropic", "mock"] = "openai"
    VISION_MODEL_NAME: str = "gpt-4o-mini"
    VISION_MAX_TOKENS: int = 1000
    VISION_TEMPERATURE: float = 0.2

    # LLM & Agentic Settings (Phase 3 & 4)
    LLM_MODEL_NAME: str = "gpt-4o"
    REWRITER_MODEL_NAME: str = "gpt-4o-mini"
    GRADER_MODEL_NAME: str = "gpt-4o-mini"
    GENERATOR_MODEL_NAME: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.0
    MAX_REWRITE_RETRIES: int = 2
    WEB_SEARCH_ENABLED: bool = True

    # Embedding & Retrieval Settings
    EMBEDDING_PROVIDER: str = "openai"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # Vector Database Settings
    VECTOR_DB_TYPE: str = "qdrant"
    QDRANT_COLLECTION_NAME: str = "cogent_search_documents"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    RRF_K: int = 60
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Redis & Cache Settings (Resume Tech Stack & Latency Optimization)
    REDIS_ENABLED: bool = True
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    CACHE_TTL_SECONDS: int = 3600
    CACHE_NAMESPACE: str = "cogent:cache"

    # Ingestion & Extraction Options
    DPI_SCALE: float = Field(default=2.5, description="Scale factor for high-resolution bounding box crops")
    MIN_IMAGE_WIDTH: int = Field(default=60, description="Minimum pixel width to treat as a meaningful visual element")
    MIN_IMAGE_HEIGHT: int = Field(default=60, description="Minimum pixel height to treat as a meaningful visual element")

    # Directories
    DATA_DIR: Path = PROJECT_ROOT / "data"
    UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"
    EXTRACTED_IMAGES_DIR: Path = PROJECT_ROOT / "data" / "extracted_images"
    PROCESSED_DOCS_DIR: Path = PROJECT_ROOT / "data" / "processed"

    def setup_directories(self) -> None:
        """Ensure all runtime directories exist."""
        for path in [self.DATA_DIR, self.UPLOAD_DIR, self.EXTRACTED_IMAGES_DIR, self.PROCESSED_DOCS_DIR]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.setup_directories()

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stderr,
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

