"""
EdukaAI Studio - Configuration Management

Principle: Zero hardcoded values. Everything configurable via database or environment.
Priority: Environment Variables > Database > Default Values
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable override."""
    
    # Application
    app_name: str = "EdukaAI Studio"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite:///./storage/app/edukaai.db"
    
    # Storage
    storage_path: str = "./storage"
    max_storage_gb: int = 50
    
    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    
    # MLX
    mlx_device: str = "gpu"  # or "cpu"
    mlx_gpu_memory_limit_mb: Optional[int] = None
    
    # Security
    secret_key: str = "change-me-in-production"
    allow_remote: bool = False  # Only accept localhost connections by default
    
    # HuggingFace
    hf_token: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./storage/app/logs/master.log"
    
    # Training Limits
    max_dataset_samples: int = 10000
    max_context_length: int = 4096
    max_concurrent_runs: int = 1
    
    class Config:
        env_prefix = "EDUKAAI_"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()


class Config:
    """
    Dynamic configuration manager.
    
    Provides access to:
    1. Environment variables (via Settings)
    2. Database-stored configuration
    3. Default values
    """
    
    def __init__(self):
        self._settings = get_settings()
        self._db_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[float] = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with priority: Env > DB > Default."""
        # 1. Check environment variables (already handled by Settings)
        env_value = os.getenv(f"EDUKAAI_{key.upper()}")
        if env_value is not None:
            return self._cast_value(env_value)
        
        # 2. Check database cache
        if key in self._db_cache:
            return self._db_cache[key]
        
        # 3. Return default
        return default
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer config value."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float config value."""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean config value."""
        value = self.get(key, default)
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    def _cast_value(self, value: str) -> Any:
        """Auto-cast string values to appropriate types."""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    def refresh_db_cache(self, db_settings: Dict[str, Any]):
        """Refresh database configuration cache."""
        self._db_cache = db_settings


@lru_cache()
def get_config() -> Config:
    """Get configuration singleton."""
    return Config()


# Ensure storage directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        "./storage/app/logs",
        "./storage/app/temp",
        "./storage/app/cache",
        "./storage/datasets",
        "./storage/runs",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


# Initialize on module load
ensure_directories()
