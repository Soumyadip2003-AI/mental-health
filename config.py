"""
Configuration settings for the Mental Health Crisis Detection System
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "Mental Health Crisis Detector"
    app_version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Database
    database_url: str = Field(default="sqlite:///./crisis_detector.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Model Settings
    model_cache_dir: str = Field(default="models", env="MODEL_CACHE_DIR")
    data_dir: str = Field(default="data", env="DATA_DIR")
    max_text_length: int = Field(default=512, env="MAX_TEXT_LENGTH")
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Monitoring
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/crisis_detector.log", env="LOG_FILE")
    
    # Model Configuration
    use_bert: bool = Field(default=True, env="USE_BERT")
    use_spacy: bool = Field(default=True, env="USE_SPACY")
    use_multimodal: bool = Field(default=True, env="USE_MULTIMODAL")
    
    # Performance
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ModelConfig:
    """Model-specific configuration"""
    
    # Text models
    TEXT_MODELS = {
        "bert": {
            "model_name": "bert-base-uncased",
            "max_length": 512,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_epochs": 3
        },
        "roberta": {
            "model_name": "roberta-base",
            "max_length": 512,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_epochs": 3
        },
        "distilbert": {
            "model_name": "distilbert-base-uncased",
            "max_length": 512,
            "batch_size": 32,
            "learning_rate": 3e-5,
            "num_epochs": 3
        }
    }
    
    # Ensemble models
    ENSEMBLE_MODELS = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "lightgbm": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
    }
    
    # Feature extraction
    FEATURE_EXTRACTION = {
        "tfidf": {
            "max_features": 5000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95
        },
        "word2vec": {
            "vector_size": 100,
            "window": 5,
            "min_count": 2,
            "workers": 4
        },
        "fasttext": {
            "vector_size": 100,
            "window": 5,
            "min_count": 2,
            "workers": 4
        }
    }


class CrisisDetectionConfig:
    """Crisis detection specific configuration"""
    
    # Crisis keywords and patterns
    CRISIS_KEYWORDS = {
        "suicide": ["suicide", "kill myself", "end my life", "take my life", "die", "death"],
        "hopelessness": ["hopeless", "worthless", "useless", "burden", "can't go on", "no reason", "pointless"],
        "planning": ["plan", "method", "pills", "rope", "gun", "jump", "bridge", "building", "cut", "wrist"],
        "goodbye": ["goodbye", "last message", "final", "farewell", "see you never", "this is it"]
    }
    
    # Risk levels
    RISK_LEVELS = {
        "low": {"threshold": 0.3, "color": "green", "action": "monitor"},
        "medium": {"threshold": 0.6, "color": "yellow", "action": "intervene"},
        "high": {"threshold": 0.8, "color": "red", "action": "urgent_intervention"},
        "critical": {"threshold": 0.9, "color": "darkred", "action": "immediate_crisis_response"}
    }
    
    # Emergency contacts
    EMERGENCY_CONTACTS = {
        "national_suicide_prevention": "988",
        "crisis_text_line": "741741",
        "emergency": "911"
    }


# Global settings instance
settings = Settings()

# Create directories if they don't exist
Path(settings.model_cache_dir).mkdir(parents=True, exist_ok=True)
Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)
Path("cache").mkdir(parents=True, exist_ok=True)
