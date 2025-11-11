"""
Shared Configuration Module

Central configuration management for both backend and frontend
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # ===== API Configuration =====
    grok_api_key: str = ""
    grok_model: str = "grok-3"
    grok_max_workers: int = 8
    grok_api_base_url: str = "https://api.x.ai/v1"
    
    # ===== Screening Configuration =====
    confidence_threshold: float = 0.8
    manual_review_threshold: float = 0.6
    max_batch_size: int = 100
    checkpoint_interval: int = 50
    
    # ===== Performance Settings =====
    max_requests_per_minute: int = 100
    request_timeout: int = 30  # seconds
    
    # ===== Deduplication Settings =====
    title_similarity_threshold: float = 0.85
    use_doi_matching: bool = True
    use_title_similarity: bool = True
    use_metadata_matching: bool = True
    
    # ===== File Upload Settings =====
    max_file_size_mb: int = 100
    supported_formats: list = ['.xlsx', '.xls', '.csv', '.ris', '.txt']
    
    # ===== Backend Settings =====
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    
    # ===== Frontend Settings =====
    frontend_host: str = "localhost"
    frontend_port: int = 8501
    
    # ===== Paths =====
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    temp_dir: Path = project_root / "temp_files"
    output_dir: Path = project_root / "output"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Returns:
        Settings instance
    """
    return Settings()


# ===== Constants =====

# Standard field names
STANDARD_FIELDS = {
    'required': ['title', 'abstract'],
    'optional': ['authors', 'journal', 'year', 'doi', 'keywords', 'pmid', 'url']
}

# Database sources
DATABASE_SOURCES = [
    'pubmed',
    'scopus',
    'web_of_science',
    'embase',
    'unknown'
]

# Screening decisions
SCREENING_DECISIONS = [
    'include',
    'exclude',
    'manual_review'
]

# Task statuses
TASK_STATUSES = [
    'pending',
    'running',
    'completed',
    'failed',
    'cancelled'
]

# Grok model pricing (per 1M tokens)
GROK_PRICING = {
    'grok-3': {
        'input': 1.00,   # $1 per 1M input tokens
        'output': 2.00   # $2 per 1M output tokens
    },
    'grok-3-mini-fast': {
        'input': 0.50,
        'output': 1.00
    },
    'grok-4': {
        'input': 2.00,
        'output': 4.00
    }
}

# Average token counts (for cost estimation)
AVG_TOKENS_PER_ARTICLE = {
    'title': 20,
    'abstract': 250,
    'criteria': 100,
    'response': 150
}
