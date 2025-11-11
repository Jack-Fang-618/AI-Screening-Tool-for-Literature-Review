"""
Database package initialization

Exports database models, session management, and utilities
"""

from .config import (
    engine,
    SessionLocal,
    get_db,
    get_db_session,
    init_db,
    drop_db,
    reset_db,
    check_db_health,
    get_db_info
)

from backend.models.database import (
    Base,
    Dataset,
    Task,
    ScreeningResult,
    DeduplicationRecord,
    SystemLog,
    TaskStatus,
    ScreeningDecision,
    DatasetStatus
)

__all__ = [
    # Session management
    'engine',
    'SessionLocal',
    'get_db',
    'get_db_session',
    
    # Database operations
    'init_db',
    'drop_db',
    'reset_db',
    'check_db_health',
    'get_db_info',
    
    # Models
    'Base',
    'Dataset',
    'Task',
    'ScreeningResult',
    'DeduplicationRecord',
    'SystemLog',
    
    # Enums
    'TaskStatus',
    'ScreeningDecision',
    'DatasetStatus'
]
