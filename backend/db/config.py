"""
Database Configuration and Session Management

Provides SQLAlchemy engine, session factory, and database utilities.
Supports both SQLite (development) and PostgreSQL (production).
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import logging
from pathlib import Path
import os
from typing import Generator

from ..models.database import Base

logger = logging.getLogger(__name__)


# ===== Database Configuration =====

def get_database_url() -> str:
    """
    Get database URL from environment or use default SQLite
    
    Supports:
    - SQLite: sqlite:///./data/app.db
    - PostgreSQL: postgresql://user:pass@host:port/dbname
    - MySQL: mysql://user:pass@host:port/dbname
    
    Returns:
        Database URL string
    """
    # Try to get from environment
    db_url = os.getenv('DATABASE_URL')
    
    if db_url:
        logger.info(f"📊 Using database from DATABASE_URL environment variable")
        return db_url
    
    # Default to SQLite in data directory
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    db_path = data_dir / "app.db"
    db_url = f"sqlite:///{db_path}"
    
    logger.info(f"📊 Using SQLite database at {db_path}")
    return db_url


# ===== Engine Configuration =====

DATABASE_URL = get_database_url()

# Engine configuration based on database type
if DATABASE_URL.startswith('sqlite'):
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # Allow multi-threading
        poolclass=StaticPool,  # Single connection pool for SQLite
        echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'  # SQL logging
    )
    
    # Enable foreign keys for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
        
else:
    # PostgreSQL/MySQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,  # Connection pool size
        max_overflow=20,  # Max overflow connections
        pool_pre_ping=True,  # Verify connections before using
        echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'
    )


# ===== Session Factory =====

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# ===== Database Utilities =====

def init_db():
    """
    Initialize database - create all tables
    
    Should be called on application startup
    """
    logger.info("🔨 Initializing database...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database initialized successfully")


def drop_db():
    """
    Drop all tables - USE WITH CAUTION
    
    Only for development/testing
    """
    logger.warning("⚠️  Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("✅ All tables dropped")


def reset_db():
    """
    Reset database - drop and recreate all tables
    
    Only for development/testing
    """
    logger.warning("🔄 Resetting database...")
    drop_db()
    init_db()
    logger.info("✅ Database reset complete")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get database session (context manager)
    
    Usage:
        with get_db() as db:
            dataset = db.query(Dataset).first()
    
    Yields:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Database error: {e}")
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get database session (for FastAPI dependency injection)
    
    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db_session)):
            return db.query(Item).all()
    
    Returns:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===== Health Check =====

def check_db_health() -> bool:
    """
    Check if database is accessible
    
    Returns:
        True if database is healthy, False otherwise
    """
    try:
        with get_db() as db:
            # Try a simple query
            db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")
        return False


# ===== Migration Helpers =====

def get_db_info() -> dict:
    """
    Get database information
    
    Returns:
        Dict with database type, version, tables, etc.
    """
    with get_db() as db:
        # Get database type
        db_type = engine.dialect.name
        
        # Get table count
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        info = {
            'type': db_type,
            'url': DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL,  # Hide credentials
            'tables': tables,
            'table_count': len(tables)
        }
        
        return info


# ===== Logging Setup =====

def log_db_query(query: str, params: dict = None):
    """Log database query (for debugging)"""
    if os.getenv('SQL_DEBUG', 'false').lower() == 'true':
        logger.debug(f"📝 SQL Query: {query}")
        if params:
            logger.debug(f"📝 Params: {params}")
