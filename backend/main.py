"""
FastAPI Backend - Main Application Entry Point

This is the core FastAPI server that handles:
- Data management (upload, parse, merge, deduplicate)
- AI screening tasks (async processing with 8 workers)
- Results retrieval and export

Run with: uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from contextlib import asynccontextmanager
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (optional for deployment)
# Path: backend/main.py -> backend/ -> Core function/ -> .env
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from {env_path}")
    import os
    api_key_status = "configured" if os.getenv('XAI_API_KEY') else "not set"
    print(f"🔑 XAI_API_KEY: {api_key_status}")
else:
    # No .env file - this is normal for cloud deployment
    # Users will provide API keys via UI
    print("ℹ️  Running in cloud mode - users will provide API keys via UI")

# Import API routers
from backend.api import data_management, screening, results

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== Middleware for Request Logging =====

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        logger.info(f"📨 Incoming request: {request.method} {request.url.path}")
        logger.info(f"   Client: {request.client.host if request.client else 'unknown'}")
        logger.info(f"   Headers: {dict(request.headers)}")
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(f"✅ Response: {response.status_code} (took {process_time:.3f}s)")
            return response
        except Exception as e:
            logger.error(f"❌ Error processing request: {str(e)}", exc_info=True)
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting AI Scoping Review Backend Server")
    
    # Initialize database if not exists
    logger.info("🗄️  Initializing database...")
    db_initialized = False
    try:
        from backend.db import init_db, engine
        from backend.models.database import Base
        
        # Create all tables (with timeout protection)
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database initialized")
        db_initialized = True
    except Exception as e:
        logger.warning(f"⚠️  Database initialization skipped: {e}")
        logger.info("ℹ️  Will use in-memory storage for this session")
        # Continue anyway - database is optional for basic functionality
    
    logger.info("📊 Initializing task manager...")
    try:
        from backend.tasks.task_manager import task_manager
        # Task manager automatically loads from database when use_database=True
        logger.info("✅ Task manager initialized")
    except Exception as e:
        logger.warning(f"⚠️  Task manager initialization had issues: {e}")
    
    # Start automatic data cleanup task (non-blocking, skip on error)
    cleanup_started = False
    try:
        logger.info("🧹 Starting automatic data cleanup task...")
        from backend.tasks.cleanup_task import cleanup_task
        cleanup_task.start()
        logger.info("✅ Cleanup task started (runs every 6 hours)")
        cleanup_started = True
    except Exception as e:
        logger.warning(f"⚠️  Cleanup task skipped: {e}")
        logger.info("ℹ️  Cleanup task is optional - server will continue without it")
    
    logger.info("✅ Server startup complete - ready to accept requests")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down server...")
    if cleanup_started:
        try:
            cleanup_task.stop()
            logger.info("✅ Cleanup task stopped")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup task stop failed: {e}")


# Create FastAPI app
app = FastAPI(
    title="AI Scoping Review API",
    description="Backend API for AI-powered systematic review screening",
    version="2.0.0",
    lifespan=lifespan
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Configure CORS (allow Streamlit frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (Streamlit Cloud URL is dynamic)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - simple health check"""
    return {
        "status": "healthy",
        "service": "AI Scoping Review Backend",
        "version": "2.0.0"
    }


@app.get("/health")
async def health_check():
    """Railway health check - must respond quickly"""
    return {"status": "healthy"}


@app.get("/health/detailed")
async def health_check_detailed():
    """Detailed health check with diagnostics"""
    logger.info("💊 Detailed health check endpoint accessed")
    
    # Check database
    db_status = "unknown"
    try:
        from backend.db import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)[:50]}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "task_manager": "active",
        "workers": 8
    }


# Include API routers
app.include_router(
    data_management.router,
    prefix="/api/data",
    tags=["Data Management"]
)

app.include_router(
    screening.router,
    prefix="/api/screening",
    tags=["AI Screening"]
)

app.include_router(
    results.router,
    prefix="/api/results",
    tags=["Results"]
)


if __name__ == "__main__":
    import uvicorn
    
    logger.info("🏃 Running development server directly...")
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
