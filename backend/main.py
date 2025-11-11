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

# Load environment variables from .env file
# Path: backend/main.py -> backend/ -> Core function/ -> .env
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment variables from {env_path}")
    import os
    print(f"🔑 XAI_API_KEY loaded: {bool(os.getenv('XAI_API_KEY'))}")
else:
    print(f"⚠️  No .env file found at {env_path}")

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
    logger.info("📊 Initializing task manager...")
    # TODO: Initialize task manager, load configs, etc.
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down server...")
    # TODO: Cleanup tasks, save state, etc.


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
    allow_origins=[
        "http://localhost:8501",  # Streamlit default port
        "http://127.0.0.1:8501",
        "http://localhost:8000",  # Allow self for docs
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check"""
    logger.info("🏠 Root endpoint accessed")
    return {
        "status": "healthy",
        "service": "AI Scoping Review Backend",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    logger.info("💊 Health check endpoint accessed")
    return {
        "status": "healthy",
        "database": "not_implemented",  # Future: check DB connection
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
