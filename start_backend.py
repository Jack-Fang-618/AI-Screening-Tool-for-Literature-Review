"""
Backend Launcher - FastAPI Server
==================================
Starts the FastAPI backend server for AI Scoping Review
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Launch FastAPI backend server"""
    if os.environ.get("QUIET_MODE") != "1":
        print("🚀 Starting AI Scoping Review - FastAPI Backend")
        print("=" * 60)
        print("📡 API will be available at: http://localhost:8000")
        print("📚 API docs will be available at: http://localhost:8000/docs")
        print("🔍 Alternative docs at: http://localhost:8000/redoc")
        print("=" * 60)
    
    # Get project root
    project_root = Path(__file__).parent
    
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=False if os.environ.get("QUIET_MODE") == "1" else True,  # Disable reload in quiet/prod mode
        reload_dirs=[str(project_root / "backend"), str(project_root / "shared")] if os.environ.get("QUIET_MODE") != "1" else [],
        reload_excludes=["*.pyc", "__pycache__/**"],
        log_level="info" if os.environ.get("QUIET_MODE") != "1" else "error"
    )

if __name__ == "__main__":
    main()
