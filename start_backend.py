"""
Backend Launcher - FastAPI Server
==================================
Starts the FastAPI backend server for AI Scoping Review
"""

import uvicorn
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Launch FastAPI backend server"""
    print("🚀 Starting AI Scoping Review - FastAPI Backend")
    print("=" * 60)
    print("📡 API will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    print("=" * 60)
    
    # Get project root
    project_root = Path(__file__).parent
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        reload_dirs=[str(project_root / "backend"), str(project_root / "shared")],  # Only watch these dirs
        reload_excludes=["*.pyc", "__pycache__/**"],  # Also exclude compiled files
        log_level="info"
    )

if __name__ == "__main__":
    main()
