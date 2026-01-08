"""
Frontend Launcher - Streamlit UI
=================================
Starts the Streamlit frontend interface for AI Scoping Review
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch Streamlit frontend"""
    if os.environ.get("QUIET_MODE") != "1":
        print("🎨 Starting AI Scoping Review - Streamlit Frontend")
        print("=" * 60)
        print("🌐 UI will be available at: http://localhost:8501")
        print("⚠️  Make sure FastAPI backend is running on port 8000")
        print("=" * 60)
    
    # Get the app path (Running the root streamlit_app.py which is compatible with root pages/ folder)
    frontend_app = Path(__file__).parent / "streamlit_app.py"
    
    # Launch Streamlit
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(frontend_app),
        "--server.port=8501",
        "--server.address=localhost",
        "--logger.level=error"
    ])

if __name__ == "__main__":
    main()
