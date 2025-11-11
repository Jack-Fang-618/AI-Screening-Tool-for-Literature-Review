"""
Frontend Launcher - Streamlit UI
=================================
Starts the Streamlit frontend interface for AI Scoping Review
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch Streamlit frontend"""
    print("🎨 Starting AI Scoping Review - Streamlit Frontend")
    print("=" * 60)
    print("🌐 UI will be available at: http://localhost:8501")
    print("⚠️  Make sure FastAPI backend is running on port 8000")
    print("=" * 60)
    
    # Get the frontend app path
    frontend_app = Path(__file__).parent / "frontend" / "app.py"
    
    # Launch Streamlit
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(frontend_app),
        "--server.port=8501",
        "--server.address=localhost"
    ])

if __name__ == "__main__":
    main()
