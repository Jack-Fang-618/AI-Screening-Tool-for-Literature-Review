"""
Launch Both Backend and Frontend
=================================
Convenience script to start both FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
from pathlib import Path
import signal

def main():
    """Launch both backend and frontend"""
    print("🚀 Starting AI Scoping Review - Full Stack")
    print("=" * 70)
    print("This will start:")
    print("  1. FastAPI Backend on http://localhost:8000")
    print("  2. Streamlit Frontend on http://localhost:8501")
    print("=" * 70)
    print("\nPress Ctrl+C to stop both servers\n")
    
    processes = []
    
    try:
        # Start backend
        print("📡 Starting backend...")
        backend_process = subprocess.Popen([
            sys.executable,
            "start_backend.py"
        ])
        processes.append(backend_process)
        time.sleep(3)  # Wait for backend to start
        
        # Start frontend
        print("🎨 Starting frontend...")
        frontend_process = subprocess.Popen([
            sys.executable,
            "start_frontend.py"
        ])
        processes.append(frontend_process)
        
        print("\n✅ Both servers started successfully!")
        print("=" * 70)
        print("🌐 Open your browser to: http://localhost:8501")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("=" * 70)
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        for process in processes:
            process.terminate()
        
        # Wait for processes to terminate
        for process in processes:
            process.wait()
        
        print("✅ Servers stopped successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
