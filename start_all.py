"""
Launch Both Backend and Frontend
=================================
Convenience script to start both FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import signal

def main():
    """Launch both backend and frontend"""
    # Set quiet mode for child processes
    os.environ["QUIET_MODE"] = "1"
    
    print("\n" + "═" * 50)
    print(" 🔬 PRISMA-ScR Toolkit - Full Stack Launcher")
    print("═" * 50)
    print(" 📡 Backend:   http://localhost:8000")
    print(" 🎨 Frontend:  http://localhost:8501")
    print(" 📚 API Docs:  http://localhost:8000/docs")
    print("═" * 50)
    print("\n🚀 Initializing services...")
    
    processes = []
    
    try:
        # Start backend
        print("➔ Starting FastAPI Backend...", end=" ", flush=True)
        backend_process = subprocess.Popen([
            sys.executable,
            "start_backend.py"
        ])
        processes.append(backend_process)
        time.sleep(2)
        print("✅ Done")
        
        # Start frontend
        print("➔ Starting Streamlit Frontend...", end=" ", flush=True)
        frontend_process = subprocess.Popen([
            sys.executable,
            "start_frontend.py"
        ])
        processes.append(frontend_process)
        print("✅ Done")
        
        print("\n✨ All services are running! Press Ctrl+C to stop.")
        print("─" * 50 + "\n")
        
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
        
        print("✅ All services stopped. See you next time!")
        sys.exit(0)

if __name__ == "__main__":
    main()
