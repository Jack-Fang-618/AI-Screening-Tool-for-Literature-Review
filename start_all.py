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
    
    # Port configuration
    backend_port = 8000
    frontend_port = os.environ.get("PORT", "8501")
    
    # Ensure Backend URL is set for the frontend if not already set
    if "BACKEND_URL" not in os.environ:
        os.environ["BACKEND_URL"] = f"http://localhost:{backend_port}"
        print(f" 🔗 Backend URL: {os.environ['BACKEND_URL']}")
    
    print(f" 📡 Backend Internal: http://localhost:{backend_port}")
    print(f" 🎨 Frontend Public:   http://localhost:{frontend_port}")
    print(f" 📚 API Docs:          http://localhost:{backend_port}/docs")
    print("═" * 50)
    print("\n🚀 Initializing services...")
    
    processes = []
    
    try:
        # Start backend
        print(f"➔ Starting FastAPI Backend on port {backend_port}...", end=" ", flush=True)
        # We use host 127.0.0.1 for internal communication if possible, or 0.0.0.0
        backend_process = subprocess.Popen([
            sys.executable,
            "start_backend.py"
        ], env={**os.environ, "PORT": str(backend_port)}) # Force backend to 8000
        processes.append(backend_process)
        time.sleep(3) # Give it more time to start
        print("✅ Done")
        
        # Start frontend
        print(f"➔ Starting Streamlit Frontend on port {frontend_port}...", end=" ", flush=True)
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
