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
        print(f"➔ Starting FastAPI Backend on port {backend_port}...", flush=True)
        # Use child environment to ensure logs are flushed
        backend_env = {**os.environ, "PORT": str(backend_port), "PYTHONUNBUFFERED": "1"}
        backend_process = subprocess.Popen([
            sys.executable,
            "start_backend.py"
        ], env=backend_env) 
        processes.append(backend_process)
        
        # Give backend more time to warm up
        print("⏳ Waiting for backend to initialize (5s)...", flush=True)
        time.sleep(5) 
        
        # Start frontend
        print(f"➔ Starting Streamlit Frontend on port {frontend_port}...", flush=True)
        frontend_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        frontend_process = subprocess.Popen([
            sys.executable,
            "start_frontend.py"
        ], env=frontend_env)
        processes.append(frontend_process)
        
        print("\n✨ All services are running!", flush=True)
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
