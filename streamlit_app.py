"""
Streamlit Cloud Deployment - Integrated Frontend + Backend

This file starts both FastAPI backend and Streamlit frontend in the same process
for deployment on Streamlit Community Cloud.

Deploy: Just upload this file to Streamlit Cloud, it will handle everything.
"""

import streamlit as st
import uvicorn
import threading
import time
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# ===== Start FastAPI Backend in Background Thread =====

def start_backend():
    """Start FastAPI backend server in a separate thread"""
    try:
        from backend.main import app
        import uvicorn
        
        # Run uvicorn in the background
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="error"  # Only show errors
        )
    except Exception as e:
        print(f"❌ Backend startup failed: {e}")

# Initialize backend in session state (only start once per session)
if 'backend_thread' not in st.session_state:
    st.session_state.backend_thread = threading.Thread(target=start_backend, daemon=True)
    st.session_state.backend_thread.start()
    st.session_state.backend_started = time.time()
    
# Wait for backend to be ready
if 'backend_ready' not in st.session_state:
    st.session_state.backend_ready = False


# ===== Import and Run Streamlit Frontend =====

from frontend.utils.api_client import APIClient

# Page Configuration
st.set_page_config(
    page_title="AI Screening Tool for Literature Review",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient(base_url="http://localhost:8000")

# Check if user has configured API key
if 'xai_api_key' not in st.session_state:
    st.session_state.xai_api_key = None

# Check if backend has API key configured
if 'backend_has_key' not in st.session_state:
    st.session_state.backend_has_key = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Check backend health
def check_backend_health():
    """Check if backend is responding"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def wait_for_backend(max_wait=10):
    """Wait for backend to start, with timeout"""
    if st.session_state.backend_ready:
        return True
    
    elapsed = time.time() - st.session_state.backend_started
    if elapsed > max_wait:
        return False
    
    if check_backend_health():
        st.session_state.backend_ready = True
        return True
    
    return False

# Main App
def main():
    """Main application entry point"""
    
    # Check backend status
    if not wait_for_backend():
        st.warning("⏳ Backend is starting up... This may take a few seconds on first load.")
        st.info(f"Elapsed: {int(time.time() - st.session_state.backend_started)}s")
        
        # Add retry button
        if st.button("🔄 Retry Connection"):
            st.rerun()
        
        # Auto-refresh every 2 seconds
        time.sleep(2)
        st.rerun()
        return
    
    # Show success message briefly
    if not st.session_state.get('welcomed', False):
        st.success("✅ System Ready!")
        st.session_state.welcomed = True
    
    # Check if user needs to configure API key
    if not st.session_state.xai_api_key:
        st.warning("⚠️ Please configure your X.AI API Key to use this application")
        
        with st.expander("🔑 Configure API Key", expanded=True):
            st.markdown("""
            ### Get Your API Key
            
            1. Visit [X.AI Console](https://console.x.ai/)
            2. Sign up or log in
            3. Navigate to **API Keys**
            4. Create a new key or copy existing one
            5. Paste it below
            
            **Privacy Notice:** Your API key is stored only in your browser session and is never shared.
            Each user pays for their own API usage.
            """)
            
            api_key_input = st.text_input(
                "Enter your X.AI API Key",
                type="password",
                placeholder="xai-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                help="Your API key starts with 'xai-'"
            )
            
            if st.button("Save API Key", type="primary"):
                if api_key_input and api_key_input.startswith("xai-"):
                    st.session_state.xai_api_key = api_key_input
                    st.success("✅ API Key saved! Reloading application...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid API key format. It should start with 'xai-'")
        
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-header">🔬 AI Screening Tool for Literature Review</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Systematic Review Assistant**")
    
    st.markdown("---")
    
    # Welcome Section
    st.markdown("## Welcome")
    
    st.markdown("""
    This toolkit streamlines systematic scoping reviews by combining intelligent data management
    with AI-powered screening. Process thousands of articles in minutes, not days.
    
    **Features:**
    - 📊 Multi-format upload (Excel, CSV, RIS)
    - 🤖 AI-powered screening with up to 50 parallel workers
    - 🔍 Smart deduplication with DOI + title similarity
    - 📈 Real-time progress tracking and cost estimation
    - 📄 PRISMA-compliant export
    """)
    
    # Quick Start Section
    st.markdown("## Quick Start Guide")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Data Management")
        st.markdown("""
        - Upload files
        - Auto-detect fields
        - Merge datasets
        - Deduplicate
        """)
        if st.button("Start Data Management", type="primary", use_container_width=True):
            st.switch_page("pages/1_Data_Management.py")
    
    with col2:
        st.markdown("### 2️⃣ AI Screening")
        st.markdown("""
        - Define criteria
        - Select AI model
        - Parallel processing
        - Track progress
        """)
        if st.button("Start AI Screening", type="primary", use_container_width=True):
            st.switch_page("pages/2_AI_Screening.py")
    
    with col3:
        st.markdown("### 3️⃣ Results")
        st.markdown("""
        - View decisions
        - Filter results
        - Generate reports
        - Export data
        """)
        if st.button("View Results", type="primary", use_container_width=True):
            st.switch_page("pages/3_Results.py")
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>Developed with ❤️ for systematic review researchers</p>
        <p><a href='https://github.com/Jack-Fang-618/AI-Screening-Tool-for-Literature-Review' target='_blank'>View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
