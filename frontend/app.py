"""
Streamlit Frontend - Main Application (Home Page)

This is the main Streamlit UI that connects to the FastAPI backend
for AI-powered systematic review screening.

Run with: streamlit run frontend/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frontend.utils.api_client import APIClient


# ===== Page Configuration =====

st.set_page_config(
    page_title="PRISMA-ScR Toolkit - Home",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ===== Initialize Session State =====

if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient(base_url="http://localhost:8000")


# ===== Custom CSS =====

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


# ===== Page Functions =====

def main():
    """Main application entry point - Home Page"""
    
    # Header
    st.markdown('<h1 class="main-header">PRISMA-ScR Toolkit</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Systematic Review Assistant**")
    
    st.markdown("---")
    
    # Welcome Section
    st.markdown("## Welcome")
    
    st.markdown("""
    This toolkit streamlines systematic scoping reviews by combining intelligent data management
    with AI-powered screening. Process thousands of articles in minutes, not days.
    """)
    
    # Quick Start Section
    st.markdown("## Quick Start Guide")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 1️⃣ Data Management")
        st.markdown("""
        - Upload files (Excel, CSV, RIS)
        - Auto-detect database sources
        - Merge and deduplicate
        - Clean and standardize
        """)
        if st.button("Start Data Management", type="primary", use_container_width=True):
            st.switch_page("pages/1_Data_Management.py")
    
    with col2:
        st.markdown("### 2️⃣ AI Screening")
        st.markdown("""
        - Define inclusion/exclusion criteria
        - Select AI model
        - 8 parallel workers
        - Real-time progress
        """)
        if st.button("Start AI Screening", type="primary", use_container_width=True):
            st.switch_page("pages/2_AI_Screening.py")
    
    with col3:
        st.markdown("### 3️⃣ Results")
        st.markdown("""
        - View decisions
        - Generate PRISMA diagrams
        - Export results
        - Analyze statistics
        """)
        if st.button("View Results", type="primary", use_container_width=True):
            st.switch_page("pages/3_Results.py")
    
    with col4:
        st.markdown("### ⚙️ Settings")
        st.markdown("""
        - Configure API keys
        - Set default criteria
        - Manage preferences
        - Save templates
        """)
        if st.button("Open Settings", type="secondary", use_container_width=True):
            st.switch_page("pages/4_Settings.py")
    
    st.markdown("---")
    
    # Features Section
    st.markdown("## Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Processing")
        st.markdown("""
        - **Multi-source support**: PubMed, Scopus, Web of Science, Embase
        - **Intelligent field mapping**: Auto-detect and standardize columns
        - **Advanced deduplication**: DOI + TF-IDF + metadata matching
        - **Batch processing**: Handle datasets with 50,000+ articles
        """)
        
        st.markdown("### AI Screening")
        st.markdown("""
        - **Parallel processing**: 8 concurrent workers for maximum speed
        - **Grok 4 Fast models**: Latest AI technology
        - **Cost transparency**: Real-time HKD cost tracking
        - **Checkpoint/resume**: Never lose progress
        """)
    
    with col2:
        st.markdown("### Quality Assurance")
        st.markdown("""
        - **Confidence scoring**: Transparent decision metrics
        - **Manual review**: Flag uncertain articles automatically
        - **Progress tracking**: Real-time status updates
        - **Error recovery**: Robust error handling
        """)
        
        st.markdown("### Export & Reporting")
        st.markdown("""
        - **PRISMA-ScR compliant**: Standard flow diagrams
        - **Multiple formats**: Excel, CSV export
        - **Detailed reports**: Include reasoning and confidence
        - **Citation support**: Ready for publication
        """)
    
    st.markdown("---")


if __name__ == "__main__":
    main()

