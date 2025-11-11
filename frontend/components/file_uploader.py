"""
File Uploader Component

Streamlit component for uploading and previewing files
"""

import streamlit as st
from typing import List
import pandas as pd


def show_file_uploader():
    """Display file uploader interface"""
    
    st.subheader("📤 Upload Files")
    st.markdown("Upload Excel (.xlsx, .xls), CSV (.csv), or RIS (.ris) files")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['xlsx', 'xls', 'csv', 'ris', 'txt'],
        accept_multiple_files=True,
        help="Upload files exported from PubMed, Scopus, Web of Science, or Embase"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        # Preview uploaded files
        with st.expander("📋 View Uploaded Files"):
            for i, file in enumerate(uploaded_files):
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.text(f"{i+1}. {file.name}")
                with col2:
                    st.text(f"{file.size / 1024:.1f} KB")
                with col3:
                    st.text(file.type)
        
        return uploaded_files
    
    return None
