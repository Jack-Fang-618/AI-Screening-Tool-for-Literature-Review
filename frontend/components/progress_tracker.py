"""
Progress Tracker Component

Real-time progress tracking for AI screening tasks
"""

import streamlit as st
import time
from typing import Dict


def show_progress_tracker(task_id: str, api_client):
    """
    Display real-time progress for screening task
    
    Args:
        task_id: Task ID to track
        api_client: APIClient instance
    """
    
    st.subheader("📊 Screening Progress")
    
    # Create placeholder for dynamic updates
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.container()
    
    # Poll backend every 2 seconds
    while True:
        try:
            status = api_client.get_screening_status(task_id)
            
            # Update progress bar
            progress_percent = status['progress_percent']
            progress_bar.progress(progress_percent / 100)
            
            # Update status text
            status_text.text(f"Status: {status['status']} - {progress_percent:.1f}% complete")
            
            # Update metrics
            with metrics_container:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Processed", f"{status['processed']}/{status['total_articles']}")
                col2.metric("Included", status['included'])
                col3.metric("Excluded", status['excluded'])
                col4.metric("Manual Review", status['manual_review'])
            
            # Check if completed
            if status['status'] == 'completed':
                st.success("✅ Screening completed!")
                break
            elif status['status'] == 'failed':
                st.error("❌ Screening failed")
                break
            
            # Wait 2 seconds before next poll
            time.sleep(2)
            
        except Exception as e:
            st.error(f"Error tracking progress: {e}")
            break
