"""
Result Viewer Component

Display and interact with screening results
"""

import streamlit as st
import pandas as pd
from typing import Dict, List


def show_result_viewer(task_id: str, api_client):
    """
    Display screening results
    
    Args:
        task_id: Task ID
        api_client: APIClient instance
    """
    
    st.subheader("📊 Screening Results")
    
    try:
        # Get results summary
        summary = api_client.get_results_summary(task_id)
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Articles", summary['total_articles'])
        col2.metric("Included", summary['included'], delta=f"{summary['included']/summary['total_articles']*100:.1f}%")
        col3.metric("Excluded", summary['excluded'], delta=f"{summary['excluded']/summary['total_articles']*100:.1f}%")
        col4.metric("Manual Review", summary['manual_review'])
        
        # Display cost and confidence
        col1, col2 = st.columns(2)
        col1.metric("Total Cost", f"${summary['total_cost']:.2f}")
        col2.metric("Avg Confidence", f"{summary['avg_confidence']:.3f}")
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 All Results", "📊 Visualizations", "📥 Export"])
        
        with tab1:
            show_results_table(task_id, api_client)
        
        with tab2:
            show_visualizations(task_id, api_client)
        
        with tab3:
            show_export_options(task_id, api_client)
        
    except Exception as e:
        st.error(f"Error loading results: {e}")


def show_results_table(task_id: str, api_client):
    """Display interactive results table"""
    st.markdown("### Results Table")
    st.info("🚧 Results table - Coming soon!")


def show_visualizations(task_id: str, api_client):
    """Display visualizations"""
    st.markdown("### Visualizations")
    st.info("🚧 Visualizations - Coming soon!")


def show_export_options(task_id: str, api_client):
    """Display export options"""
    st.markdown("### Export Results")
    st.info("🚧 Export options - Coming soon!")
