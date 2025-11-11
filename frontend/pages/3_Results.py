"""
Results Page - Screening Results Visualization and Export

Handles:
- Results summary and statistics
- Interactive results table with filters
- Decision breakdown charts
- PRISMA diagram generation
- Export functionality
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from frontend.utils.api_client import APIClient

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize API client
api_client = APIClient()

# Page configuration
st.set_page_config(
    page_title="Results - PRISMA-ScR Toolkit",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for beautiful styling
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
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-card-relevant {
        border-left: 4px solid #28a745;
    }
    .metric-card-irrelevant {
        border-left: 4px solid #dc3545;
    }
    .metric-card-uncertain {
        border-left: 4px solid #ffc107;
    }
    .metric-card-cost {
        border-left: 4px solid #667eea;
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
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_task_id' not in st.session_state:
    st.session_state.selected_task_id = None


def format_time(seconds):
    """Format seconds into readable time string"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def main():
    # Header
    st.markdown('<h1 class="main-header">Screening Results</h1>', unsafe_allow_html=True)
    st.markdown("**View and analyze AI screening results**")
    
    st.markdown("---")
    
    # Check if there's a task from screening page
    if hasattr(st.session_state, 'screening_task_id') and st.session_state.screening_task_id:
        st.session_state.selected_task_id = st.session_state.screening_task_id
    
    # Load available completed tasks
    completed_tasks = []
    try:
        completed_tasks = api_client.list_tasks(status='COMPLETED', limit=50)
    except Exception as e:
        logger.error(f"Failed to load tasks: {e}")
    
    # Task Selection
    if not st.session_state.selected_task_id:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**Select a screening task to view results**")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if completed_tasks:
            # Show dropdown of completed tasks
            st.markdown("### Recent Completed Screenings")
            
            # Format options for dropdown
            task_options = {}
            for task in completed_tasks:
                task_id = task['task_id']
                completed_at = task.get('completed_at', 'Unknown')
                if completed_at != 'Unknown':
                    # Parse ISO datetime and format nicely
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                        time_str = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        time_str = completed_at[:16]
                else:
                    time_str = 'Unknown time'
                
                total = task.get('total_items', 0)
                relevant = task.get('included_count', 0)
                irrelevant = task.get('excluded_count', 0)
                
                label = f"� {time_str} | {total} articles | ✅ {relevant} relevant, ❌ {irrelevant} irrelevant"
                task_options[label] = task_id
            
            selected_label = st.selectbox(
                "Choose a screening task:",
                options=list(task_options.keys()),
                help="Select from your completed screening tasks"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Load Selected Task", type="primary", width="stretch"):
                    st.session_state.selected_task_id = task_options[selected_label]
                    st.rerun()
            
            with col2:
                if st.button("🔬 Go to AI Screening", width="stretch"):
                    st.switch_page("pages/2_AI_Screening.py")
        else:
            # No completed tasks - show manual entry
            st.markdown("**No completed screening tasks found**")
            st.markdown("Please complete a screening task first, or enter a task ID manually.")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                task_id_input = st.text_input(
                    "Enter Task ID",
                    placeholder="e.g., 0723cf09-3af3-43b5-8a6b-63f8ab632c6e",
                    help="Enter the task ID from a previous screening"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Load Results", type="primary", width="stretch"):
                    if task_id_input:
                        st.session_state.selected_task_id = task_id_input
                        st.rerun()
                    else:
                        st.warning("Please enter a task ID")
            
            st.markdown("---")
            
            if st.button("Go to AI Screening", type="primary"):
                st.switch_page("pages/2_AI_Screening.py")
        
        return
    
    # Load results
    task_id = st.session_state.selected_task_id
    
    try:
        # Get task status
        status = api_client.get_screening_status(task_id)
        
        # Normalize status to uppercase for comparison
        status_upper = status.get('status', '').upper()
        
        # Show completion status badge
        if status_upper == 'COMPLETED':
            st.success("✅ Screening Status: Completed")
        else:
            st.warning(f"⚠️ Screening Status: {status['status']}")
        
        if status_upper != 'COMPLETED':
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown(f"**Screening Status: {status['status']}**")
            
            if status_upper == 'RUNNING':
                st.markdown("Screening is still in progress. Please wait for completion.")
                
                if st.button("Go to AI Screening", type="primary"):
                    st.switch_page("pages/2_AI_Screening.py")
            
            elif status_upper == 'FAILED':
                st.markdown(f"**Error:** {status.get('error', 'Unknown error')}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Get results
        st.markdown("---")
        st.markdown("**Loading results...**")
        results = api_client.get_screening_results(task_id)
        
        if not results:
            st.error("❌ Failed to load results - API returned empty response")
            st.write("Debug info:", results)
            return
        
        if not results.get('results'):
            st.warning("⚠️ No screening results found for this task")
            st.write("Debug - Results structure:", results.keys() if results else "None")
            return
        
        st.success(f"✅ Loaded {len(results.get('results', []))} screening results")
        
        # Summary Section
        st.markdown("## Summary")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Get counts from status (backend returns included, excluded, manual_review)
        relevant = status.get('included', 0)
        irrelevant = status.get('excluded', 0)
        uncertain = status.get('manual_review', 0)
        total = status.get('total', 0)
        
        with col1:
            st.markdown('<div class="metric-card metric-card-relevant">', unsafe_allow_html=True)
            st.metric("Relevant", relevant)
            if total > 0:
                st.caption(f"{relevant/total*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card metric-card-irrelevant">', unsafe_allow_html=True)
            st.metric("Irrelevant", irrelevant)
            if total > 0:
                st.caption(f"{irrelevant/total*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card metric-card-uncertain">', unsafe_allow_html=True)
            st.metric("Uncertain", uncertain)
            if total > 0:
                st.caption(f"{uncertain/total*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card metric-card-cost">', unsafe_allow_html=True)
            cost = status.get('current_cost', 0)
            st.metric("Total Cost", f"HKD ${cost:.2f}")
            elapsed = status.get('elapsed_time', 0)
            st.caption(f"Time: {format_time(elapsed)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Results Table
        st.markdown("## Screening Results")
        
        # Convert results to DataFrame
        if results.get('results'):
            df = pd.DataFrame(results['results'])
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                decision_filter = st.multiselect(
                    "Filter by Decision",
                    options=['include', 'exclude', 'manual_review'],
                    default=['include', 'exclude', 'manual_review'],
                    format_func=lambda x: {
                        'include': '✅ Relevant',
                        'exclude': '❌ Irrelevant',
                        'manual_review': '⚠️ Uncertain'
                    }.get(x, x)
                )
            
            with col2:
                min_confidence = st.slider(
                    "Minimum Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )
            
            with col3:
                search_term = st.text_input(
                    "Search Titles",
                    placeholder="Enter keywords..."
                )
            
            # Apply filters
            filtered_df = df.copy()
            
            if decision_filter:
                filtered_df = filtered_df[filtered_df['decision'].isin(decision_filter)]
            
            if min_confidence > 0:
                filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
            
            if search_term:
                filtered_df = filtered_df[
                    filtered_df['title'].str.contains(search_term, case=False, na=False)
                ]
            
            # Display count
            st.caption(f"Showing {len(filtered_df):,} of {len(df):,} articles")
            
            # Display table with all available columns
            # Determine which columns to show
            available_cols = ['title', 'abstract', 'decision', 'confidence', 'reasoning']
            display_columns = [col for col in available_cols if col in filtered_df.columns]
            
            if display_columns:
                # Format for display
                display_df = filtered_df[display_columns].copy()
                
                # Format confidence as percentage
                if 'confidence' in display_df.columns:
                    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
                
                # Map decision values for display
                if 'decision' in display_df.columns:
                    decision_map = {
                        'include': '✅ Relevant',
                        'exclude': '❌ Irrelevant',
                        'manual_review': '⚠️ Uncertain'
                    }
                    display_df['decision'] = display_df['decision'].map(lambda x: decision_map.get(x, x))
                
                # Truncate abstract for display (full text in export)
                if 'abstract' in display_df.columns:
                    display_df['abstract'] = display_df['abstract'].apply(
                        lambda x: (str(x)[:150] + '...') if pd.notna(x) and len(str(x)) > 150 else str(x)
                    )
                
                # Configure column widths and enable horizontal scroll
                column_config = {
                    "title": st.column_config.TextColumn("Title", width="medium"),
                    "abstract": st.column_config.TextColumn("Abstract", width="large"),
                    "decision": st.column_config.TextColumn("Decision", width="small"),
                    "confidence": st.column_config.TextColumn("Confidence", width="small"),
                    "reasoning": st.column_config.TextColumn("AI Reasoning", width="large")
                }
                
                st.dataframe(
                    display_df,
                    width="stretch",
                    height=500,
                    column_config=column_config,
                    hide_index=True
                )
            else:
                st.dataframe(filtered_df, width="stretch", height=500)
        else:
            st.info("No results available")
        
        st.markdown("---")
        
        # Export Section
        st.markdown("## Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export All (CSV)", type="primary", width="stretch"):
                if results.get('results'):
                    df_export = pd.DataFrame(results['results'])
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"screening_results_{task_id[:8]}.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                else:
                    st.warning("No results to export")
        
        with col2:
            if st.button("✅ Export Relevant Only (CSV)", width="stretch"):
                if results.get('results'):
                    df_export = pd.DataFrame(results['results'])
                    df_relevant = df_export[df_export['decision'] == 'include']
                    csv = df_relevant.to_csv(index=False)
                    st.download_button(
                        label="Download Relevant CSV",
                        data=csv,
                        file_name=f"relevant_articles_{task_id[:8]}.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                else:
                    st.warning("No results to export")
        
        with col3:
            if st.button("📊 Export Summary (JSON)", width="stretch"):
                if results.get('summary'):
                    import json
                    summary_json = json.dumps(results['summary'], indent=2)
                    st.download_button(
                        label="Download Summary JSON",
                        data=summary_json,
                        file_name=f"screening_summary_{task_id[:8]}.json",
                        mime="application/json",
                        width="stretch"
                    )
                else:
                    st.warning("No summary to export")
        
        st.markdown("---")
        
        # Additional Information
        with st.expander("Screening Details"):
            st.markdown("### Task Information")
            st.markdown(f"**Task ID:** `{task_id}`")
            st.markdown(f"**Status:** {status['status']}")
            st.markdown(f"**Total Articles:** {total:,}")
            st.markdown(f"**Total Cost:** HKD ${status.get('current_cost', 0):.2f}")
            
            # Show metadata from status
            metadata = status.get('metadata', {})
            if metadata:
                st.markdown("### Screening Configuration")
                st.markdown(f"**Model:** {metadata.get('model', 'N/A')}")
                st.markdown(f"**Workers:** {metadata.get('num_workers', 'N/A')}")
                
                criteria = metadata.get('criteria', {})
                if criteria:
                    st.markdown("### Screening Criteria")
                    st.markdown(f"**Population:** {criteria.get('population', 'N/A')}")
                    st.markdown(f"**Concept:** {criteria.get('concept', 'N/A')}")
                    st.markdown(f"**Context:** {criteria.get('context', 'N/A')}")
        
        # Actions
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start New Screening", width="stretch"):
                st.session_state.selected_task_id = None
                st.session_state.screening_task_id = None
                st.session_state.screening_in_progress = False
                st.switch_page("pages/2_AI_Screening.py")
        
        with col2:
            if st.button("Back to Data Management", width="stretch"):
                st.switch_page("pages/1_Data_Management.py")
        
        with col3:
            if st.button("Clear Results", type="secondary", width="stretch"):
                st.session_state.selected_task_id = None
                st.rerun()
    
    except Exception as e:
        st.error(f"❌ Failed to load results: {str(e)}")
        
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.markdown("**Error Loading Results**")
        st.markdown(f"**Error Type:** {type(e).__name__}")
        st.markdown(f"**Details:** {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show detailed traceback in expander
        with st.expander("🔍 Technical Details"):
            import traceback
            st.code(traceback.format_exc())
        
        if st.button("Go to AI Screening"):
            st.session_state.selected_task_id = None
            st.switch_page("pages/2_AI_Screening.py")


if __name__ == "__main__":
    main()
