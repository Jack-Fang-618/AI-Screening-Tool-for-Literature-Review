"""
AI Screening Page - Parallel Article Screening with Real-time Progress

Handles:
- PCC criteria input
- Model selection (Grok 4 Fast)
- Configurable parallel workers (1-60, optimized for 480 RPM limit)
- Real-time progress tracking
- Cost estimation and tracking
"""

import streamlit as st
import pandas as pd
import time
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from frontend.utils.api_client import APIClient

# Initialize API client
api_client = APIClient()

# Page configuration
st.set_page_config(
    page_title="AI Screening - PRISMA-ScR Toolkit",
    page_icon="🤖",
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
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .progress-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .decision-badge-relevant {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
    .decision-badge-irrelevant {
        background: #dc3545;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
    .decision-badge-uncertain {
        background: #ffc107;
        color: #333;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'screening_task_id' not in st.session_state:
    st.session_state.screening_task_id = None
if 'screening_in_progress' not in st.session_state:
    st.session_state.screening_in_progress = False
if 'last_status' not in st.session_state:
    st.session_state.last_status = None
if 'settings' not in st.session_state:
    st.session_state.settings = {}
# Session ID for data isolation (shared with Data Management)
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
# Track dataset IDs that belong to this session (shared with Data Management)
if 'my_dataset_ids' not in st.session_state:
    st.session_state.my_dataset_ids = set()
# Track task IDs that belong to this session
if 'my_task_ids' not in st.session_state:
    st.session_state.my_task_ids = set()


def load_settings_from_file():
    """Load settings from JSON file if not already in session state"""
    if not st.session_state.settings:
        settings_file = Path(__file__).parent.parent.parent / 'config' / 'user_settings.json'
        
        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    st.session_state.settings = json.load(f)
            except Exception:
                # Silently fail and use empty settings
                st.session_state.settings = {}


def parse_criteria_to_list(criteria_text: str) -> list:
    """
    Parse multi-line criteria text into a list of individual criteria.
    Removes bullet points, numbering, and empty lines.
    
    Args:
        criteria_text: Multi-line text with criteria (one per line)
        
    Returns:
        List of cleaned criteria strings
    """
    if not criteria_text:
        return []
    
    criteria_list = []
    for line in criteria_text.split('\n'):
        # Remove leading/trailing whitespace
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Remove common bullet points and numbering
        # Handles: -, *, •, 1., 2), etc.
        import re
        line = re.sub(r'^[\s\-\*•]+', '', line)  # Remove leading bullets
        line = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove numbering like "1." or "1)"
        line = line.strip()
        
        if line:
            criteria_list.append(line)
    
    return criteria_list


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


def show_progress_tracker():
    """Display real-time progress for active screening task"""
    if not st.session_state.screening_task_id:
        return
    
    try:
        status = api_client.get_screening_status(st.session_state.screening_task_id)
        st.session_state.last_status = status
        
        # Progress container
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        
        # Status header
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### Screening Status: **{status['status']}**")
        
        with col2:
            status_value = status['status'].upper() if isinstance(status['status'], str) else str(status['status']).upper()
            if status_value == 'RUNNING':
                st.markdown('<span style="color: #28a745; font-size: 1.2rem;">●</span> Active', unsafe_allow_html=True)
            elif status_value == 'COMPLETED':
                st.markdown('<span style="color: #17a2b8; font-size: 1.2rem;">✓</span> Complete', unsafe_allow_html=True)
            elif status_value == 'FAILED':
                st.markdown('<span style="color: #dc3545; font-size: 1.2rem;">✗</span> Failed', unsafe_allow_html=True)
        
        with col3:
            status_value = status['status'].upper() if isinstance(status['status'], str) else str(status['status']).upper()
            if status_value == 'RUNNING':
                if st.button("🛑 Cancel Screening", type="secondary", width="stretch"):
                    try:
                        api_client.cancel_screening(st.session_state.screening_task_id)
                        st.success("✅ Screening cancelled successfully")
                        st.session_state.screening_in_progress = False
                        del st.session_state.screening_task_id
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to cancel: {str(e)}")
        
        # Progress bar
        processed = status.get('processed', 0)
        total_articles = status.get('total_articles', 1)
        progress_percent = status.get('progress_percent', 0)
        
        st.progress(progress_percent / 100)
        st.caption(f"Progress: {processed:,} / {total_articles:,} articles ({progress_percent:.1f}%)")
        
        # Metrics row
        col1, col2 = st.columns(2)
        
        with col1:
            processed = status.get('processed', 0)
            st.metric("Processed", f"{processed:,}")
        
        with col2:
            cost = status.get('current_cost', 0)
            st.metric("Cost (HKD)", f"${cost:.2f}")
        
        # Decision breakdown
        if status.get('processed', 0) > 0:
            st.markdown("### Decision Breakdown")
            
            included = status.get('included', 0)
            excluded = status.get('excluded', 0)
            manual_review = status.get('manual_review', 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f'<div class="decision-badge-relevant">Relevant: {included}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'<div class="decision-badge-irrelevant">Irrelevant: {excluded}</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'<div class="decision-badge-uncertain">Uncertain: {manual_review}</div>', unsafe_allow_html=True)
        
        # Error message if failed
        if status['status'] == 'FAILED' and status.get('error'):
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.markdown(f"**Error:** {status['error']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Completion message
        if status['status'] == 'COMPLETED':
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### ✅ Screening Complete!")
            st.markdown(f"**Total articles processed:** {status.get('progress', {}).get('total', 0):,}")
            st.markdown(f"**Total cost:** HKD ${status.get('cost_hkd', 0):.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed cost report button
            if st.button("📊 View Detailed Cost Report", width="stretch"):
                try:
                    with st.spinner("Loading detailed cost analysis..."):
                        # Get full results from API
                        results = api_client.get_screening_results(st.session_state.screening_task_id)
                        
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown("### 💰 Actual Cost Breakdown")
                        
                        # Main metrics
                        col_cost1, col_cost2 = st.columns(2)
                        with col_cost1:
                            st.metric("Total Cost", f"HKD ${results.get('total_cost', 0):.2f}")
                        with col_cost2:
                            articles = results.get('total_articles', 1)
                            st.metric("Cost per Article", f"${results.get('total_cost', 0)/articles:.4f}")
                        
                        st.markdown("---")
                        st.markdown("#### 🔢 Token Usage Statistics")
                        
                        # Token usage
                        input_tokens = results.get('total_input_tokens', 0)
                        output_tokens = results.get('total_output_tokens', 0)
                        reasoning_tokens = results.get('total_reasoning_tokens', 0)
                        cached_tokens = results.get('total_cached_tokens', 0)
                        total_tokens = input_tokens + output_tokens + reasoning_tokens
                        
                        col_token1, col_token2, col_token3, col_token4 = st.columns(4)
                        with col_token1:
                            st.metric("Input Tokens", f"{input_tokens:,}")
                        with col_token2:
                            st.metric("Output Tokens", f"{output_tokens:,}")
                        with col_token3:
                            st.metric("Reasoning Tokens", f"{reasoning_tokens:,}")
                        with col_token4:
                            if cached_tokens > 0:
                                cache_rate = (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0
                                st.metric("Cached Tokens", f"{cached_tokens:,}", delta=f"{cache_rate:.1f}% cache hit")
                            else:
                                st.metric("Cached Tokens", "N/A", help="Cache data not available")
                        
                        st.markdown("---")
                        st.markdown("#### 💵 Cost Components")
                        
                        input_cost = results.get('input_cost', 0)
                        cached_cost = results.get('cached_cost', 0)
                        output_cost = results.get('output_cost', 0)
                        
                        col_breakdown1, col_breakdown2, col_breakdown3 = st.columns(3)
                        with col_breakdown1:
                            st.metric("Input Cost", f"HKD ${input_cost:.4f}")
                            st.caption(f"{input_tokens:,} tokens")
                        with col_breakdown2:
                            st.metric("Cached Input Cost", f"HKD ${cached_cost:.4f}")
                            if cached_tokens > 0:
                                savings = (input_cost - cached_cost) if input_cost > cached_cost else 0
                                st.caption(f"Saved: ${savings:.4f}")
                            else:
                                st.caption("No cache data")
                        with col_breakdown3:
                            st.metric("Output Cost", f"HKD ${output_cost:.4f}")
                            st.caption(f"{output_tokens:,} tokens")
                        
                        # Average per article
                        st.markdown("---")
                        st.markdown("#### 📈 Per-Article Averages")
                        avg_tokens = results.get('avg_tokens_per_article', 0)
                        st.markdown(f"- **Average tokens per article:** {avg_tokens:.1f}")
                        st.markdown(f"- **Average cost per article:** HKD ${results.get('total_cost', 0)/articles:.4f}")
                        st.markdown(f"- **Average time per article:** {results.get('total_time', 0)/articles:.2f}s")
                        
                        # Cache efficiency
                        if cached_tokens > 0:
                            st.markdown("---")
                            st.markdown("#### ⚡ Cache Efficiency")
                            cache_hit_rate = (cached_tokens / input_tokens * 100) if input_tokens > 0 else 0
                            st.markdown(f"- **Cache hit rate:** {cache_hit_rate:.1f}%")
                            st.markdown(f"- **Tokens served from cache:** {cached_tokens:,} / {input_tokens:,}")
                            
                            # Calculate what cost would be without cache
                            uncached_cost = results.get('total_cost', 0) + (cached_tokens * 0.00001556)  # Rough estimate
                            savings_pct = ((uncached_cost - results.get('total_cost', 0)) / uncached_cost * 100) if uncached_cost > 0 else 0
                            st.markdown(f"- **Estimated savings from cache:** HKD ${uncached_cost - results.get('total_cost', 0):.2f} ({savings_pct:.1f}%)")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Failed to load cost report: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 View Results", type="primary", width="stretch"):
                    st.switch_page("pages/3_Results.py")
            
            with col2:
                if st.button("🔄 Start New Screening", width="stretch"):
                    st.session_state.screening_task_id = None
                    st.session_state.screening_in_progress = False
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-refresh while running
        status_value = status['status'].upper() if isinstance(status['status'], str) else str(status['status']).upper()
        if status_value == 'RUNNING':
            time.sleep(5)
            st.rerun()
    
    except Exception as e:
        st.error(f"Failed to get screening status: {str(e)}")
        st.session_state.screening_in_progress = False


def main():
    # Load settings from file on first run
    load_settings_from_file()
    
    # Header
    st.markdown('<h1 class="main-header">AI Screening</h1>', unsafe_allow_html=True)
    st.markdown("**Parallel AI-powered article screening with configurable workers (1-50). Conservative settings with 25% safety margin for stable performance.**")
    
    st.markdown("---")
    
    # Show progress tracker if screening is active
    if st.session_state.screening_in_progress and st.session_state.screening_task_id:
        show_progress_tracker()
        return
    
    # Configuration Section
    st.markdown("## Screening Configuration")
    
    # Get available datasets
    try:
        datasets_response = api_client.get_datasets()
        # get_datasets() returns a list directly, not a dict
        all_datasets = datasets_response if isinstance(datasets_response, list) else datasets_response.get('datasets', [])
        
        # Filter 1: Only show datasets that belong to this session
        my_datasets = [
            ds for ds in all_datasets
            if ds['dataset_id'] in st.session_state.my_dataset_ids
        ]
        
        # Filter 2: From my datasets, show only cleaned/processed datasets (merged or manual_review)
        # Exclude raw uploads (individual PubMed, Scopus, WoS files)
        available_datasets = [
            ds for ds in my_datasets
            if ds.get('file_type') in ['merged', 'manual_review'] or
               'merged' in ds.get('filename', '').lower() or
               'clean' in ds.get('filename', '').lower() or
               'deduplicated' in ds.get('filename', '').lower()
        ]
        
        # If no processed datasets but there are my uploads, suggest processing first
        if not available_datasets and my_datasets:
            st.session_state.show_process_suggestion = True
        else:
            st.session_state.show_process_suggestion = False
            
    except Exception as e:
        st.error(f"Failed to load datasets: {str(e)}")
        available_datasets = []
        my_datasets = []
    
    # Show suggestion if no processed datasets but raw uploads exist
    if st.session_state.get('show_process_suggestion'):
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**⚠️ No processed datasets ready for screening**")
        st.markdown(f"You have {len(my_datasets)} uploaded dataset(s), but they need to be processed first.")
        st.markdown("**Next steps:**")
        st.markdown("1. Go to Data Management")
        st.markdown("2. Merge your datasets (with auto field mapping)")
        st.markdown("3. Run Smart Deduplication")
        st.markdown("4. Return here to screen the cleaned dataset")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("📊 Go to Data Management", type="primary", width="stretch"):
                st.switch_page("pages/1_Data_Management.py")
        
        with col_nav2:
            if st.button("🔄 Show All My Datasets Anyway", width="stretch"):
                # Allow override to show all my datasets
                available_datasets = my_datasets
                st.session_state.show_process_suggestion = False
                st.rerun()
        
        return
    
    if not available_datasets:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("**No datasets available**")
        st.markdown("Upload a dataset below or go to Data Management to process your data.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick upload option
        st.markdown("### Quick Upload")
        uploaded_file = st.file_uploader(
            "Upload a dataset file",
            type=['xlsx', 'xls', 'csv', 'ris', 'txt', 'nbib', 'medline'],
            help="Upload your cleaned dataset. For full processing (merging, deduplication), use Data Management."
        )
        
        col_upload1, col_upload2 = st.columns(2)
        
        with col_upload1:
            if uploaded_file and st.button("📤 Upload & Continue", type="primary", width="stretch"):
                with st.spinner("Uploading file..."):
                    try:
                        result = api_client.upload_file(uploaded_file)
                        st.success(f"✅ Uploaded: {result['filename']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")
        
        with col_upload2:
            if st.button("📊 Go to Data Management", width="stretch"):
                st.switch_page("pages/1_Data_Management.py")
        
        return
    
    # Dataset Selection
    st.markdown("### 1. Select Dataset")
    
    # Create more informative dataset labels
    dataset_options = {}
    for ds in available_datasets:
        file_type = ds.get('file_type', 'unknown')
        filename = ds['filename']
        record_count = ds['record_count']
        
        # Add type indicator
        if file_type == 'merged':
            type_label = "🔗 Merged"
        elif file_type == 'manual_review':
            type_label = "👁️ Review"
        elif 'clean' in filename.lower() or 'deduplicated' in filename.lower():
            type_label = "✨ Cleaned"
        else:
            type_label = "📄 Processed"
        
        label = f"{type_label} | {filename} ({record_count:,} records)"
        dataset_options[label] = ds['dataset_id']
    
    selected_dataset_label = st.selectbox(
        "Choose dataset to screen",
        options=list(dataset_options.keys()),
        help="Select the cleaned and deduplicated dataset for screening. Merged datasets are recommended."
    )
    
    selected_dataset_id = dataset_options[selected_dataset_label]
    
    # Get dataset preview
    try:
        preview = api_client.get_dataset_preview(selected_dataset_id)
        total_articles = preview['total_records']
        
        st.info(f"Dataset: **{preview['filename']}** | Records: **{total_articles:,}**")
        
        # Test Mode Option
        st.markdown("---")
        st.markdown("#### 🧪 Test Mode (Optional)")
        
        col_test1, col_test2 = st.columns([1, 2])
        
        with col_test1:
            enable_test_mode = st.checkbox(
                "Enable Test Mode",
                value=st.session_state.get('enable_test_mode', False),
                key='enable_test_mode',
                help="Screen only a subset of articles for testing before running full screening"
            )
        
        with col_test2:
            if enable_test_mode:
                test_sample_size = st.number_input(
                    "Number of articles to screen",
                    min_value=1,
                    max_value=total_articles,
                    value=min(100, total_articles),
                    step=10,
                    key='test_sample_size',
                    help="Select how many articles to screen in test mode"
                )
                
                # Update total_articles for cost estimation
                if test_sample_size < total_articles:
                    st.info(f"📊 Test mode: Will screen **{test_sample_size:,}** out of {total_articles:,} articles")
                    total_articles = test_sample_size
            else:
                test_sample_size = None
        
    except Exception as e:
        st.error(f"Failed to load dataset preview: {str(e)}")
        return
    
    st.markdown("---")
    
    # Load default criteria from settings if available
    if 'settings' not in st.session_state:
        st.session_state.settings = {}
    
    # Screening Criteria Input
    st.markdown("### 2. Define Screening Criteria")
    
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.markdown("""
        Define **inclusion** and **exclusion** criteria for screening articles.
        You can set default criteria in the Settings page.
        """)
    
    with col_header2:
        if st.button("⚙️ Settings", width="stretch"):
            st.switch_page("pages/4_Settings.py")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Inclusion Criteria")
        inclusion_criteria = st.text_area(
            "Articles must meet these criteria",
            value=st.session_state.settings.get('inclusion_criteria', ''),
            placeholder="""e.g.,
- Published in peer-reviewed journals
- Written in English
- Adult population (18+ years)
- Empirical research
- Published 2015-2025""",
            height=200,
            help="List criteria that articles MUST meet to be included",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("#### Exclusion Criteria")
        exclusion_criteria = st.text_area(
            "Articles will be excluded if they meet any of these",
            value=st.session_state.settings.get('exclusion_criteria', ''),
            placeholder="""e.g.,
- Conference abstracts or posters
- Grey literature
- Non-English publications
- Case studies
- Pediatric populations""",
            height=200,
            help="List criteria that will EXCLUDE articles from the review",
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    
    # Model and Worker Configuration
    st.markdown("### 3. Screening Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model = st.selectbox(
            "AI Model",
            options=[
                "grok-4-fast-reasoning",
                "grok-4-fast-non-reasoning"
            ],
            index=0,
            help="Grok 4 Fast Reasoning provides better quality but costs slightly more"
        )
        
        # Model info
        if model == "grok-4-fast-reasoning":
            st.caption("✓ Best quality | Reasoning-based decisions | HKD $1.556 per 1M input tokens")
        else:
            st.caption("✓ Fastest | Direct decisions | HKD $1.556 per 1M input tokens")
    
    with col2:
        num_workers = st.slider(
            "Parallel Workers",
            min_value=1,
            max_value=50,
            value=30,
            help="Number of concurrent workers. Conservative setting with safety margin. Recommended: 20-30 workers"
        )
        
        # Dynamic caption based on worker count
        if num_workers <= 8:
            st.caption(f"⚠️ Low: ~{num_workers * 20} articles/hour. Consider increasing for faster processing.")
        elif num_workers <= 20:
            st.caption(f"✓ Moderate: ~{num_workers * 20} articles/hour")
        elif num_workers <= 35:
            st.caption(f"✓ Recommended: ~{num_workers * 20} articles/hour (stable, safe margin)")
        else:
            st.caption(f"⚡ High: ~{num_workers * 20} articles/hour (approaching rate limits, may throttle)")
    
    st.markdown("---")
    
    # Cost Estimation
    st.markdown("### 4. Cost Estimation")
    
    if st.button("📊 Calculate Accurate Cost Estimate", type="secondary", width="stretch"):
        with st.spinner("Analyzing dataset and calculating cost..."):
            try:
                # Parse criteria text into lists
                inclusion_list = parse_criteria_to_list(inclusion_criteria)
                exclusion_list = parse_criteria_to_list(exclusion_criteria)
                
                # Get dataset preview to analyze content
                preview_data = api_client.get_dataset_preview(selected_dataset_id, limit=100)
                records = preview_data['records']
                
                # Calculate average tokens per article (title + abstract)
                total_chars = 0
                valid_records = 0
                for record in records:
                    title = str(record.get('title', ''))
                    abstract = str(record.get('abstract', ''))
                    content = f"{title} {abstract}"
                    if content.strip():
                        total_chars += len(content)
                        valid_records += 1
                
                if valid_records > 0:
                    avg_chars_per_article = total_chars / valid_records
                    # Rough estimate: 1 token ≈ 4 characters for English text
                    avg_tokens_per_article = avg_chars_per_article / 4
                else:
                    avg_tokens_per_article = 500  # Default fallback
                
                # Calculate criteria tokens (these will be cached after first article)
                criteria_text = f"Inclusion criteria:\n{inclusion_criteria}\n\nExclusion criteria:\n{exclusion_criteria}"
                criteria_chars = len(criteria_text)
                criteria_tokens = criteria_chars / 4
                
                # System prompt tokens (approximately)
                system_prompt_tokens = 300  # Estimated
                
                # Token calculation per article
                # First article: full cost (system + criteria + article + completion)
                # Subsequent articles: cached cost (system + criteria are cached, only article + completion)
                
                # Grok pricing (HKD per 1M tokens)
                input_price_per_1m = 1.556
                cached_price_per_1m = 0.1556  # 10% of input price (typical cache discount)
                output_price_per_1m = 6.224  # Completion tokens are more expensive
                
                # Estimated completion tokens per article (decision + reasoning)
                avg_completion_tokens = 150
                
                # First article (no cache)
                first_article_input_tokens = system_prompt_tokens + criteria_tokens + avg_tokens_per_article
                first_article_cost = (first_article_input_tokens * input_price_per_1m / 1_000_000) + \
                                   (avg_completion_tokens * output_price_per_1m / 1_000_000)
                
                # Subsequent articles (with cache)
                cached_tokens = system_prompt_tokens + criteria_tokens
                subsequent_input_tokens = avg_tokens_per_article  # Only article content
                subsequent_article_cost = (cached_tokens * cached_price_per_1m / 1_000_000) + \
                                        (subsequent_input_tokens * input_price_per_1m / 1_000_000) + \
                                        (avg_completion_tokens * output_price_per_1m / 1_000_000)
                
                # Total cost
                if total_articles > 0:
                    total_cost = first_article_cost + (subsequent_article_cost * (total_articles - 1))
                else:
                    total_cost = 0
                
                # Time estimation (based on API speed)
                # Assume ~3 seconds per article with parallel workers
                avg_seconds_per_article = 3
                estimated_time_seconds = (total_articles * avg_seconds_per_article) / num_workers
                
                # Display detailed breakdown
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("#### 💰 Cost Breakdown")
                
                col_est1, col_est2, col_est3 = st.columns(3)
                with col_est1:
                    st.metric("Total Cost", f"HKD ${total_cost:.2f}")
                with col_est2:
                    st.metric("Cost per Article", f"${total_cost/total_articles:.4f}" if total_articles > 0 else "$0")
                with col_est3:
                    st.metric("Estimated Time", format_time(estimated_time_seconds))
                
                st.markdown("---")
                st.markdown("**Token Usage Estimate:**")
                st.markdown(f"- Average tokens per article: ~{avg_tokens_per_article:.0f} (title + abstract)")
                st.markdown(f"- Criteria tokens (cached): ~{criteria_tokens:.0f}")
                st.markdown(f"- System prompt (cached): ~{system_prompt_tokens:.0f}")
                st.markdown(f"- Completion tokens per article: ~{avg_completion_tokens:.0f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Store in session for later use
                st.session_state.cost_estimate = {
                    'total_cost': total_cost,
                    'per_article_cost': total_cost/total_articles if total_articles > 0 else 0,
                    'estimated_time': estimated_time_seconds,
                    'avg_tokens': avg_tokens_per_article
                }
                
            except Exception as e:
                st.error(f"Failed to estimate cost: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Start Screening
    st.markdown("### 5. Start Screening")
    
    # Validation
    can_start = True
    validation_messages = []
    
    if not inclusion_criteria and not exclusion_criteria:
        can_start = False
        validation_messages.append("Please provide at least inclusion or exclusion criteria")
    
    if validation_messages:
        for msg in validation_messages:
            st.warning(msg)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button(
            "Start Screening",
            type="primary",
            disabled=not can_start,
            width="stretch"
        ):
            with st.spinner("Initializing screening task..."):
                try:
                    # Parse criteria text into lists
                    inclusion_list = parse_criteria_to_list(inclusion_criteria)
                    exclusion_list = parse_criteria_to_list(exclusion_criteria)
                    
                    # Prepare criteria dict
                    criteria = {
                        "population": "",  # Legacy field - can be empty
                        "concept": "",  # Legacy field - can be empty
                        "context": "",  # Legacy field - can be empty
                        "inclusion_criteria": inclusion_list,
                        "exclusion_criteria": exclusion_list
                    }
                    
                    # Get test mode settings from session state
                    enable_test_mode = st.session_state.get('enable_test_mode', False)
                    test_sample_size = st.session_state.get('test_sample_size', None)
                    
                    # Debug: Show what we're sending
                    limit_param = test_sample_size if enable_test_mode else None
                    st.info(f"🔍 Debug: enable_test_mode={enable_test_mode}, test_sample_size={test_sample_size}, limit_param={limit_param}")
                    
                    # Start screening with correct parameters
                    response = api_client.start_screening(
                        data_id=selected_dataset_id,
                        criteria=criteria,
                        model=model,
                        num_workers=num_workers,
                        limit=limit_param
                    )
                    
                    task_id = response['task_id']
                    st.session_state.screening_task_id = task_id
                    st.session_state.screening_in_progress = True
                    # Track this task as belonging to this session
                    st.session_state.my_task_ids.add(task_id)
                    
                    if enable_test_mode and test_sample_size:
                        st.success(f"🧪 Test screening started! Screening {test_sample_size} articles. Task ID: {task_id}")
                    else:
                        st.success(f"✅ Screening started! Task ID: {task_id}")
                    time.sleep(1)
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Failed to start screening: {str(e)}")
    
    with col2:
        if st.button("Save as Draft", width="stretch"):
            st.info("Draft save functionality coming soon!")
    
    with col3:
        if st.button("Load Template", width="stretch"):
            st.info("Template functionality coming soon!")
    
    # Help section
    with st.expander("Help & Tips"):
        st.markdown("""
        ### How to Use AI Screening
        
        1. **Select Dataset**: Choose your cleaned and deduplicated dataset
        2. **Define Screening Criteria**: 
           - **Inclusion**: What articles MUST have to be included
           - **Exclusion**: What will automatically exclude articles
           - Set default criteria in Settings page
        3. **Choose Model**: 
           - Reasoning model: Better quality, explains decisions
           - Non-reasoning: Faster, direct yes/no decisions
        4. **Set Workers**: 8 workers recommended for optimal speed
        5. **Start Screening**: Monitor progress in real-time
        
        ### Tips for Better Results
        
        - Be specific and clear in your criteria
        - Use bullet points for each criterion
        - Include examples when helpful
        - Review uncertain articles manually
        - Test with a small sample first
        
        ### Cost Information
        
        - **Grok 4 Fast**: ~HKD $0.01-0.05 per article
        - Cost varies based on abstract length
        - Reasoning model may cost slightly more due to explanation tokens
        - All costs are displayed in real-time during screening
        
        ### Screening Criteria Examples
        
        **Inclusion:**
        - Published in peer-reviewed journals
        - Written in English
        - Focus on adult population (18+ years)
        - Empirical research (quantitative or qualitative)
        - Published within the last 10 years (2015-2025)
        
        **Exclusion:**
        - Conference abstracts, posters, or presentations
        - Grey literature (dissertations, reports, etc.)
        - Non-English publications
        - Case studies or single-subject designs
        - Pediatric or adolescent populations (under 18 years)
        """)


if __name__ == "__main__":
    main()
