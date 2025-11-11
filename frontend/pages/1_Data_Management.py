"""
Data Management Page - Beautiful UI for Dataset Processing

Handles:
- File upload (Excel, CSV, RIS)
- Field mapping with auto-detection
- Multi-dataset merging
- Smart deduplication with manual review
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import requests

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from frontend.utils.api_client import APIClient

# Initialize API client
api_client = APIClient()

# Page configuration
st.set_page_config(
    page_title="Data Management - PRISMA-ScR Toolkit",
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
    .step-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin-right: 0.5rem;
    }
    .step-badge-inactive {
        display: inline-block;
        background: #e0e0e0;
        color: #666;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        margin-right: 0.5rem;
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
    .dataset-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .dataset-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_datasets' not in st.session_state:
    st.session_state.uploaded_datasets = []
if 'merged_dataset_id' not in st.session_state:
    st.session_state.merged_dataset_id = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1


def sync_with_backend():
    """Sync session state with backend datasets (handles backend restarts)"""
    try:
        # Add timeout to prevent hanging
        import requests
        original_timeout = api_client.session.timeout if hasattr(api_client.session, 'timeout') else None
        
        # Temporarily set a short timeout for this check
        backend_datasets = api_client.get_datasets()
        
        # Create a map of existing dataset IDs
        backend_ids = {ds['dataset_id'] for ds in backend_datasets}
        
        # Remove datasets from session that no longer exist in backend
        st.session_state.uploaded_datasets = [
            ds for ds in st.session_state.uploaded_datasets 
            if ds['dataset_id'] in backend_ids
        ]
        
        # Add new datasets from backend that aren't in session
        session_ids = {ds['dataset_id'] for ds in st.session_state.uploaded_datasets}
        for backend_ds in backend_datasets:
            if backend_ds['dataset_id'] not in session_ids:
                st.session_state.uploaded_datasets.append(backend_ds)
        
    except requests.exceptions.Timeout:
        st.error("⚠️ Backend connection timeout. Please check if the backend server is running.")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to backend. Please make sure the backend server is running at http://localhost:8000")
    except Exception as e:
        # Backend might be down or no datasets exist
        st.error(f"⚠️ Could not sync with backend: {e}")


def main():
    # Check for navigation request
    if st.session_state.get('navigate_to_screening', False):
        st.session_state.navigate_to_screening = False
        st.switch_page("pages/2_AI_Screening.py")
    
    # Sync with backend on page load (handles backend restarts)
    sync_with_backend()
    
    # Header
    st.markdown('<h1 class="main-header">Data Management</h1>', unsafe_allow_html=True)
    st.markdown("**Intelligent data processing pipeline for systematic reviews**")
    
    st.markdown("---")
    
    # Progress indicator
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        badge_class = "step-badge" if st.session_state.current_step >= 1 else "step-badge-inactive"
        st.markdown(f'<span class="{badge_class}">STEP 1</span>', unsafe_allow_html=True)
        st.markdown("**Upload**")
        st.caption("Import datasets")
    
    with col2:
        badge_class = "step-badge" if st.session_state.current_step >= 2 else "step-badge-inactive"
        st.markdown(f'<span class="{badge_class}">STEP 2</span>', unsafe_allow_html=True)
        st.markdown("**Map Fields**")
        st.caption("Standardize columns")
    
    with col3:
        badge_class = "step-badge" if st.session_state.current_step >= 3 else "step-badge-inactive"
        st.markdown(f'<span class="{badge_class}">STEP 3</span>', unsafe_allow_html=True)
        st.markdown("**Merge**")
        st.caption("Combine sources")
    
    with col4:
        badge_class = "step-badge" if st.session_state.current_step >= 4 else "step-badge-inactive"
        st.markdown(f'<span class="{badge_class}">STEP 4</span>', unsafe_allow_html=True)
        st.markdown("**Deduplicate**")
        st.caption("Remove duplicates")
    
    st.markdown("---")
    
    # Step 1: Upload Files
    with st.expander("**STEP 1: Upload Datasets**", expanded=st.session_state.current_step == 1):
        # Add Clear All button at the top
        if st.session_state.uploaded_datasets:
            col_header1, col_header2 = st.columns([3, 1])
            with col_header1:
                st.markdown("**Supported formats**: Excel (.xlsx, .xls), CSV, RIS, NBIB (PubMed/MEDLINE)")
            with col_header2:
                if st.button("🗑️ Clear All Datasets", type="secondary", width="stretch", key="clear_all_datasets"):
                    if st.session_state.get('confirm_clear_all'):
                        # Actually delete
                        with st.spinner("Deleting all datasets..."):
                            try:
                                result = api_client.delete_all_datasets()
                                st.session_state.uploaded_datasets = []
                                st.session_state.merged_dataset_id = None
                                st.session_state.dedup_result = None
                                st.session_state.current_step = 1
                                st.session_state.confirm_clear_all = False
                                st.success(f"✅ Deleted {result.get('database_deleted', 0)} datasets")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete datasets: {e}")
                                st.session_state.confirm_clear_all = False
                    else:
                        # Ask for confirmation
                        st.session_state.confirm_clear_all = True
                        st.rerun()
            
            # Show confirmation warning if needed
            if st.session_state.get('confirm_clear_all'):
                st.warning("⚠️ Are you sure? This will delete ALL datasets (uploaded, merged, review lists). Click 'Clear All Datasets' again to confirm, or upload new files to cancel.")
        else:
            st.markdown("**Supported formats**: Excel (.xlsx, .xls), CSV, RIS, NBIB (PubMed/MEDLINE)")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['xlsx', 'xls', 'csv', 'ris', 'txt', 'nbib', 'medline'],
            accept_multiple_files=True,
            help="Upload one or more dataset files. Supports PubMed, Scopus, Web of Science, IEEE, and custom formats."
        )
        
        if uploaded_files:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(f"{len(uploaded_files)} file(s) selected")
            
            with col2:
                if st.button("Upload All", type="primary", width="stretch"):
                    with st.spinner("Uploading and parsing files..."):
                        upload_results = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, file in enumerate(uploaded_files):
                            status_text.text(f"Processing {file.name}...")
                            
                            try:
                                result = api_client.upload_file(file)
                                upload_results.append(result)
                                st.session_state.uploaded_datasets.append(result)
                            except Exception as e:
                                st.error(f"Failed to upload {file.name}: {str(e)}")
                            
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        if upload_results:
                            st.success(f"Successfully uploaded {len(upload_results)} dataset(s)!")
                            st.session_state.current_step = 2
                            st.rerun()
        
        # Show uploaded datasets
        if st.session_state.uploaded_datasets:
            st.markdown("### Uploaded Datasets")
            
            for dataset in st.session_state.uploaded_datasets:
                st.markdown(f"""
                <div class="dataset-card">
                    <strong>{dataset['filename']}</strong><br>
                    <small>{dataset['record_count']} records • {dataset['file_type']} • ID: {dataset['dataset_id'][:8]}...</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Preview button
                if st.button("Preview", key=f"view_{dataset['dataset_id']}"):
                    st.session_state.preview_dataset = dataset['dataset_id']
                
                # Show preview if selected
                if st.session_state.get('preview_dataset') == dataset['dataset_id']:
                    with st.spinner("Loading preview..."):
                        try:
                            preview = api_client.get_dataset_preview(dataset['dataset_id'])
                            
                            st.caption(f"**Columns** ({preview['column_count']}): {', '.join(preview['columns'][:10])}")
                            
                            if preview['records']:
                                st.dataframe(
                                    pd.DataFrame(preview['records']),
                                    width="stretch",
                                    height=200
                                )
                        except Exception as e:
                            st.error(f"❌ Could not load preview: {e}")
                            st.info("💡 The dataset may have been lost due to backend restart. Please re-upload the file.")
                            # Clear this dataset from session
                            st.session_state.uploaded_datasets = [
                                ds for ds in st.session_state.uploaded_datasets 
                                if ds['dataset_id'] != dataset['dataset_id']
                            ]
                            st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)
    
    # Step 2: Merge Datasets (Auto-mapping happens during merge)
    if len(st.session_state.uploaded_datasets) > 1:
        with st.expander("**STEP 2: Merge Datasets**", expanded=st.session_state.current_step == 2):
            st.markdown("**Combine multiple datasets into a unified dataset**")
            st.markdown("*Field mapping happens automatically using AI during merge*")
            
            st.markdown("### Select datasets to merge:")
            
            # Filter out already-merged datasets (only show original uploads)
            original_datasets = [
                ds for ds in st.session_state.uploaded_datasets 
                if ds.get('file_type') not in ['merged', 'manual_review']
            ]
            
            if not original_datasets:
                st.warning("No original datasets available. Please upload files first.")
            else:
                # Add "Select All" checkbox
                select_all = st.checkbox("✅ Select All Datasets", key="select_all_datasets")
                
                st.markdown("---")
                
                selected_for_merge = []
                for dataset in original_datasets:
                    # If "Select All" is checked, auto-add to selection
                    if select_all:
                        is_selected = True
                        selected_for_merge.append(dataset['dataset_id'])
                        # Show checkbox state when "Select All" is active
                        st.markdown(f"✅ {dataset['filename']} ({dataset['record_count']} records)")
                    else:
                        is_selected = st.checkbox(
                            f"{dataset['filename']} ({dataset['record_count']} records)",
                            key=f"merge_{dataset['dataset_id']}"
                        )
                        if is_selected:
                            selected_for_merge.append(dataset['dataset_id'])
            
            if len(selected_for_merge) >= 2:
                total_records = sum(d['record_count'] for d in original_datasets if d['dataset_id'] in selected_for_merge)
                st.info(f"📊 {len(selected_for_merge)} datasets selected, {total_records} total records")
                st.info(f"🤖 AI will automatically analyze and map fields from each dataset to standard schema")
                
                if st.button("🔗 Merge Datasets with AI Mapping", type="primary", width="stretch"):
                    with st.spinner("🤖 AI is analyzing fields and merging datasets..."):
                        try:
                            merge_result = api_client.merge_datasets(
                                dataset_ids=selected_for_merge,
                                source_names=[
                                    d['filename'] for d in original_datasets
                                    if d['dataset_id'] in selected_for_merge
                                ]
                            )
                            
                            st.session_state.merged_dataset_id = merge_result['merged_dataset_id']
                            st.session_state.merge_result = merge_result  # Store for download
                            
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.markdown(f"### ✅ Merge Complete!")
                            st.markdown(f"**Total records**: {merge_result['total_records']}")
                            st.markdown(f"**Sources merged**: {', '.join(merge_result['sources_merged'])}")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            with st.expander("📋 View Merge Summary"):
                                st.text(merge_result['summary'])
                            
                            st.session_state.current_step = 3
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Merge failed: {str(e)}")
            elif selected_for_merge:
                st.warning("Select at least 2 datasets to merge")
    
    # Download Merged Dataset (show if merge was completed)
    if st.session_state.get('merged_dataset_id') and st.session_state.get('merge_result'):
        st.markdown("---")
        st.markdown("### 📥 Download Merged Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download as CSV", width="stretch"):
                try:
                    with st.spinner("Preparing CSV file..."):
                        csv_data = api_client.export_dataset(st.session_state.merged_dataset_id, format='csv')
                        st.download_button(
                            label="💾 Save CSV File",
                            data=csv_data,
                            file_name=f"merged_dataset_{st.session_state.merge_result['total_records']}_records.csv",
                            mime="text/csv",
                            width="stretch"
                        )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col2:
            if st.button("📥 Download as Excel", width="stretch"):
                try:
                    with st.spinner("Preparing Excel file..."):
                        excel_data = api_client.export_dataset(st.session_state.merged_dataset_id, format='excel')
                        st.download_button(
                            label="💾 Save Excel File",
                            data=excel_data,
                            file_name=f"merged_dataset_{st.session_state.merge_result['total_records']}_records.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch"
                        )
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col3:
            st.info(f"📊 {st.session_state.merge_result['total_records']} records ready")
    
    # Step 3: Smart Deduplication
    dataset_to_dedupe = st.session_state.merged_dataset_id or (
        st.session_state.uploaded_datasets[0]['dataset_id'] if st.session_state.uploaded_datasets else None
    )
    
    if dataset_to_dedupe:
        with st.expander("**STEP 3: Remove Duplicates (Smart AI Deduplication)**", expanded=st.session_state.current_step == 3):
            st.markdown("**🧠 Intelligent 5-Step Deduplication with Manual Review**")
            
            # Add explanation
            with st.info("ℹ️ **How Smart Deduplication Works**"):
                st.markdown("""
                The system uses a **5-step intelligent workflow**:
                
                **Step 1: Data Quality Check** 📋
                - Removes records without title or abstract
                - Removes records with very short abstracts (< 50 chars)
                - Ensures only valid data enters deduplication
                
                **Step 2: DOI Exact Matching** 🎯
                - Most reliable deduplication method
                - Same DOI = definitely same paper → removed automatically
                
                **Step 3: Title Similarity Detection** 🔍
                - Uses TF-IDF to find papers with similar titles
                - Default threshold: 0.85 (recommended for safety)
                
                **Step 4: Metadata Validation** ✅
                - **ONLY for title-similar pairs**
                - Checks author + journal + year:
                  - ✅ **All match** → Confirmed duplicate → removed
                  - ⚠️ **Don't match** → Flagged for manual review
                
                **Step 5: Manual Review List** 👤
                - Papers that are similar but have different metadata
                - **You decide** whether to keep or remove
                - Download the review list with all details
                
                **What's Different from Basic Deduplication:**
                - ✅ Pre-filters invalid data first
                - ✅ Sequential, logical workflow
                - ✅ Metadata check only for similar titles (not independent)
                - ✅ Human-in-the-loop for ambiguous cases
                - ✅ Detailed review list with all article info
                """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                title_threshold = st.slider(
                    "Title similarity threshold",
                    min_value=0.70,
                    max_value=0.95,
                    value=0.85,
                    step=0.05,
                    help="Higher = stricter matching (0.85 recommended)"
                )
            
            with col2:
                min_abstract_length = st.number_input(
                    "Minimum abstract length",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Records with shorter abstracts will be removed"
                )
            
            if st.button("🚀 Start Smart Deduplication", type="primary", width="stretch"):
                with st.spinner("Running intelligent deduplication... This may take a few minutes."):
                    try:
                        dedup_result = api_client.smart_deduplicate(
                            dataset_id=dataset_to_dedupe,
                            title_threshold=title_threshold,
                            min_abstract_length=min_abstract_length,
                            min_title_length=10
                        )
                        
                        # Store deduplication result in session state
                        st.session_state.dedup_result = dedup_result
                        st.session_state.dataset_to_dedupe = dataset_to_dedupe
                        
                        # Store review dataset ID if exists
                        if dedup_result.get('review_dataset_id'):
                            st.session_state['review_dataset_id'] = dedup_result['review_dataset_id']
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Deduplication failed: {str(e)}")
            
            # Display deduplication results if they exist in session state
            if 'dedup_result' in st.session_state and st.session_state.dedup_result:
                dedup_result = st.session_state.dedup_result
                dataset_to_dedupe = st.session_state.dataset_to_dedupe
                
                # Show results
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("### ✅ Smart Deduplication Complete!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Metrics - 5 columns for 5 steps
                st.markdown("#### Step-by-Step Results")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "1️⃣ Original",
                        dedup_result['original_count']
                    )
                
                with col2:
                    invalid_rate = dedup_result['invalid_count']/dedup_result['original_count']*100
                    st.metric(
                        "📋 Invalid",
                        dedup_result['invalid_count'],
                        delta=f"-{invalid_rate:.1f}%",
                        delta_color="inverse"
                    )
                
                with col3:
                    doi_rate = dedup_result['doi_duplicates']/dedup_result['after_quality_check']*100
                    st.metric(
                        "🎯 DOI Dupes",
                        dedup_result['doi_duplicates'],
                        delta=f"-{doi_rate:.1f}%",
                        delta_color="inverse"
                    )
                
                with col4:
                    st.metric(
                        "✅ Title Dupes",
                        dedup_result['title_duplicates_confirmed']
                    )
                
                with col5:
                    st.metric(
                        "👤 Need Review",
                        dedup_result['title_duplicates_need_review'],
                        help="Papers with similar titles but different metadata"
                    )
                
                # Final count
                st.markdown("---")
                total_removed = dedup_result['original_count'] - dedup_result['final_count']
                removal_rate = total_removed / dedup_result['original_count'] * 100
                
                col_final1, col_final2, col_final3 = st.columns(3)
                with col_final1:
                    st.metric("📊 Final Clean Records", dedup_result['final_count'])
                with col_final2:
                    st.metric(
                        "🗑️ Total Removed",
                        total_removed,
                        delta=f"-{removal_rate:.1f}%",
                        delta_color="inverse"
                    )
                with col_final3:
                    if dedup_result['title_duplicates_need_review'] > 0:
                        st.metric(
                            "⚠️ Pending Review",
                            dedup_result['title_duplicates_need_review'],
                            help="Review these before finalizing"
                        )
                
                # Manual Review Section
                if dedup_result.get('review_dataset_id'):
                    st.markdown("---")
                    st.markdown("### 📋 Manual Review Required")
                    st.warning(
                        f"**{dedup_result['title_duplicates_need_review']} pairs** of similar papers "
                        "have different metadata and need your review."
                    )
                    
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        # Download review list as CSV
                        if st.button("📥 Download Review List (CSV)", width="stretch", key="dl_csv_review"):
                            try:
                                csv_data = api_client.export_dataset(
                                    dedup_result['review_dataset_id'],
                                    format='csv'
                                )
                                st.download_button(
                                    label="⬇️ Save Review List",
                                    data=csv_data,
                                    file_name=f"manual_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    width="stretch"
                                )
                            except Exception as e:
                                st.error(f"Export failed: {e}")
                    
                    with col_r2:
                        # Download review list as Excel
                        if st.button("📥 Download Review List (Excel)", width="stretch", key="dl_excel_review"):
                            try:
                                excel_data = api_client.export_dataset(
                                    dedup_result['review_dataset_id'],
                                    format='excel'
                                )
                                st.download_button(
                                    label="⬇️ Save Review List",
                                    data=excel_data,
                                    file_name=f"manual_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    width="stretch"
                                )
                            except Exception as e:
                                st.error(f"Export failed: {e}")
                    
                    # Preview review list
                    with st.expander("👁️ Preview Review List"):
                        try:
                            review_preview = api_client.get_dataset_preview(
                                dedup_result['review_dataset_id'],
                                limit=10
                            )
                            st.dataframe(review_preview['records'], width="stretch")
                            st.caption(f"Showing first 10 of {review_preview['total_records']} pairs")
                        except Exception as e:
                            st.error(f"Preview failed: {e}")
                
                # Strategies used
                st.markdown("---")
                st.markdown("### 🔧 Strategies Applied")
                strategy_names = {
                    'quality_check': '📋 Quality Check',
                    'doi_matching': '🎯 DOI Exact Matching',
                    'title_similarity': '🔍 Title Similarity',
                    'metadata_validation': '✅ Metadata Validation'
                }
                cols = st.columns(len(dedup_result['strategies_used']))
                for i, strategy in enumerate(dedup_result['strategies_used']):
                    with cols[i]:
                        st.success(strategy_names.get(strategy, strategy))
                
                # Detailed report
                with st.expander("📊 Detailed Deduplication Report"):
                    st.text(dedup_result['report'])
                
                # Action buttons
                st.markdown("### Next Steps")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📥 Export Clean Dataset", width="stretch", key="export_clean_btn"):
                        try:
                            # Export the deduplicated dataset
                            clean_data = api_client.export_dataset(
                                dataset_to_dedupe,
                                format='excel'
                            )
                            st.download_button(
                                label="⬇️ Download Clean Dataset",
                                data=clean_data,
                                file_name=f"clean_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                width="stretch"
                            )
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
                
                with col2:
                    # Use a dedicated key to avoid conflicts
                    screening_btn = st.button(
                        "🤖 Start AI Screening →", 
                        type="primary", 
                        width="stretch",
                        key="start_screening_btn"
                    )
                    
                    if screening_btn:
                        # Store dataset ID in session state
                        st.session_state.screening_dataset_id = dataset_to_dedupe
                        st.session_state.navigate_to_screening = True
                        st.rerun()


if __name__ == "__main__":
    main()
