"""
API Client - HTTP wrapper for FastAPI backend

Handles all communication between Streamlit frontend and FastAPI backend
"""

import requests
from typing import Dict, List, Optional, Any, Union, BinaryIO
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class APIClient:
    """Client for communicating with FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 10):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of FastAPI backend
            timeout: Request timeout in seconds (default: 10)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    # ===== General Methods =====
    
    def health_check(self) -> Dict:
        """Check if backend is healthy"""
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # ===== Data Management Methods =====
    
    def upload_file(self, file) -> Dict:
        """
        Upload a file for processing
        
        Args:
            file: Either a file path (str/Path) or Streamlit UploadedFile object
            
        Returns:
            Dict with dataset_id and metadata
        """
        # Handle different file input types
        if isinstance(file, (str, Path)):
            # File path provided
            file_path = Path(file)
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, self._get_content_type(file_path.name))}
                # Remove Content-Type header for multipart/form-data
                headers = {k: v for k, v in self.session.headers.items() if k.lower() != 'content-type'}
                response = requests.post(
                    f"{self.base_url}/api/data/upload",
                    files=files,
                    headers=headers
                )
        else:
            # Assume it's a Streamlit UploadedFile object
            files = {'file': (file.name, file.getvalue(), file.type)}
            # Remove Content-Type header for multipart/form-data
            headers = {k: v for k, v in self.session.headers.items() if k.lower() != 'content-type'}
            response = requests.post(
                f"{self.base_url}/api/data/upload",
                files=files,
                headers=headers
            )
        
        response.raise_for_status()
        return response.json()
    
    def _get_content_type(self, filename: str) -> str:
        """Get content type based on file extension"""
        ext = filename.lower().split('.')[-1]
        content_types = {
            'csv': 'text/csv',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'xls': 'application/vnd.ms-excel',
            'ris': 'application/x-research-info-systems',
            'txt': 'text/plain'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def parse_file(self, file_id: str) -> Dict:
        """Parse an uploaded file"""
        response = self.session.get(f"{self.base_url}/api/data/parse/{file_id}")
        response.raise_for_status()
        return response.json()
    
    def get_dataset_preview(self, dataset_id: str, limit: int = 20) -> Dict:
        """
        Get preview of dataset (first N rows)
        
        Args:
            dataset_id: Dataset ID
            limit: Number of rows to preview (default: 20)
            
        Returns:
            Dict with dataset_id, total_records, preview_count, records, columns, column_count
        """
        response = self.session.get(
            f"{self.base_url}/api/data/preview/{dataset_id}",
            params={'limit': limit}
        )
        response.raise_for_status()
        data = response.json()
        
        # Add columns and column_count for frontend compatibility
        if 'records' in data and len(data['records']) > 0:
            data['columns'] = list(data['records'][0].keys())
            data['column_count'] = len(data['columns'])
        else:
            data['columns'] = []
            data['column_count'] = 0
        
        return data
    
    def get_dataset_columns(self, dataset_id: str) -> List[str]:
        """
        Get column names for a dataset
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            List of column names
        """
        response = self.session.get(f"{self.base_url}/api/data/columns/{dataset_id}")
        response.raise_for_status()
        return response.json()
    
    def get_datasets(self) -> List[Dict]:
        """
        Get list of all uploaded datasets
        
        Returns:
            List of dataset info dictionaries
        """
        response = self.session.get(
            f"{self.base_url}/api/data/datasets",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def delete_dataset(self, dataset_id: str) -> Dict:
        """
        Delete a dataset
        
        Args:
            dataset_id: Dataset ID to delete
            
        Returns:
            Success message
        """
        response = self.session.delete(f"{self.base_url}/api/data/dataset/{dataset_id}")
        response.raise_for_status()
        return response.json()
    
    def delete_all_datasets(self) -> Dict:
        """
        Delete ALL datasets from database and storage
        
        WARNING: This removes all uploaded datasets permanently!
        
        Returns:
            Dict with message, database_deleted count, storage_cleared count
        """
        response = self.session.delete(f"{self.base_url}/api/data/datasets/all")
        response.raise_for_status()
        return response.json()
    
    def map_fields(self, file_id: str, auto_detect: bool = True) -> Dict:
        """Auto-detect and map fields"""
        response = self.session.post(
            f"{self.base_url}/api/data/map-fields",
            json={
                'file_id': file_id,
                'auto_detect': auto_detect
            }
        )
        response.raise_for_status()
        return response.json()
    
    def auto_map_fields(self, dataset_id: str, api_key: Optional[str] = None) -> Dict:
        """
        Auto-detect database source and suggest field mappings using LLM
        
        Args:
            dataset_id: Dataset ID
            api_key: User's XAI API key for LLM mapping
            
        Returns:
            Dict with detected_database, confidence, mappings, unmapped_fields
        """
        import streamlit as st
        
        # Get API key from session if not provided
        if not api_key and hasattr(st, 'session_state') and 'xai_api_key' in st.session_state:
            api_key = st.session_state.xai_api_key
        
        response = self.session.post(
            f"{self.base_url}/api/data/auto-map",
            json={'dataset_id': dataset_id, 'api_key': api_key}
        )
        response.raise_for_status()
        return response.json()
    
    def apply_field_mappings(self, dataset_id: str, mappings: Dict[str, str]) -> Dict:
        """
        Apply custom field mappings to dataset
        
        Args:
            dataset_id: Dataset ID
            mappings: Dictionary mapping original field names to standard field names
            
        Returns:
            Updated dataset info
        """
        response = self.session.post(
            f"{self.base_url}/api/data/map-fields",
            json={
                'dataset_id': dataset_id,
                'mappings': mappings
            }
        )
        response.raise_for_status()
        return response.json()
    
    def merge_datasets(self, dataset_ids: List[str], source_names: Optional[List[str]] = None) -> Dict:
        """
        Merge multiple datasets
        
        Args:
            dataset_ids: List of dataset IDs to merge
            source_names: Optional list of source names for each dataset
            
        Returns:
            Dict with merged_dataset_id, total_records, sources_merged, summary
        """
        payload = {'dataset_ids': dataset_ids}
        if source_names:
            payload['source_names'] = source_names
            
        response = self.session.post(
            f"{self.base_url}/api/data/merge",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def deduplicate(
        self,
        dataset_id: str,
        use_doi: bool = True,
        use_title_similarity: bool = True,
        title_threshold: float = 0.85,
        use_metadata: bool = True
    ) -> Dict:
        """
        Remove duplicates from dataset
        
        Args:
            dataset_id: Dataset ID to deduplicate
            use_doi: Use DOI-based matching
            use_title_similarity: Use title similarity matching
            title_threshold: Similarity threshold for titles (0-1)
            use_metadata: Use metadata-based matching
            
        Returns:
            Dict with deduplication results
        """
        response = self.session.post(
            f"{self.base_url}/api/data/deduplicate",
            json={
                'dataset_id': dataset_id,
                'use_doi': use_doi,
                'use_title_similarity': use_title_similarity,
                'title_threshold': title_threshold,
                'use_metadata': use_metadata
            }
        )
        response.raise_for_status()
        return response.json()
    
    # Alias for backward compatibility
    def deduplicate_dataset(self, *args, **kwargs):
        """Alias for deduplicate method"""
        return self.deduplicate(*args, **kwargs)
    
    def smart_deduplicate(
        self,
        dataset_id: str,
        title_threshold: float = 0.85,
        min_abstract_length: int = 50,
        min_title_length: int = 10
    ) -> Dict:
        """
        智能去重 - Smart deduplication with manual review
        
        Five-step intelligent deduplication:
        1. Quality check - Remove invalid records
        2. DOI exact matching
        3. Title similarity matching
        4. Metadata validation (only for similar titles)
        5. Generate manual review list
        
        Args:
            dataset_id: Dataset ID to deduplicate
            title_threshold: Similarity threshold for titles (0-1)
            min_abstract_length: Minimum abstract length (characters)
            min_title_length: Minimum title length (characters)
            
        Returns:
            Dict with:
                - dataset_id: Updated cleaned dataset
                - review_dataset_id: Dataset with records needing manual review (if any)
                - Detailed statistics and report
        """
        response = self.session.post(
            f"{self.base_url}/api/data/smart-deduplicate",
            json={
                'dataset_id': dataset_id,
                'title_threshold': title_threshold,
                'min_abstract_length': min_abstract_length,
                'min_title_length': min_title_length
            }
        )
        response.raise_for_status()
        return response.json()
    
    def export_dataset(self, dataset_id: str, format: str = "csv") -> bytes:
        """
        Export dataset to file
        
        Args:
            dataset_id: Dataset ID to export
            format: Export format ('csv' or 'excel')
            
        Returns:
            File content as bytes
        """
        response = self.session.get(
            f"{self.base_url}/api/data/export/{dataset_id}",
            params={'format': format}
        )
        response.raise_for_status()
        
        # The endpoint returns JSON with file path, we need to get the actual file
        # For now, let's get the dataset and convert to CSV client-side
        import pandas as pd
        import io
        
        preview = self.get_dataset_preview(dataset_id, limit=999999)  # Get all records
        df = pd.DataFrame(preview['records'])
        
        if format == 'csv':
            return df.to_csv(index=False).encode('utf-8-sig')
        elif format in ['excel', 'xlsx']:
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    # ===== Screening Methods =====
    
    def list_tasks(self, status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """
        List all screening tasks
        
        Args:
            status: Filter by status (COMPLETED, RUNNING, FAILED, etc.)
            limit: Maximum number of tasks to return
        """
        params = {'limit': limit}
        if status:
            params['status'] = status
        response = self.session.get(
            f"{self.base_url}/api/screening/tasks",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def start_screening(
        self,
        data_id: str,
        criteria: Dict,
        model: str = "grok-3",
        num_workers: int = 8,
        confidence_threshold: float = 0.8,
        manual_review_threshold: float = 0.6,
        limit: Optional[int] = None
    ) -> Dict:
        """Start AI screening task"""
        payload = {
            'data_id': data_id,
            'criteria': criteria,
            'model': model,
            'num_workers': num_workers,
            'confidence_threshold': confidence_threshold,
            'manual_review_threshold': manual_review_threshold
        }
        
        # Add limit if provided (for test mode)
        if limit is not None:
            payload['limit'] = limit
        
        response = self.session.post(
            f"{self.base_url}/api/screening/start",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_screening_status(self, task_id: str) -> Dict:
        """Get screening task status (for polling)"""
        response = self.session.get(f"{self.base_url}/api/screening/status/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def get_screening_results(self, task_id: str) -> Dict:
        """Get completed screening results"""
        response = self.session.get(f"{self.base_url}/api/screening/results/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def cancel_screening(self, task_id: str) -> Dict:
        """Cancel running screening task"""
        response = self.session.post(f"{self.base_url}/api/screening/cancel/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def estimate_screening_cost(self, data_id: str, model: str = "grok-3") -> Dict:
        """Estimate screening cost"""
        response = self.session.get(
            f"{self.base_url}/api/screening/estimate-cost/{data_id}",
            params={'model': model}
        )
        response.raise_for_status()
        return response.json()
    
    # ===== Results Methods =====
    
    def get_results_summary(self, task_id: str) -> Dict:
        """Get results summary statistics"""
        response = self.session.get(f"{self.base_url}/api/results/summary/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def get_confidence_distribution(self, task_id: str) -> Dict:
        """Get confidence score distribution"""
        response = self.session.get(f"{self.base_url}/api/results/confidence-distribution/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def get_decision_breakdown(self, task_id: str) -> Dict:
        """Get decision breakdown (include/exclude/manual)"""
        response = self.session.get(f"{self.base_url}/api/results/decision-breakdown/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def generate_prisma_diagram(self, task_id: str) -> bytes:
        """Generate PRISMA diagram"""
        response = self.session.post(f"{self.base_url}/api/results/prisma/{task_id}")
        response.raise_for_status()
        return response.content
    
    def export_results(
        self,
        task_id: str,
        format: str = "excel",
        include_excluded: bool = True,
        include_reasoning: bool = True,
        include_confidence: bool = True
    ) -> bytes:
        """Export results to file"""
        response = self.session.post(
            f"{self.base_url}/api/results/export",
            json={
                'task_id': task_id,
                'format': format,
                'include_excluded': include_excluded,
                'include_reasoning': include_reasoning,
                'include_confidence': include_confidence
            }
        )
        response.raise_for_status()
        return response.content
