"""
Backend Wrapper for Streamlit Cloud
直接調用後端邏輯，不通過 FastAPI
"""

import uuid
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Import backend modules
from backend.core.data_processor import DataProcessor
from backend.core.data_merger import DataMerger
from backend.core.smart_deduplicator import SmartDeduplicator
from backend.core.llm_field_mapper import LLMFieldMapper
from backend.core.screener import AIScreener
from backend.tasks.task_manager import TaskManager
from backend.services.grok_client import GrokClient
from backend.db import get_db
from backend.models.database import Dataset, Task, ScreeningResult

# Initialize components
task_manager = TaskManager(checkpoint_dir=Path("data/checkpoints"), use_database=True)


class BackendWrapper:
    """Wrapper class to replace API calls with direct function calls"""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.merger = DataMerger()
        self.deduplicator = SmartDeduplicator()
        
    # ===== Data Management Functions =====
    
    def upload_file(self, file_content: bytes, filename: str, file_format: str) -> Dict[str, Any]:
        """Process uploaded file"""
        try:
            # Parse file based on format
            if file_format in ['csv', 'excel']:
                df = self.processor.parse_file(file_content, file_format)
            elif file_format == 'ris':
                df = self.processor.parse_ris_file(file_content)
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            # Save to database
            dataset_id = str(uuid.uuid4())
            with get_db() as db:
                dataset = Dataset(
                    id=dataset_id,
                    name=filename,
                    file_format=file_format,
                    data=df.to_json(orient='records'),
                    status='uploaded',
                    total_records=len(df),
                    is_merged=False
                )
                db.add(dataset)
                db.commit()
            
            return {
                'dataset_id': dataset_id,
                'filename': filename,
                'records': len(df),
                'columns': list(df.columns)
            }
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")
    
    def get_datasets(self, session_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Get all datasets"""
        with get_db() as db:
            datasets = db.query(Dataset).all()
            
            result = []
            for ds in datasets:
                # Determine file_type
                if ds.is_merged:
                    file_type = 'merged'
                elif 'manual_review' in ds.name.lower() or 'review' in ds.name.lower():
                    file_type = 'manual_review'
                elif ds.status == 'DEDUPLICATED':
                    file_type = 'deduplicated'
                else:
                    file_type = ds.file_format
                
                result.append({
                    'dataset_id': ds.id,
                    'name': ds.name,
                    'file_format': ds.file_format,
                    'file_type': file_type,
                    'status': ds.status,
                    'total_records': ds.total_records,
                    'is_merged': ds.is_merged,
                    'created_at': ds.created_at.isoformat() if ds.created_at else None
                })
            
            return result
    
    def map_fields(self, dataset_id: str, user_api_key: str) -> Dict[str, Any]:
        """Use LLM to auto-map fields"""
        try:
            with get_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                if not dataset:
                    raise ValueError("Dataset not found")
                
                df = pd.read_json(dataset.data, orient='records')
            
            # Use LLM to map fields
            mapper = LLMFieldMapper(api_key=user_api_key)
            mapped_df, mapping = mapper.map_fields(df)
            
            # Update dataset
            with get_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                dataset.data = mapped_df.to_json(orient='records')
                dataset.status = 'mapped'
                db.commit()
            
            return {
                'dataset_id': dataset_id,
                'mapping': mapping,
                'preview': mapped_df.head(5).to_dict(orient='records')
            }
        except Exception as e:
            raise Exception(f"Field mapping failed: {str(e)}")
    
    def merge_datasets(self, dataset_ids: List[str]) -> Dict[str, Any]:
        """Merge multiple datasets"""
        try:
            # Load datasets
            dataframes = []
            with get_db() as db:
                for ds_id in dataset_ids:
                    dataset = db.query(Dataset).filter_by(id=ds_id).first()
                    if dataset:
                        df = pd.read_json(dataset.data, orient='records')
                        dataframes.append(df)
            
            # Merge
            merged_df = self.merger.merge_datasets(dataframes)
            
            # Save merged dataset
            merged_id = str(uuid.uuid4())
            with get_db() as db:
                dataset = Dataset(
                    id=merged_id,
                    name=f"Merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    file_format='merged',
                    data=merged_df.to_json(orient='records'),
                    status='merged',
                    total_records=len(merged_df),
                    is_merged=True
                )
                db.add(dataset)
                db.commit()
            
            return {
                'dataset_id': merged_id,
                'total_records': len(merged_df),
                'columns': list(merged_df.columns)
            }
        except Exception as e:
            raise Exception(f"Merge failed: {str(e)}")
    
    def smart_deduplicate(self, dataset_id: str, threshold: float = 0.85) -> Dict[str, Any]:
        """Smart deduplication with metadata validation"""
        try:
            with get_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                if not dataset:
                    raise ValueError("Dataset not found")
                
                df = pd.read_json(dataset.data, orient='records')
            
            # Deduplicate
            clean_df, review_df = self.deduplicator.deduplicate(df, similarity_threshold=threshold)
            
            # Save cleaned dataset
            clean_id = str(uuid.uuid4())
            with get_db() as db:
                clean_dataset = Dataset(
                    id=clean_id,
                    name=f"Cleaned_{dataset.name}",
                    file_format=dataset.file_format,
                    data=clean_df.to_json(orient='records'),
                    status='DEDUPLICATED',
                    total_records=len(clean_df),
                    is_merged=dataset.is_merged
                )
                db.add(clean_dataset)
                
                # Save review dataset if any
                review_id = None
                if len(review_df) > 0:
                    review_id = str(uuid.uuid4())
                    review_dataset = Dataset(
                        id=review_id,
                        name=f"Manual_Review_{dataset.name}",
                        file_format=dataset.file_format,
                        data=review_df.to_json(orient='records'),
                        status='needs_review',
                        total_records=len(review_df),
                        is_merged=dataset.is_merged
                    )
                    db.add(review_dataset)
                
                db.commit()
            
            return {
                'dataset_id': clean_id,
                'review_dataset_id': review_id,
                'original_count': len(df),
                'cleaned_count': len(clean_df),
                'removed_count': len(df) - len(clean_df),
                'review_count': len(review_df)
            }
        except Exception as e:
            raise Exception(f"Deduplication failed: {str(e)}")
    
    # ===== AI Screening Functions =====
    
    def estimate_cost(self, dataset_id: str, model: str = "grok-beta") -> Dict[str, Any]:
        """Estimate screening cost"""
        try:
            with get_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                if not dataset:
                    raise ValueError("Dataset not found")
            
            screener = AIScreener(api_key="dummy")  # Just for cost estimation
            cost_info = screener.estimate_cost(dataset.total_records, model)
            
            return cost_info
        except Exception as e:
            raise Exception(f"Cost estimation failed: {str(e)}")
    
    def start_screening(self, dataset_id: str, criteria: Dict[str, str], 
                       model: str, num_workers: int, user_api_key: str,
                       limit: Optional[int] = None) -> Dict[str, Any]:
        """Start AI screening task"""
        try:
            # Load dataset
            with get_db() as db:
                dataset = db.query(Dataset).filter_by(id=dataset_id).first()
                if not dataset:
                    raise ValueError("Dataset not found")
                
                df = pd.read_json(dataset.data, orient='records')
            
            # Limit for testing
            if limit:
                df = df.head(limit)
            
            # Create screening task
            task_id = task_manager.create_task(
                task_type="screening",
                total_items=len(df),
                metadata={
                    "dataset_id": dataset_id,
                    "criteria": criteria,
                    "model": model,
                    "num_workers": num_workers
                }
            )
            
            # Start screening in background (we'll use st.session_state for this)
            return {
                'task_id': task_id,
                'status': 'PENDING',
                'total_items': len(df),
                'dataset_data': df.to_dict(orient='records'),  # Return data for processing
                'criteria': criteria,
                'model': model,
                'num_workers': num_workers,
                'user_api_key': user_api_key
            }
        except Exception as e:
            raise Exception(f"Failed to start screening: {str(e)}")
    
    def get_screening_status(self, task_id: str) -> Dict[str, Any]:
        """Get screening task status"""
        task = task_manager.get_task(task_id)
        if not task:
            raise ValueError("Task not found")
        
        return {
            'task_id': task.id,
            'status': task.status.value if hasattr(task.status, 'value') else task.status,
            'processed_items': task.processed_items,
            'total_items': task.total_items,
            'progress_percent': (task.processed_items / task.total_items * 100) if task.total_items > 0 else 0,
            'included_count': task.result.get('included', 0) if task.result else 0,
            'excluded_count': task.result.get('excluded', 0) if task.result else 0,
            'manual_review_count': task.result.get('manual_review', 0) if task.result else 0,
            'total_cost': task.result.get('total_cost', 0) if task.result else 0,
            'error': task.error,
            'metadata': task.metadata,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None
        }
    
    def get_screening_results(self, task_id: str) -> Dict[str, Any]:
        """Get screening results"""
        with get_db() as db:
            results = db.query(ScreeningResult).filter_by(task_id=task_id).all()
            
            result_list = []
            for r in results:
                result_list.append({
                    'title': r.title,
                    'abstract': r.abstract,
                    'decision': r.decision,
                    'confidence': r.confidence,
                    'reasoning': r.reasoning
                })
            
            # Get task info
            task = task_manager.get_task(task_id)
            
            return {
                'task_id': task_id,
                'results': result_list,
                'summary': task.result if task else {}
            }
    
    def list_tasks(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List all tasks"""
        with get_db() as db:
            query = db.query(Task)
            if status:
                query = query.filter_by(status=status)
            tasks = query.order_by(Task.created_at.desc()).limit(limit).all()
            
            result = []
            for t in tasks:
                result.append({
                    'task_id': t.id,
                    'status': t.status,
                    'total_items': t.total_items,
                    'processed_items': t.processed_items,
                    'included_count': t.result.get('included', 0) if t.result else 0,
                    'excluded_count': t.result.get('excluded', 0) if t.result else 0,
                    'manual_review_count': t.result.get('manual_review', 0) if t.result else 0,
                    'completed_at': t.completed_at.isoformat() if t.completed_at else None,
                    'created_at': t.created_at.isoformat() if t.created_at else None
                })
            
            return result


# Global instance
backend = BackendWrapper()
