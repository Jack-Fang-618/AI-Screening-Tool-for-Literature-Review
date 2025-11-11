"""
Data Management API Endpoints

Handles file upload, parsing, field mapping, merging, and deduplication
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging
import uuid
from datetime import datetime

from ..core.data_processor import DataProcessor
from ..core.field_mapper import FieldMapper
from ..core.llm_field_mapper import LLMFieldMapper
from ..core.data_merger import DataMerger
from ..core.deduplicator import Deduplicator
from ..core.smart_deduplicator import SmartDeduplicator, DeduplicationResult

# Database imports
from ..db import get_db_session, Dataset, DatasetStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Legacy in-memory storage (for backward compatibility during migration)
# TODO: Remove once all endpoints are migrated to database
datasets_storage: Dict[str, Dict] = {}

# Initialize processors
data_processor = DataProcessor()
field_mapper = FieldMapper()
# LLM Field Mapper: Created on-demand with user's API key (not initialized here)
data_merger = DataMerger(field_mapper=field_mapper)
deduplicator = Deduplicator()  # Legacy deduplicator (keep for compatibility)
smart_deduplicator = SmartDeduplicator()  # NEW: Intelligent deduplicator with manual review


# Helper to create LLM mapper with user's API key
def get_llm_field_mapper(api_key: Optional[str] = None) -> Optional[LLMFieldMapper]:
    """
    Create LLM Field Mapper with user's API key
    
    Args:
        api_key: User's XAI API key (from session/request)
        
    Returns:
        LLMFieldMapper instance or None if no API key
    """
    if api_key:
        return LLMFieldMapper(api_key=api_key)
    
    # Try to get from environment (for local development)
    import os
    env_key = os.getenv('XAI_API_KEY')
    if env_key:
        return LLMFieldMapper(api_key=env_key)
    
    return None


# ===== Helper Functions =====

def get_dataset_dataframe(dataset_id: str, db: Session) -> pd.DataFrame:
    """
    Get DataFrame for dataset from database or legacy storage
    
    Args:
        dataset_id: Dataset ID
        db: Database session
        
    Returns:
        pandas DataFrame
        
    Raises:
        HTTPException: If dataset not found
    """
    # Try database first
    dataset = db.query(Dataset).filter_by(id=dataset_id).first()
    
    if dataset:
        # Load from database
        return pd.read_json(dataset.data_json, orient='records')
    
    # Fallback to legacy storage
    if dataset_id in datasets_storage:
        return datasets_storage[dataset_id]['dataframe']
    
    # Not found
    raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")


def update_dataset_dataframe(dataset_id: str, df: pd.DataFrame, db: Session):
    """
    Update DataFrame for dataset in database and legacy storage
    
    Args:
        dataset_id: Dataset ID
        df: Updated DataFrame
        db: Database session
    """
    # Update in database
    dataset = db.query(Dataset).filter_by(id=dataset_id).first()
    
    if dataset:
        dataset.data_json = df.to_json(orient='records')
        dataset.total_records = len(df)
        dataset.column_count = len(df.columns)
        dataset.columns = df.columns.tolist()
        dataset.updated_at = datetime.utcnow()
        db.commit()
    
    # Also update legacy storage if exists
    if dataset_id in datasets_storage:
        datasets_storage[dataset_id]['dataframe'] = df


def df_to_json_safe(df: pd.DataFrame) -> List[Dict]:
    """
    Convert DataFrame to JSON-safe list of dicts
    
    Handles:
    - NaN/inf values → empty strings
    - Timestamp objects → ISO format strings
    - Other non-serializable types → strings
    """
    # Replace inf/-inf with NaN, then NaN with empty string
    df_clean = df.replace([np.inf, -np.inf], np.nan).fillna('')
    
    # Convert Timestamp columns to strings
    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].astype(str)
        elif df_clean[col].dtype == 'object':
            # Convert any remaining objects to strings to be safe
            df_clean[col] = df_clean[col].apply(
                lambda x: str(x) if not isinstance(x, (str, int, float, bool)) else x
            )
    
    return df_clean.to_dict('records')


# ===== Request/Response Models =====

class DatasetInfo(BaseModel):
    """Dataset information model"""
    dataset_id: str
    filename: str
    upload_time: str
    record_count: int
    columns: List[str]
    file_type: str


class ParseFileResponse(BaseModel):
    """Response model for file parsing"""
    dataset_id: str
    record_count: int
    columns: List[str]
    preview: List[Dict]  # First 5 records
    file_info: Dict


class FieldMappingRequest(BaseModel):
    """Request model for applying custom field mappings"""
    dataset_id: str
    mappings: Dict[str, str]  # original_field -> standard_field


class AutoMapRequest(BaseModel):
    """Request model for auto field mapping"""
    dataset_id: str
    api_key: Optional[str] = None  # User's XAI API key


class FieldMappingResponse(BaseModel):
    """Response model for field mapping"""
    dataset_id: str
    detected_database: str
    confidence: float
    mappings: List[Dict]
    unmapped_fields: List[str]


class MergeRequest(BaseModel):
    """Request model for merging datasets"""
    dataset_ids: List[str]
    source_names: Optional[List[str]] = None
    api_key: Optional[str] = None  # User's API key for LLM field mapping


class MergeResponse(BaseModel):
    """Response model for merge operation"""
    merged_dataset_id: str
    total_records: int
    sources_merged: List[str]
    summary: str


class DeduplicationRequest(BaseModel):
    """Request model for deduplication"""
    dataset_id: str
    title_threshold: float = 0.85
    use_doi: bool = True
    use_title_similarity: bool = True
    use_metadata: bool = True


class DeduplicationResponse(BaseModel):
    """Response model for deduplication"""
    dataset_id: str
    original_count: int
    duplicate_count: int
    final_count: int
    strategies_used: List[str]
    report: str


class SmartDeduplicationRequest(BaseModel):
    """Request model for smart deduplication with manual review"""
    dataset_id: str
    title_threshold: float = 0.85
    min_abstract_length: int = 50
    min_title_length: int = 10


class SmartDeduplicationResponse(BaseModel):
    """Response model for smart deduplication"""
    dataset_id: str
    review_dataset_id: Optional[str] = None  # ID of the manual review dataset
    original_count: int
    invalid_count: int
    after_quality_check: int
    doi_duplicates: int
    title_duplicates_confirmed: int
    title_duplicates_need_review: int
    final_count: int
    strategies_used: List[str]
    report: str


# ===== API Endpoints =====

@router.post("/upload", response_model=DatasetInfo)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db_session)):
    """
    Upload and parse a file (Excel, CSV, or RIS)
    
    Supports:
    - Excel (.xlsx, .xls)
    - CSV (.csv)
    - RIS (.ris, .txt)
    
    Returns dataset info with unique ID for further operations
    """
    try:
        logger.info(f"📤 Uploading file: {file.filename}")
        
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Create temporary file
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            # Save uploaded file
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Parse file
        df = data_processor.parse_file(tmp_path)
        
        # Get file info
        file_info = data_processor.get_file_info(tmp_path)
        
        # Create dataset in database
        dataset = Dataset(
            id=dataset_id,
            name=file.filename,
            description=f"Uploaded from {file.filename}",
            status=DatasetStatus.PARSED,
            original_filename=file.filename,
            file_format=file_info['file_type'],
            file_size=file_info.get('file_size', 0),
            total_records=len(df),
            column_count=len(df.columns),
            columns=df.columns.tolist(),
            data_json=df.to_json(orient='records'),  # Store DataFrame as JSON
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        # Also store in legacy dict for backward compatibility
        datasets_storage[dataset_id] = {
            'dataframe': df,
            'filename': file.filename,
            'upload_time': dataset.created_at.isoformat(),
            'file_type': file_info['file_type'],
            'original_path': tmp_path
        }
        
        logger.info(f"✅ Dataset {dataset_id} uploaded: {len(df)} records")
        
        return DatasetInfo(
            dataset_id=dataset_id,
            filename=file.filename,
            upload_time=dataset.created_at.isoformat(),
            record_count=len(df),
            columns=df.columns.tolist(),
            file_type=file_info['file_type']
        )
        
    except Exception as e:
        logger.error(f"❌ File upload failed: {e}")
        db.rollback()
        raise HTTPException(status_code=400, detail=f"File upload failed: {str(e)}")
    finally:
        await file.close()


@router.get("/parse/{dataset_id}", response_model=ParseFileResponse)
async def parse_dataset(dataset_id: str, db: Session = Depends(get_db_session)):
    """
    Get parsed dataset info and preview
    
    Returns column info and first 5 records
    """
    # Try database first
    dataset = db.query(Dataset).filter_by(id=dataset_id).first()
    
    if not dataset:
        # Fallback to legacy storage
        if dataset_id not in datasets_storage:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        dataset_dict = datasets_storage[dataset_id]
        df = dataset_dict['dataframe']
        
        # Get preview (first 5 records) with NaN handling
        preview = df_to_json_safe(df.head(5))
        
        # Get file info
        file_info = {
            'filename': dataset_dict['filename'],
            'upload_time': dataset_dict['upload_time'],
            'file_type': dataset_dict['file_type']
        }
    else:
        # Load DataFrame from database
        df = pd.read_json(dataset.data_json, orient='records')
        
        # Get preview
        preview = df_to_json_safe(df.head(5))
        
        # Get file info
        file_info = {
            'filename': dataset.original_filename or dataset.name,
            'upload_time': dataset.created_at.isoformat(),
            'file_type': dataset.file_format or 'unknown'
        }
    
    return ParseFileResponse(
        dataset_id=dataset_id,
        record_count=len(df),
        columns=df.columns.tolist(),
        preview=preview,
        file_info=file_info
    )


@router.post("/auto-map", response_model=FieldMappingResponse)
async def auto_map_fields(request: AutoMapRequest, db: Session = Depends(get_db_session)):
    """
    Auto-detect database source and map fields to standard schema using LLM
    
    Uses Grok AI to intelligently analyze column names and sample data,
    then maps to standardized schema for systematic reviews.
    
    Returns suggested mappings with confidence scores and reasoning
    """
    # Get dataset using helper function
    df = get_dataset_dataframe(request.dataset_id, db)
    
    # Get dataset info
    dataset = db.query(Dataset).filter_by(id=request.dataset_id).first()
    if dataset:
        filename = dataset.original_filename or dataset.name
    elif request.dataset_id in datasets_storage:
        filename = datasets_storage[request.dataset_id]['filename']
    else:
        filename = "unknown"
    
    columns = df.columns.tolist()
    
    logger.info(f"🗺️  LLM-based auto-mapping for dataset {request.dataset_id}")
    logger.info(f"📋 Columns to analyze: {columns}")
    
    # Get sample data to help LLM understand content (use helper for safe JSON conversion)
    sample_rows = min(3, len(df))
    sample_data = df_to_json_safe(df.head(sample_rows))
    
    logger.info(f"📊 Sending {sample_rows} sample rows to LLM for content analysis")
    
    # Get LLM mapper with user's API key
    llm_mapper = get_llm_field_mapper(request.api_key)
    if not llm_mapper:
        raise HTTPException(
            status_code=400,
            detail="XAI API key required for LLM-based field mapping. Please provide api_key in request."
        )
    
    # Use LLM to analyze and map fields based on both column names AND data content
    mapping_result = llm_mapper.analyze_and_map_fields(
        columns=columns,
        sample_data=sample_data,
        filename=filename
    )
    
    # Update dataset with mapping info
    if dataset:
        dataset.detected_database = mapping_result.get('detected_database', 'unknown')
        dataset.mapping_confidence = mapping_result.get('confidence', 0.0)
        db.commit()
    
    # Convert to response format
    detected_db = mapping_result.get('detected_database', 'unknown')
    confidence = mapping_result.get('confidence', 0.0)
    mappings = mapping_result.get('mappings', [])
    unmapped = mapping_result.get('unmapped_fields', [])
    
    logger.info(
        f"✅ LLM detected {detected_db} "
        f"(confidence: {confidence:.2f}, {len(mappings)} mappings)"
    )
    
    return FieldMappingResponse(
        dataset_id=request.dataset_id,
        detected_database=detected_db,
        confidence=confidence,
        mappings=mappings,
        unmapped_fields=unmapped
    )


@router.post("/map-fields", response_model=DatasetInfo)
async def apply_field_mappings(request: FieldMappingRequest, db: Session = Depends(get_db_session)):
    """
    Apply custom field mappings to dataset
    
    Renames columns according to provided mappings
    Updates dataset in-place
    """
    # Get dataset
    df = get_dataset_dataframe(request.dataset_id, db)
    
    # Get dataset info
    dataset = db.query(Dataset).filter_by(id=request.dataset_id).first()
    if not dataset and request.dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")
    
    logger.info(f"📝 Applying {len(request.mappings)} field mappings to {request.dataset_id}")
    
    # Apply mappings
    mapped_df = df.rename(columns=request.mappings)
    
    # Update dataset
    update_dataset_dataframe(request.dataset_id, mapped_df, db)
    
    # Update status and mapping info in database
    if dataset:
        dataset.field_mappings = request.mappings
        dataset.status = DatasetStatus.MAPPED
        db.commit()
        db.refresh(dataset)
    
    logger.info(f"✅ Field mappings applied successfully")
    
    # Get updated info
    if dataset:
        filename = dataset.original_filename or dataset.name
        upload_time = dataset.created_at.isoformat()
        file_type = dataset.file_format or 'unknown'
    else:
        dataset_dict = datasets_storage[request.dataset_id]
        filename = dataset_dict['filename']
        upload_time = dataset_dict['upload_time']
        file_type = dataset_dict['file_type']
    
    return DatasetInfo(
        dataset_id=request.dataset_id,
        filename=filename,
        upload_time=upload_time,
        record_count=len(mapped_df),
        columns=mapped_df.columns.tolist(),
        file_type=file_type
    )


@router.post("/merge", response_model=MergeResponse)
async def merge_datasets(request: MergeRequest, db: Session = Depends(get_db_session)):
    """
    Merge multiple datasets with field mapping
    
    Creates new merged dataset with:
    - Standardized field names
    - Source tracking
    - Combined records
    
    Returns unique ID for merged dataset
    """
    # Check if API key is provided for LLM field mapping
    if not request.api_key:
        raise HTTPException(
            status_code=400,
            detail="API key is required for intelligent field mapping during merge"
        )
    
    # Create LLM field mapper with user's API key
    llm_mapper = get_llm_field_mapper(request.api_key)
    if not llm_mapper:
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize LLM field mapper"
        )
    
    # Create a new data merger instance with the LLM mapper
    merger = DataMerger(field_mapper=field_mapper)
    merger.llm_field_mapper = llm_mapper
    
    # Try database first, then legacy storage
    dataframes = []
    source_names_list = []
    
    for idx, dataset_id in enumerate(request.dataset_ids):
        # Try database
        db_dataset = db.query(Dataset).filter_by(id=dataset_id).first()
        if db_dataset:
            df = pd.read_json(db_dataset.data_json)
            dataframes.append(df)
            source_name = request.source_names[idx] if request.source_names else db_dataset.name
            source_names_list.append(source_name)
        # Try legacy storage
        elif dataset_id in datasets_storage:
            df = datasets_storage[dataset_id]['dataframe']
            dataframes.append(df)
            source_name = request.source_names[idx] if request.source_names else datasets_storage[dataset_id]['filename']
            source_names_list.append(source_name)
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {dataset_id} not found in database or storage"
            )
    
    logger.info(f"🔗 Merging {len(dataframes)} datasets: {', '.join(source_names_list)}")
    
    # Perform merge
    merge_result = merger.merge_datasets(
        dataframes,
        source_names=source_names_list
    )
    
    # Store merged dataset IN DATABASE
    merged_id = str(uuid.uuid4())
    merged_filename = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Create database record
    merged_json = merge_result.merged_dataframe.to_json(orient='records')
    db_dataset = Dataset(
        id=merged_id,
        name=merged_filename,
        data_json=merged_json,
        file_size=len(merged_json),
        total_records=len(merge_result.merged_dataframe),
        column_count=len(merge_result.merged_dataframe.columns),
        columns=merge_result.merged_dataframe.columns.tolist(),
        status=DatasetStatus.MERGED,
        is_merged=True,
        merged_from=request.dataset_ids,
        created_at=datetime.utcnow()
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    
    logger.info(f"✅ Merge complete: {merge_result.total_records} records saved to database as {merged_id}")
    
    return MergeResponse(
        merged_dataset_id=merged_id,
        total_records=merge_result.total_records,
        sources_merged=merge_result.sources_merged,
        summary=merge_result.merge_summary
    )


@router.post("/deduplicate", response_model=DeduplicationResponse)
async def deduplicate_dataset(request: DeduplicationRequest):
    """
    Remove duplicates using multiple strategies:
    - DOI exact matching
    - TF-IDF title similarity (configurable threshold)
    - Author-Year-Journal metadata matching
    
    Updates dataset in-place and returns detailed report
    """
    if request.dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")
    
    dataset = datasets_storage[request.dataset_id]
    df = dataset['dataframe']
    
    logger.info(f"🔍 Deduplicating dataset {request.dataset_id}")
    
    # Create deduplicator with custom settings
    dedup = Deduplicator(
        title_threshold=request.title_threshold,
        use_doi=request.use_doi,
        use_title_similarity=request.use_title_similarity,
        use_metadata=request.use_metadata
    )
    
    # Run deduplication
    result = dedup.find_duplicates(df)
    
    # Update stored dataset
    datasets_storage[request.dataset_id]['dataframe'] = result.cleaned_dataframe
    
    # Generate report
    report = dedup.generate_report(result)
    
    logger.info(
        f"✅ Deduplication complete: {result.original_count} → {result.final_count} "
        f"({result.duplicate_count} removed)"
    )
    
    return DeduplicationResponse(
        dataset_id=request.dataset_id,
        original_count=result.original_count,
        duplicate_count=result.duplicate_count,
        final_count=result.final_count,
        strategies_used=result.strategies_used,
        report=report
    )


@router.post("/smart-deduplicate", response_model=SmartDeduplicationResponse)
async def smart_deduplicate_dataset(request: SmartDeduplicationRequest, db: Session = Depends(get_db_session)):
    """
    智能去重 - Smart Deduplication with Manual Review
    
    五步流程:
    1. 数据质量检查 - 移除无标题/无摘要/摘要过短的记录
    2. DOI精确匹配 - 最可靠的去重方式
    3. 标题相似度匹配 - 识别可能重复的文章
    4. 元数据验证 - 对标题相似的文章检查作者/期刊/年份
       - 元数据一致 → 确认重复，删除
       - 元数据不一致 → 标记为"需要人工审核"
    5. 生成人工审核列表 - 供用户最终决策
    
    Returns detailed report + manual review dataset (if any)
    """
    # Get dataset from database or legacy storage
    df = get_dataset_dataframe(request.dataset_id, db)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")
    
    logger.info(f"🧠 Smart deduplication for dataset {request.dataset_id}")
    
    # Create smart deduplicator with custom settings
    smart_dedup = SmartDeduplicator(
        title_threshold=request.title_threshold,
        min_abstract_length=request.min_abstract_length,
        min_title_length=request.min_title_length
    )
    
    # Run smart deduplication
    result = smart_dedup.deduplicate(df)
    
    # Update dataset in DATABASE (create new cleaned version)
    cleaned_id = str(uuid.uuid4())
    cleaned_filename = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Calculate total duplicates removed
    total_duplicates = result.doi_duplicates + result.title_duplicates_confirmed
    
    cleaned_json = result.cleaned_dataframe.to_json(orient='records')
    db_cleaned = Dataset(
        id=cleaned_id,
        name=cleaned_filename,
        data_json=cleaned_json,
        file_size=len(cleaned_json),
        total_records=len(result.cleaned_dataframe),
        column_count=len(result.cleaned_dataframe.columns),
        columns=result.cleaned_dataframe.columns.tolist(),
        status=DatasetStatus.DEDUPLICATED,
        duplicates_removed=total_duplicates,
        dedup_method='smart_deduplication',
        created_at=datetime.utcnow()
    )
    db.add(db_cleaned)
    
    # Store manual review dataset if exists
    review_dataset_id = None
    if result.review_dataframe is not None and len(result.review_dataframe) > 0:
        review_dataset_id = str(uuid.uuid4())
        review_filename = f"manual_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        review_json = result.review_dataframe.to_json(orient='records')
        db_review = Dataset(
            id=review_dataset_id,
            name=review_filename,
            data_json=review_json,
            file_size=len(review_json),
            total_records=len(result.review_dataframe),
            column_count=len(result.review_dataframe.columns),
            columns=result.review_dataframe.columns.tolist(),
            status=DatasetStatus.READY,  # Ready for manual review
            created_at=datetime.utcnow()
        )
        db.add(db_review)
        logger.info(f"📋 Created manual review dataset: {review_dataset_id} ({len(result.review_dataframe)} pairs)")
    
    db.commit()
    logger.info(f"✅ Saved cleaned dataset to database: {cleaned_id}")
    
    # Generate detailed report
    report = smart_dedup.generate_report(result)
    
    logger.info(
        f"✅ Smart deduplication complete: {result.original_count} → {result.final_count} "
        f"({result.original_count - result.final_count} removed, "
        f"{result.title_duplicates_need_review} need review)"
    )
    
    return SmartDeduplicationResponse(
        dataset_id=cleaned_id,  # Return the NEW cleaned dataset ID
        review_dataset_id=review_dataset_id,
        original_count=result.original_count,
        invalid_count=result.invalid_count,
        after_quality_check=result.after_quality_check,
        doi_duplicates=result.doi_duplicates,
        title_duplicates_confirmed=result.title_duplicates_confirmed,
        title_duplicates_need_review=result.title_duplicates_need_review,
        final_count=result.final_count,
        strategies_used=result.strategies_used,
        report=report
    )


@router.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets(db: Session = Depends(get_db_session)):
    """
    List all uploaded datasets
    
    Returns info for all datasets in storage (database + legacy)
    """
    dataset_list = []
    
    # Get from database
    db_datasets = db.query(Dataset).all()
    for dataset in db_datasets:
        dataset_list.append(
            DatasetInfo(
                dataset_id=dataset.id,
                filename=dataset.original_filename or dataset.name,
                upload_time=dataset.created_at.isoformat(),
                record_count=dataset.total_records,
                columns=dataset.columns or [],
                file_type=dataset.file_format or 'unknown'
            )
        )
    
    # Add from legacy storage (if not already in database)
    db_ids = {d.id for d in db_datasets}
    for dataset_id, dataset in datasets_storage.items():
        if dataset_id not in db_ids:
            df = dataset['dataframe']
            dataset_list.append(
                DatasetInfo(
                    dataset_id=dataset_id,
                    filename=dataset['filename'],
                    upload_time=dataset['upload_time'],
                    record_count=len(df),
                    columns=df.columns.tolist(),
                    file_type=dataset['file_type']
                )
            )
    
    return dataset_list


@router.get("/preview/{dataset_id}")
async def preview_data(dataset_id: str, limit: int = 20, db: Session = Depends(get_db_session)):
    """
    Get preview of dataset (first N rows)
    
    Returns JSON records for display
    """
    # Get dataset from database or legacy storage
    dataset = db.query(Dataset).filter_by(id=dataset_id).first()
    
    if dataset:
        # Load from database
        df = pd.read_json(dataset.data_json, orient='records')
        filename = dataset.original_filename or dataset.name
    elif dataset_id in datasets_storage:
        # Load from legacy storage
        df = datasets_storage[dataset_id]['dataframe']
        filename = datasets_storage[dataset_id]['filename']
    else:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    # Convert to records and handle NaN/inf values for JSON serialization
    preview = df_to_json_safe(df.head(limit))
    
    return {
        'dataset_id': dataset_id,
        'filename': filename,
        'total_records': len(df),
        'preview_count': len(preview),
        'records': preview
    }


@router.get("/columns/{dataset_id}")
async def get_columns(dataset_id: str):
    """
    Get column names for a dataset
    
    Returns list of column names
    """
    if dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    df = datasets_storage[dataset_id]['dataframe']
    
    return {
        'dataset_id': dataset_id,
        'columns': df.columns.tolist(),
        'column_count': len(df.columns)
    }


@router.get("/export/{dataset_id}")
async def export_dataset(dataset_id: str, format: str = "csv"):
    """
    Export dataset to file
    
    Supports CSV, Excel formats
    Returns file content as downloadable response
    """
    if dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    df = datasets_storage[dataset_id]['dataframe']
    
    # Normalize format
    if format in ["excel", "xlsx"]:
        file_format = "xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif format == "csv":
        file_format = "csv"
        media_type = "text/csv"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as tmp_file:
        if file_format == "csv":
            df.to_csv(tmp_file.name, index=False, encoding='utf-8-sig')
        elif file_format == "xlsx":
            df.to_excel(tmp_file.name, index=False, engine='openpyxl')
        
        tmp_path = tmp_file.name
    
    # Read file content
    try:
        with open(tmp_path, 'rb') as f:
            content = f.read()
        
        # Clean up temporary file
        Path(tmp_path).unlink()
        
        # Return file as download
        from fastapi.responses import Response
        filename = f"{datasets_storage[dataset_id]['filename'].rsplit('.', 1)[0]}_export.{file_format}"
        
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        # Clean up on error
        if Path(tmp_path).exists():
            Path(tmp_path).unlink()
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str, db: Session = Depends(get_db_session)):
    """
    Delete dataset from storage
    
    Removes dataset from database and cleans up temporary files
    """
    # Try to delete from database
    dataset = db.query(Dataset).filter_by(id=dataset_id).first()
    
    if dataset:
        db.delete(dataset)
        db.commit()
        logger.info(f"🗑️  Deleted dataset {dataset_id} from database")
    
    # Also remove from legacy storage if exists
    if dataset_id in datasets_storage:
        dataset_dict = datasets_storage[dataset_id]
        
        # Clean up temporary file if exists
        if dataset_dict.get('original_path') and Path(dataset_dict['original_path']).exists():
            try:
                Path(dataset_dict['original_path']).unlink()
                logger.info(f"🗑️  Deleted temporary file: {dataset_dict['original_path']}")
            except Exception as e:
                logger.warning(f"⚠️  Could not delete temporary file: {e}")
        
        # Remove from storage
        del datasets_storage[dataset_id]
    
    if not dataset and dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    logger.info(f"✅ Dataset {dataset_id} deleted successfully")
    
    return {"message": f"Dataset {dataset_id} deleted successfully"}


@router.delete("/datasets/all")
async def delete_all_datasets(db: Session = Depends(get_db_session)):
    """
    Delete ALL datasets from database and storage
    
    WARNING: This removes all uploaded datasets permanently!
    Also deletes all related tasks and screening results.
    """
    deleted_count = 0
    
    # Delete from database
    try:
        # First, delete all related data that references datasets
        from backend.models.database import Task, ScreeningResult
        
        # Delete all screening results
        results = db.query(ScreeningResult).all()
        for result in results:
            db.delete(result)
        logger.info(f"🗑️  Deleted {len(results)} screening results")
        
        # Delete all tasks
        tasks = db.query(Task).all()
        for task in tasks:
            db.delete(task)
        logger.info(f"🗑️  Deleted {len(tasks)} tasks")
        
        # Now delete all datasets
        db_datasets = db.query(Dataset).all()
        for dataset in db_datasets:
            db.delete(dataset)
            deleted_count += 1
        
        db.commit()
        logger.info(f"🗑️  Deleted {deleted_count} datasets from database")
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to delete datasets from database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete datasets: {str(e)}")
    
    # Clear legacy storage
    legacy_count = len(datasets_storage)
    datasets_storage.clear()
    logger.info(f"🗑️  Cleared {legacy_count} datasets from legacy storage")
    
    total_deleted = max(deleted_count, legacy_count)
    
    return {
        "message": f"Deleted {total_deleted} datasets successfully",
        "database_deleted": deleted_count,
        "storage_cleared": legacy_count,
        "tasks_deleted": len(tasks) if 'tasks' in locals() else 0,
        "results_deleted": len(results) if 'results' in locals() else 0
    }
