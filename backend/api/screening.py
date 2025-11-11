"""
AI Screening API Endpoints

Handles async article screening with progress tracking and cost monitoring
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
from enum import Enum
import pandas as pd
import logging
import uuid
from datetime import datetime
from pathlib import Path

from ..core.screener import AIScreener, ScreeningConfig, ScreeningResult, ArticleDecision
from ..services.grok_client import GrokClient
from ..tasks.task_manager import task_manager, TaskStatus

# Database imports
from ..db import get_db_session
from ..models.database import Task, ScreeningResult as DBScreeningResult, ScreeningDecision as DBScreeningDecision, TaskStatus as DBTaskStatus
from ..api.data_management import get_dataset_dataframe

logger = logging.getLogger(__name__)

router = APIRouter()

# Use global task manager instance (configured in backend/tasks/task_manager.py)

# Legacy screening tasks storage (for backward compatibility during migration)
screening_tasks: Dict[str, Dict] = {}

# Dataset storage reference (from data_management API)
from ..api.data_management import datasets_storage


# ===== Enums =====

class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScreeningDecision(str, Enum):
    """Screening decision enumeration"""
    INCLUDE = "include"
    EXCLUDE = "exclude"
    MANUAL_REVIEW = "manual_review"


# ===== Request/Response Models =====

class ScreeningCriteria(BaseModel):
    """PCC criteria for screening"""
    population: str
    concept: str
    context: str
    inclusion_criteria: Optional[List[str]] = None
    exclusion_criteria: Optional[List[str]] = None


class StartScreeningRequest(BaseModel):
    """Request model to start screening task"""
    data_id: str  # ID of deduplicated dataset
    criteria: ScreeningCriteria
    model: str = "grok-3"  # 'grok-3', 'grok-3-mini-fast', 'grok-4'
    num_workers: int = 8
    confidence_threshold: float = 0.8
    manual_review_threshold: float = 0.6
    limit: Optional[int] = None  # Optional limit for test mode
    api_key: Optional[str] = None  # User's X.AI API key


class ScreeningProgress(BaseModel):
    """Progress information for screening task"""
    task_id: str
    status: TaskStatus
    total_articles: int
    processed: int
    included: int
    excluded: int
    manual_review: int
    errors: int
    progress_percent: float
    estimated_time_remaining: Optional[float] = None  # seconds
    current_cost: float = 0.0
    estimated_total_cost: float = 0.0
    elapsed_time: float = 0.0  # seconds since start
    current_cost: float
    estimated_total_cost: float


class ArticleScreeningResult(BaseModel):
    """Individual article screening result"""
    article_id: str
    title: str
    abstract: Optional[str] = None
    decision: ScreeningDecision
    confidence: float
    reasoning: str
    cost: float
    processing_time: float  # seconds


class ScreeningResults(BaseModel):
    """Complete screening results"""
    task_id: str
    status: TaskStatus
    total_articles: int
    results: List[ArticleScreeningResult]
    summary: Dict
    total_cost: float
    total_time: float
    # Token usage details
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cached_tokens: int = 0  # NEW: Track cached tokens
    avg_tokens_per_article: float = 0.0
    # Cost breakdown
    input_cost: float = 0.0
    cached_cost: float = 0.0
    output_cost: float = 0.0


# ===== API Endpoints =====

@router.get("/tasks", response_model=List[Dict])
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db_session)
):
    """
    List all screening tasks (most recent first)
    
    Args:
        status: Filter by status (COMPLETED, RUNNING, FAILED, etc.)
        limit: Maximum number of tasks to return (default: 50)
    """
    query = db.query(Task)
    
    # Filter by status if provided
    if status:
        try:
            task_status = DBTaskStatus[status.upper()]
            query = query.filter(Task.status == task_status)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    # Order by most recent first
    tasks = query.order_by(Task.created_at.desc()).limit(limit).all()
    
    # Convert to response format
    task_list = []
    for task in tasks:
        task_list.append({
            'task_id': task.id,
            'dataset_id': task.dataset_id,
            'status': task.status.value,
            'total_items': task.total_items,
            'processed_items': task.processed_items,
            'created_at': task.created_at.isoformat() if task.created_at else None,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'total_cost': task.total_cost or 0.0,
            'included_count': task.included_count or 0,
            'excluded_count': task.excluded_count or 0,
            'manual_review_count': task.manual_review_count or 0
        })
    
    logger.info(f"📋 Listed {len(task_list)} tasks")
    return task_list


@router.post("/start", response_model=Dict)
async def start_screening(
    request: StartScreeningRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session)
):
    """
    Start a new AI screening task (async)
    
    Returns task_id for polling progress
    """
    logger.info(f"🚀 Starting screening task for dataset: {request.data_id}")
    logger.info(f"⚙️  Model: {request.model}, Workers: {request.num_workers}")
    
    # Get dataset (database or legacy storage)
    df = get_dataset_dataframe(request.data_id, db)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Dataset {request.data_id} not found")
    
    # Log original dataset size
    logger.info(f"📊 Original dataset size: {len(df)} articles")
    logger.info(f"📊 Request limit parameter: {request.limit}")
    
    # Apply limit if provided (test mode)
    if request.limit is not None and request.limit > 0:
        original_count = len(df)
        df = df.head(request.limit).copy()  # Use .copy() to avoid SettingWithCopyWarning
        logger.info(f"🧪 Test mode: Limited to {len(df)} articles (out of {original_count})")
        logger.info(f"🧪 Test mode active: will screen {len(df)} articles")
    else:
        logger.info(f"📊 No limit applied, will screen all {len(df)} articles")
    
    # Validate required columns
    if 'title' not in df.columns or 'abstract' not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Dataset must have 'title' and 'abstract' columns"
        )
    
    # Create screening config
    inclusion_criteria = request.criteria.inclusion_criteria or [
        request.criteria.population,
        request.criteria.concept,
        request.criteria.context
    ]
    exclusion_criteria = request.criteria.exclusion_criteria or []
    
    config = ScreeningConfig(
        inclusion_criteria=inclusion_criteria,
        exclusion_criteria=exclusion_criteria,
        confidence_threshold=request.confidence_threshold,
        manual_review_threshold=request.manual_review_threshold,
        max_workers=request.num_workers,
        checkpoint_interval=100
    )
    
    # Create task (now saves to database with dataset_id)
    task_id = task_manager.create_task(
        total_items=len(df),
        metadata={
            'dataset_id': request.data_id,
            'model': request.model,
            'num_workers': request.num_workers,
            'criteria': request.criteria.dict()
        },
        dataset_id=request.data_id  # Link to dataset
    )
    
    # Store screening task info in legacy dict (backward compatibility)
    screening_tasks[task_id] = {
        'dataset_id': request.data_id,
        'config': config,
        'model': request.model,
        'start_time': datetime.now().isoformat(),
        'current_cost': 0.0,
        'included': 0,
        'excluded': 0,
        'manual_review': 0,
        'errors': 0
    }
    
    # Submit background task
    background_tasks.add_task(
        _run_screening_task,
        task_id=task_id,
        df=df,
        config=config,
        model=request.model,
        api_key=request.api_key
    )
    
    logger.info(f"✅ Created screening task: {task_id}")
    
    # Prepare response message
    message = f'Screening task created with {request.num_workers} workers'
    if request.limit is not None and request.limit > 0:
        message = f'🧪 Test mode: Screening {len(df)} articles with {request.num_workers} workers'
    
    return {
        'task_id': task_id,
        'status': 'pending',
        'total_articles': len(df),
        'message': message
    }


@router.get("/status/{task_id}", response_model=ScreeningProgress)
async def get_screening_status(task_id: str, db: Session = Depends(get_db_session)):
    """
    Poll screening task progress
    
    Frontend should call this every 2 seconds to update UI
    """
    # Get task from manager (queries database first)
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Get screening-specific data from legacy dict (backward compatibility)
    screening_data = screening_tasks.get(task_id, {})
    
    # Calculate elapsed time
    elapsed_time = 0.0
    if task.started_at:
        # Handle both datetime objects (from database) and ISO strings (from in-memory)
        if isinstance(task.started_at, str):
            start_time = datetime.fromisoformat(task.started_at.replace('Z', '+00:00'))
        else:
            start_time = task.started_at
        
        # Ensure both datetimes are naive (remove timezone info for comparison)
        if start_time.tzinfo is not None:
            start_time = start_time.replace(tzinfo=None)
        
        current_time = datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()
    
    # Calculate estimated time remaining
    estimated_time_remaining = None
    if task.processed_items > 0 and task.status == TaskStatus.RUNNING and elapsed_time > 0:
        rate = task.processed_items / elapsed_time
        remaining_items = task.total_items - task.processed_items
        estimated_time_remaining = remaining_items / rate if rate > 0 else None
    
    # Get counts from database Task model (preferred) or fall back to legacy dict
    included = 0
    excluded = 0
    manual_review = 0
    errors = 0
    current_cost = 0.0
    
    # Try database first (Task model has these fields after migration)
    try:
        db_task = db.query(Task).filter_by(id=task_id).first()
        if db_task:
            included = db_task.included_count or 0
            excluded = db_task.excluded_count or 0
            manual_review = db_task.manual_review_count or 0
            errors = db_task.error_count or 0
            current_cost = db_task.total_cost or 0.0
        else:
            # Fall back to legacy dict
            included = screening_data.get('included', 0)
            excluded = screening_data.get('excluded', 0)
            manual_review = screening_data.get('manual_review', 0)
            errors = screening_data.get('errors', 0)
            current_cost = screening_data.get('current_cost', 0.0)
    except Exception as e:
        logger.warning(f"⚠️ Failed to get counts from database: {e}, using legacy dict")
        included = screening_data.get('included', 0)
        excluded = screening_data.get('excluded', 0)
        manual_review = screening_data.get('manual_review', 0)
        errors = screening_data.get('errors', 0)
        current_cost = screening_data.get('current_cost', 0.0)
    
    # Calculate cost estimates
    avg_cost = current_cost / task.processed_items if task.processed_items > 0 else 0
    estimated_total_cost = avg_cost * task.total_items
    
    # If task is completed, get from final result
    result = task.result if hasattr(task, 'result') and task.result else None
    if result and isinstance(result, ScreeningResult):
        included = result.relevant
        excluded = result.irrelevant
        manual_review = result.needs_manual_review
        errors = result.errors
    
    return ScreeningProgress(
        task_id=task_id,
        status=task.status,
        total_articles=task.total_items,
        processed=task.processed_items,
        included=included,
        excluded=excluded,
        manual_review=manual_review,
        errors=errors,
        progress_percent=task.progress_percent,
        estimated_time_remaining=estimated_time_remaining,
        current_cost=current_cost,
        estimated_total_cost=estimated_total_cost,
        elapsed_time=elapsed_time
    )


@router.get("/results/{task_id}", response_model=ScreeningResults)
async def get_screening_results(task_id: str, db: Session = Depends(get_db_session)):
    """
    Retrieve screening results (only works if task completed)
    """
    # Try to get results from database first
    try:
        db_task = db.query(Task).filter_by(id=task_id).first()
        if db_task and db_task.status == DBTaskStatus.COMPLETED:
            # Get all screening results for this task
            db_results = db.query(DBScreeningResult).filter_by(task_id=task_id).all()
            
            if db_results:
                logger.info(f"📊 Retrieved {len(db_results)} screening results from database for task {task_id}")
                
                # Convert database results to API response
                article_results = []
                total_confidence = 0.0
                for db_result in db_results:
                    # Debug: Check if abstract exists
                    if db_result.abstract is None:
                        logger.warning(f"⚠️ No abstract for article: {db_result.title[:50]}")
                    
                    article_results.append(
                        ArticleScreeningResult(
                            article_id=str(hash(db_result.title)) if db_result.title else str(db_result.id),
                            title=db_result.title[:200] if db_result.title else "",
                            abstract=db_result.abstract,
                            decision=_map_db_decision_to_api(db_result.decision),
                            confidence=db_result.confidence or 0.0,
                            reasoning=db_result.reasoning or "",
                            cost=db_result.api_cost or 0.0,
                            processing_time=0.0
                        )
                    )
                    total_confidence += (db_result.confidence or 0.0)
                
                # Build summary
                # Safely get metadata dict (handle SQLAlchemy JSON type)
                metadata_dict = {}
                if db_task.metadata:
                    try:
                        # If metadata is already a dict
                        if isinstance(db_task.metadata, dict):
                            metadata_dict = db_task.metadata
                        # If metadata is a string (JSON), parse it
                        elif isinstance(db_task.metadata, str):
                            import json
                            metadata_dict = json.loads(db_task.metadata)
                    except Exception as meta_err:
                        logger.warning(f"⚠️ Failed to parse metadata: {meta_err}")
                        metadata_dict = {}
                
                summary = {
                    'total_articles': db_task.processed_items,
                    'relevant': db_task.included_count or 0,
                    'irrelevant': db_task.excluded_count or 0,
                    'uncertain': db_task.manual_review_count or 0,
                    'errors': db_task.error_count or 0,
                    'needs_manual_review': db_task.manual_review_count or 0,
                    'avg_confidence': total_confidence / len(db_results) if db_results else 0.0,
                    'model_used': metadata_dict.get('model', 'unknown'),
                    'provider_used': 'grok'
                }
                
                # Calculate elapsed time
                elapsed_time = 0.0
                if db_task.completed_at and db_task.started_at:
                    elapsed_time = (db_task.completed_at - db_task.started_at).total_seconds()
                
                return ScreeningResults(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    total_articles=db_task.processed_items,
                    results=article_results,
                    summary=summary,
                    total_cost=db_task.total_cost or 0.0,
                    total_time=elapsed_time,
                    total_input_tokens=db_task.total_input_tokens or 0,
                    total_output_tokens=db_task.total_output_tokens or 0,
                    total_reasoning_tokens=db_task.total_reasoning_tokens or 0,
                    total_cached_tokens=0,
                    avg_tokens_per_article=((db_task.total_input_tokens or 0) + (db_task.total_output_tokens or 0)) / db_task.processed_items if db_task.processed_items > 0 else 0,
                    input_cost=(db_task.total_cost or 0.0) * 0.3,
                    cached_cost=(db_task.total_cost or 0.0) * 0.1,
                    output_cost=(db_task.total_cost or 0.0) * 0.6
                )
    except Exception as e:
        logger.warning(f"⚠️ Failed to get results from database: {e}, falling back to file or in-memory")
    
    # Fall back to in-memory or file-based results
    task = task_manager.get_task(task_id)
    
    if task and task.status == TaskStatus.COMPLETED and task.result:
        # Use in-memory result
        result: ScreeningResult = task.result
    else:
        # Try to load from file
        results_file = Path("data/screening_results") / f"{task_id}.json"
        if not results_file.exists():
            raise HTTPException(status_code=404, detail=f"Results not found for task {task_id}")
        
        import json
        with open(results_file, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Convert to response format directly from file
        article_results = []
        for d in results_data.get('decisions', []):
            article_results.append(
                ArticleScreeningResult(
                    article_id=str(hash(d['title'])),
                    title=d['title'][:200],
                    abstract=d.get('abstract'),  # Include abstract from JSON
                    decision=_map_decision(d['decision']),
                    confidence=d['confidence'],
                    reasoning=d['reason'],
                    cost=d['api_cost'],
                    processing_time=0.0
                )
            )
        
        summary = {
            'total_articles': results_data['total_articles'],
            'relevant': results_data['relevant'],
            'irrelevant': results_data['irrelevant'],
            'uncertain': results_data['uncertain'],
            'errors': results_data['errors'],
            'needs_manual_review': results_data['needs_manual_review'],
            'avg_confidence': sum(d['confidence'] for d in results_data['decisions']) / len(results_data['decisions']) if results_data['decisions'] else 0,
            'model_used': results_data['model_used'],
            'provider_used': results_data['provider_used']
        }
        
        return ScreeningResults(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            total_articles=results_data['total_articles'],
            results=article_results,
            summary=summary,
            total_cost=results_data['total_cost'],
            total_time=results_data['total_time'],
            total_input_tokens=results_data['total_input_tokens'],
            total_output_tokens=results_data['total_output_tokens'],
            total_reasoning_tokens=results_data['total_reasoning_tokens'],
            total_cached_tokens=0,
            avg_tokens_per_article=(results_data['total_input_tokens'] + results_data['total_output_tokens']) / results_data['total_articles'] if results_data['total_articles'] > 0 else 0,
            input_cost=results_data['total_cost'] * 0.3,
            cached_cost=results_data['total_cost'] * 0.1,
            output_cost=results_data['total_cost'] * 0.6
        )
    
    # Original in-memory path
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed yet. Current status: {task.status}"
        )
    
    # Get result
    result: ScreeningResult = task.result
    
    if not result:
        raise HTTPException(status_code=500, detail="No results available")
    
    # Convert ArticleDecisions to API response format
    article_results = []
    for decision in result.decisions:
        article_results.append(
            ArticleScreeningResult(
                article_id=str(hash(decision.title)),  # Simple ID generation
                title=decision.title[:200],  # Truncate for response
                decision=_map_decision(decision.decision),
                confidence=decision.confidence,
                reasoning=decision.reason,
                cost=decision.api_cost,
                processing_time=0.0  # Would need to track this separately
            )
        )
    
    # Build summary
    summary = {
        'total_articles': result.total_articles,
        'relevant': result.relevant,
        'irrelevant': result.irrelevant,
        'uncertain': result.uncertain,
        'errors': result.errors,
        'needs_manual_review': result.needs_manual_review,
        'avg_confidence': sum(d.confidence for d in result.decisions) / len(result.decisions) if result.decisions else 0,
        'model_used': result.model_used,
        'provider_used': result.provider_used
    }
    
    return ScreeningResults(
        task_id=task_id,
        status=task.status,
        total_articles=result.total_articles,
        results=article_results,
        summary=summary,
        total_cost=result.total_cost,
        total_time=result.total_time,
        # Token usage details
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
        total_reasoning_tokens=result.total_reasoning_tokens,
        total_cached_tokens=0,  # Will be calculated from individual decisions
        avg_tokens_per_article=(result.total_input_tokens + result.total_output_tokens) / result.total_articles if result.total_articles > 0 else 0,
        # Cost breakdown (approximate based on pricing)
        input_cost=result.total_cost * 0.3,  # Rough estimate, should be calculated properly
        cached_cost=result.total_cost * 0.1,  # Rough estimate
        output_cost=result.total_cost * 0.6   # Rough estimate
    )


@router.post("/cancel/{task_id}")
async def cancel_screening(task_id: str):
    """
    Cancel a running screening task
    """
    logger.info(f"🛑 Cancelling task: {task_id}")
    
    try:
        task_manager.cancel_task(task_id)
        return {'message': f'Cancellation requested for task {task_id}'}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/checkpoint/{task_id}")
async def get_checkpoint(task_id: str):
    """
    Get checkpoint data for resuming interrupted screening
    """
    checkpoint = task_manager.load_checkpoint(task_id)
    
    if not checkpoint:
        raise HTTPException(status_code=404, detail=f"No checkpoint found for task {task_id}")
    
    return checkpoint


@router.post("/resume/{task_id}")
async def resume_screening(task_id: str, background_tasks: BackgroundTasks):
    """
    Resume an interrupted screening task from checkpoint
    """
    logger.info(f"▶️  Resuming task: {task_id}")
    
    # Load checkpoint
    checkpoint = task_manager.load_checkpoint(task_id)
    
    if not checkpoint:
        raise HTTPException(status_code=404, detail=f"No checkpoint found for task {task_id}")
    
    # Get original task info
    screening_data = screening_tasks.get(task_id)
    
    if not screening_data:
        raise HTTPException(status_code=404, detail=f"Screening task data not found for {task_id}")
    
    # Get dataset
    dataset_id = screening_data['dataset_id']
    if dataset_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    
    df = datasets_storage[dataset_id]['dataframe']
    
    # Submit resume task
    background_tasks.add_task(
        _run_screening_task,
        task_id=task_id,
        df=df,
        config=screening_data['config'],
        model=screening_data['model'],
        resume_from=checkpoint['processed_items']
    )
    
    return {
        'message': f'Resuming task {task_id} from item {checkpoint["processed_items"]}',
        'task_id': task_id
    }


@router.get("/estimate-cost/{data_id}")
async def estimate_screening_cost(
    data_id: str,
    model: str = "grok-4-fast-reasoning"
):
    """
    Estimate total cost before starting screening
    """
    # Validate dataset
    if data_id not in datasets_storage:
        raise HTTPException(status_code=404, detail=f"Dataset {data_id} not found")
    
    df = datasets_storage[data_id]['dataframe']
    
    # Estimate average tokens per article
    # Rough estimate: ~200 tokens for title+abstract input, ~100 tokens output
    avg_input_tokens = 200
    avg_output_tokens = 100
    
    # Initialize Grok client for cost estimation
    try:
        grok_client = GrokClient(model=model)
        cost_per_article = grok_client.estimate_cost(
            input_tokens=avg_input_tokens,
            output_tokens=avg_output_tokens
        )
        
        total_articles = len(df)
        total_estimated_cost = cost_per_article['total_cost'] * total_articles
        
        return {
            'dataset_id': data_id,
            'total_articles': total_articles,
            'model': model,
            'cost_per_article': cost_per_article,
            'estimated_total_cost': total_estimated_cost,
            'currency': 'HKD',
            'note': 'This is a rough estimate based on average article length'
        }
        
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")


# ===== Background Task Functions =====

def _run_screening_task(
    task_id: str,
    df: pd.DataFrame,
    config: ScreeningConfig,
    model: str,
    api_key: Optional[str] = None,
    resume_from: int = 0
):
    """
    Background task to run screening
    
    Args:
        task_id: Task ID
        df: Dataset DataFrame
        config: Screening configuration
        model: Model name
        api_key: User's X.AI API key
        resume_from: Resume from this item index
    """
    try:
        # Mark task as running
        task_manager.start_task(task_id)
        
        # Initialize Grok client with user's API key
        grok_client = GrokClient(model=model, api_key=api_key)
        
        # Initialize screener
        screener = AIScreener(
            llm_client=grok_client,
            config=config,
            checkpoint_dir=Path("data/checkpoints")
        )
        
        # Prepare articles (skip if resuming)
        articles = [
            (row['title'], row['abstract'])
            for _, row in df.iterrows()
        ]
        
        if resume_from > 0:
            articles = articles[resume_from:]
            logger.info(f"Resuming from article {resume_from}")
        
        # Progress callback
        def on_progress(completed, total, decision):
            actual_completed = completed + resume_from
            
            # Update task progress
            task_manager.update_progress(task_id, actual_completed)
            
            # Save individual screening result to database
            try:
                from ..db import get_db
                with get_db() as db:
                    # Map decision to database enum
                    db_decision = _map_api_decision_to_db(decision.decision)
                    
                    # Create ScreeningResult record (no article_id field in model)
                    db_result = DBScreeningResult(
                        task_id=task_id,
                        title=decision.title,
                        abstract=decision.abstract,
                        decision=db_decision,
                        confidence=decision.confidence,
                        reasoning=decision.reason,
                        needs_manual_review=decision.needs_manual_review,
                        input_tokens=decision.input_tokens,
                        output_tokens=decision.output_tokens,
                        reasoning_tokens=decision.reasoning_tokens,
                        total_tokens=decision.total_tokens,
                        api_cost=decision.api_cost
                    )
                    db.add(db_result)
                    db.commit()
                    
                    # Update Task model counts
                    db_task = db.query(Task).filter_by(id=task_id).first()
                    if db_task:
                        if decision.decision == 'Relevant':
                            db_task.included_count = (db_task.included_count or 0) + 1
                        elif decision.decision == 'Irrelevant':
                            db_task.excluded_count = (db_task.excluded_count or 0) + 1
                        elif decision.decision == 'Uncertain' or decision.needs_manual_review:
                            db_task.manual_review_count = (db_task.manual_review_count or 0) + 1
                        
                        if decision.decision == 'Error':
                            db_task.error_count = (db_task.error_count or 0) + 1
                        
                        # Update cost tracking
                        db_task.total_cost = (db_task.total_cost or 0.0) + decision.api_cost
                        db_task.total_input_tokens = (db_task.total_input_tokens or 0) + decision.input_tokens
                        db_task.total_output_tokens = (db_task.total_output_tokens or 0) + decision.output_tokens
                        db_task.total_reasoning_tokens = (db_task.total_reasoning_tokens or 0) + decision.reasoning_tokens
                        
                        db.commit()
            except Exception as e:
                logger.warning(f"⚠️ Failed to save screening result to database: {e}")
            
            # Update cost and decision tracking in legacy dict (backward compatibility)
            if task_id in screening_tasks:
                current_cost = screening_tasks[task_id].get('current_cost', 0.0)
                screening_tasks[task_id]['current_cost'] = current_cost + decision.api_cost
                
                # Update decision counters
                if decision.decision == 'Relevant':
                    screening_tasks[task_id]['included'] = screening_tasks[task_id].get('included', 0) + 1
                elif decision.decision == 'Irrelevant':
                    screening_tasks[task_id]['excluded'] = screening_tasks[task_id].get('excluded', 0) + 1
                elif decision.decision == 'Uncertain' or decision.needs_manual_review:
                    screening_tasks[task_id]['manual_review'] = screening_tasks[task_id].get('manual_review', 0) + 1
                
                if decision.decision == 'Error':
                    screening_tasks[task_id]['errors'] = screening_tasks[task_id].get('errors', 0) + 1
            
            # Save checkpoint every 100 articles
            if actual_completed % config.checkpoint_interval == 0:
                task_manager.save_checkpoint(task_id, {
                    'processed_items': actual_completed,
                    'current_cost': screening_tasks[task_id].get('current_cost', 0.0) if task_id in screening_tasks else 0.0
                })
            
            # Check for cancellation
            if task_manager.is_cancelled(task_id):
                logger.warning(f"🛑 Task {task_id} cancelled - requesting screener to stop")
                screener.request_cancel()
                raise InterruptedError("Task cancelled by user")
        
        # Run screening
        result = screener.screen_parallel(articles, progress_callback=on_progress)
        
        # Save complete results to file
        results_dir = Path("data/screening_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{task_id}.json"
        
        import json
        results_data = {
            'task_id': task_id,
            'status': 'completed',
            'total_articles': result.total_articles,
            'screened_count': result.screened_count,
            'relevant': result.relevant,
            'irrelevant': result.irrelevant,
            'uncertain': result.uncertain,
            'errors': result.errors,
            'needs_manual_review': result.needs_manual_review,
            'total_time': result.total_time,
            'total_cost': result.total_cost,
            'total_input_tokens': result.total_input_tokens,
            'total_output_tokens': result.total_output_tokens,
            'total_reasoning_tokens': result.total_reasoning_tokens,
            'model_used': result.model_used,
            'provider_used': result.provider_used,
            'decisions': [
                {
                    'title': d.title,
                    'abstract': d.abstract,
                    'decision': d.decision,
                    'reason': d.reason,
                    'confidence': d.confidence,
                    'needs_manual_review': d.needs_manual_review,
                    'timestamp': d.timestamp,
                    'input_tokens': d.input_tokens,
                    'output_tokens': d.output_tokens,
                    'reasoning_tokens': d.reasoning_tokens,
                    'total_tokens': d.total_tokens,
                    'api_cost': d.api_cost
                }
                for d in result.decisions
            ]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved results to {results_file}")
        
        # Mark task as completed
        task_manager.complete_task(task_id, result)
        
        logger.info(f"✅ Screening task {task_id} completed successfully")
        
    except InterruptedError as e:
        logger.warning(f"Task {task_id} cancelled: {e}")
        task_manager.cancel_task(task_id)
        
    except Exception as e:
        logger.error(f"Screening task {task_id} failed: {e}")
        task_manager.fail_task(task_id, str(e))


def _map_decision(decision: str) -> ScreeningDecision:
    """Map screener decision to API enum"""
    if decision == "Relevant":
        return ScreeningDecision.INCLUDE
    elif decision == "Irrelevant":
        return ScreeningDecision.EXCLUDE
    else:
        return ScreeningDecision.MANUAL_REVIEW


def _map_db_decision_to_api(db_decision: DBScreeningDecision) -> ScreeningDecision:
    """Map database ScreeningDecision enum to API ScreeningDecision enum"""
    if db_decision == DBScreeningDecision.INCLUDE:
        return ScreeningDecision.INCLUDE
    elif db_decision == DBScreeningDecision.EXCLUDE:
        return ScreeningDecision.EXCLUDE
    else:
        return ScreeningDecision.MANUAL_REVIEW


def _map_api_decision_to_db(api_decision: str) -> DBScreeningDecision:
    """Map API decision string to database ScreeningDecision enum"""
    if api_decision == "Relevant":
        return DBScreeningDecision.INCLUDE
    elif api_decision == "Irrelevant":
        return DBScreeningDecision.EXCLUDE
    else:
        return DBScreeningDecision.MANUAL_REVIEW
