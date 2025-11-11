"""
Task Manager - Async Task Execution with Progress Tracking

Manages long-running screening tasks with:
- Database persistence
- Status tracking
- Progress updates
- Checkpointing
- Cancellation support
- Error handling

Database-backed implementation with fallback to in-memory for backward compatibility
"""

import uuid
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import logging
from pathlib import Path
import json

# Database imports
from ..db import get_db, Task, TaskStatus as DBTaskStatus

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Information about a task"""
    task_id: str
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    # Progress
    total_items: int = 0
    processed_items: int = 0
    progress_percent: float = 0.0
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    # Metadata
    metadata: Dict = field(default_factory=dict)
    # Cancellation
    cancel_requested: bool = False


class TaskManager:
    """
    Manager for async screening tasks
    
    Features:
    - Database persistence (with in-memory fallback)
    - Thread-safe task state
    - Progress tracking
    - Checkpoint support
    - Cancellation handling
    """
    
    def __init__(self, checkpoint_dir: Optional[Path] = None, use_database: bool = True):
        """
        Initialize task manager
        
        Args:
            checkpoint_dir: Optional directory for task checkpoints
            use_database: Whether to use database (default: True)
        """
        self._tasks: Dict[str, TaskInfo] = {}  # Legacy in-memory storage
        self._lock = threading.Lock()
        self.checkpoint_dir = checkpoint_dir
        self.use_database = use_database
        
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _to_db_status(self, status: TaskStatus) -> DBTaskStatus:
        """Convert TaskStatus enum to database enum"""
        return DBTaskStatus(status.value)
    
    def _from_db_status(self, status: DBTaskStatus) -> TaskStatus:
        """Convert database enum to TaskStatus"""
        return TaskStatus(status.value)
    
    def create_task(
        self,
        total_items: int,
        metadata: Optional[Dict] = None,
        dataset_id: Optional[str] = None
    ) -> str:
        """
        Create a new task
        
        Args:
            total_items: Total number of items to process
            metadata: Optional task metadata
            dataset_id: Optional dataset ID for database relation
            
        Returns:
            task_id
        """
        task_id = str(uuid.uuid4())
        
        if self.use_database and dataset_id:
            # Create in database
            try:
                with get_db() as db:
                    db_task = Task(
                        id=task_id,
                        dataset_id=dataset_id,
                        task_type=metadata.get('task_type', 'screening') if metadata else 'screening',
                        status=DBTaskStatus.PENDING,
                        total_items=total_items,
                        config=metadata or {},
                        model_name=metadata.get('model') if metadata else None,
                        num_workers=metadata.get('num_workers', 8) if metadata else 8,
                        created_at=datetime.utcnow()
                    )
                    db.add(db_task)
                    db.commit()
                    logger.info(f"Created task {task_id} in database: {total_items} items")
            except Exception as e:
                logger.error(f"Failed to create task in database: {e}")
                # Fallback to in-memory
                self.use_database = False
        
        # Also create in memory for backward compatibility
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat(),
            total_items=total_items,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._tasks[task_id] = task_info
        
        logger.info(f"Created task {task_id}: {total_items} items")
        return task_id
    
    def start_task(self, task_id: str):
        """Mark task as started"""
        # Update in database
        if self.use_database:
            try:
                with get_db() as db:
                    db_task = db.query(Task).filter_by(id=task_id).first()
                    if db_task:
                        db_task.status = DBTaskStatus.RUNNING
                        db_task.started_at = datetime.utcnow()
                        db.commit()
            except Exception as e:
                logger.error(f"Failed to update task in database: {e}")
        
        # Update in memory
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            self._tasks[task_id].status = TaskStatus.RUNNING
            self._tasks[task_id].started_at = datetime.now().isoformat()
        
        logger.info(f"Started task {task_id}")
    
    def update_progress(
        self,
        task_id: str,
        processed_items: int,
        result_data: Optional[Any] = None
    ):
        """
        Update task progress
        
        Args:
            task_id: Task ID
            processed_items: Number of items processed so far
            result_data: Optional partial result data
        """
        # Update in database
        if self.use_database:
            try:
                with get_db() as db:
                    db_task = db.query(Task).filter_by(id=task_id).first()
                    if db_task:
                        db_task.processed_items = processed_items
                        db_task.progress_percent = (
                            (processed_items / db_task.total_items * 100) 
                            if db_task.total_items > 0 else 0
                        )
                        db.commit()
            except Exception as e:
                logger.error(f"Failed to update task progress in database: {e}")
        
        # Update in memory
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.processed_items = processed_items
            task.progress_percent = (processed_items / task.total_items * 100) if task.total_items > 0 else 0
            
            if result_data is not None:
                task.result = result_data
    
    def complete_task(
        self,
        task_id: str,
        result: Any
    ):
        """
        Mark task as completed
        
        Args:
            task_id: Task ID
            result: Final result data
        """
        # Update in database
        if self.use_database:
            try:
                with get_db() as db:
                    db_task = db.query(Task).filter_by(id=task_id).first()
                    if db_task:
                        db_task.status = DBTaskStatus.COMPLETED
                        db_task.completed_at = datetime.utcnow()
                        db_task.progress_percent = 100.0
                        db_task.processed_items = db_task.total_items
                        
                        # Update result summary if available
                        if hasattr(result, 'relevant'):
                            db_task.included_count = result.relevant
                        if hasattr(result, 'irrelevant'):
                            db_task.excluded_count = result.irrelevant
                        if hasattr(result, 'needs_manual_review'):
                            db_task.manual_review_count = result.needs_manual_review
                        if hasattr(result, 'errors'):
                            db_task.error_count = result.errors
                        if hasattr(result, 'total_cost'):
                            db_task.total_cost = result.total_cost
                        if hasattr(result, 'total_input_tokens'):
                            db_task.total_input_tokens = result.total_input_tokens
                        if hasattr(result, 'total_output_tokens'):
                            db_task.total_output_tokens = result.total_output_tokens
                        if hasattr(result, 'total_reasoning_tokens'):
                            db_task.total_reasoning_tokens = result.total_reasoning_tokens
                        
                        db.commit()
            except Exception as e:
                logger.error(f"Failed to complete task in database: {e}")
        
        # Update in memory
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.result = result
            task.progress_percent = 100.0
        
        logger.info(f"Completed task {task_id}")
    
    def fail_task(
        self,
        task_id: str,
        error: str
    ):
        """
        Mark task as failed
        
        Args:
            task_id: Task ID
            error: Error message
        """
        # Update in database
        if self.use_database:
            try:
                with get_db() as db:
                    db_task = db.query(Task).filter_by(id=task_id).first()
                    if db_task:
                        db_task.status = DBTaskStatus.FAILED
                        db_task.completed_at = datetime.utcnow()
                        db_task.error_message = error
                        db.commit()
            except Exception as e:
                logger.error(f"Failed to update task failure in database: {e}")
        
        # Update in memory
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now().isoformat()
            task.error = error
        
        logger.error(f"Failed task {task_id}: {error}")
    
    def cancel_task(self, task_id: str):
        """
        Request task cancellation
        
        Args:
            task_id: Task ID
        """
        # Update in database
        if self.use_database:
            try:
                with get_db() as db:
                    db_task = db.query(Task).filter_by(id=task_id).first()
                    if db_task and db_task.status == DBTaskStatus.PENDING:
                        # Not started yet, cancel immediately
                        db_task.status = DBTaskStatus.CANCELLED
                        db_task.completed_at = datetime.utcnow()
                        db.commit()
            except Exception as e:
                logger.error(f"Failed to cancel task in database: {e}")
        
        # Update in memory
        with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.cancel_requested = True
            
            if task.status == TaskStatus.PENDING:
                # Not started yet, cancel immediately
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now().isoformat()
        
        logger.info(f"Cancellation requested for task {task_id}")
    
    def is_cancelled(self, task_id: str) -> bool:
        """
        Check if task cancellation was requested
        
        Args:
            task_id: Task ID
            
        Returns:
            True if cancellation requested
        """
        # Check database first
        if self.use_database:
            try:
                with get_db() as db:
                    db_task = db.query(Task).filter_by(id=task_id).first()
                    if db_task:
                        return db_task.status == DBTaskStatus.CANCELLED
            except Exception as e:
                logger.error(f"Failed to check cancellation in database: {e}")
        
        # Fallback to in-memory
        with self._lock:
            if task_id not in self._tasks:
                return False
            return self._tasks[task_id].cancel_requested
    
    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        Get task information from database or memory
        
        Args:
            task_id: Task ID
            
        Returns:
            TaskInfo or None if not found
        """
        # Try database first
        if self.use_database:
            try:
                with get_db() as db:
                    db_task = db.query(Task).filter_by(id=task_id).first()
                    if db_task:
                        # Convert database task to TaskInfo
                        return TaskInfo(
                            task_id=db_task.id,
                            status=self._from_db_status(db_task.status),
                            created_at=db_task.created_at.isoformat(),
                            started_at=db_task.started_at.isoformat() if db_task.started_at else None,
                            completed_at=db_task.completed_at.isoformat() if db_task.completed_at else None,
                            total_items=db_task.total_items,
                            processed_items=db_task.processed_items,
                            progress_percent=db_task.progress_percent,
                            result=None,  # Result stored separately
                            error=db_task.error_message,
                            metadata=db_task.config or {},
                            cancel_requested=False  # Would need separate flag
                        )
            except Exception as e:
                logger.error(f"Failed to get task from database: {e}")
        
        # Fallback to in-memory
        with self._lock:
            return self._tasks.get(task_id)
    
    def list_tasks(self) -> Dict[str, TaskInfo]:
        """
        List all tasks from database and memory
        
        Returns:
            Dict of task_id -> TaskInfo
        """
        all_tasks = {}
        
        # Get from database first
        if self.use_database:
            try:
                with get_db() as db:
                    db_tasks = db.query(Task).all()
                    for db_task in db_tasks:
                        all_tasks[db_task.id] = TaskInfo(
                            task_id=db_task.id,
                            status=self._from_db_status(db_task.status),
                            created_at=db_task.created_at.isoformat(),
                            started_at=db_task.started_at.isoformat() if db_task.started_at else None,
                            completed_at=db_task.completed_at.isoformat() if db_task.completed_at else None,
                            total_items=db_task.total_items,
                            processed_items=db_task.processed_items,
                            progress_percent=db_task.progress_percent,
                            result=None,
                            error=db_task.error_message,
                            metadata=db_task.config or {},
                            cancel_requested=db_task.status == DBTaskStatus.CANCELLED
                        )
            except Exception as e:
                logger.error(f"Failed to list tasks from database: {e}")
        
        # Merge with in-memory tasks (in-memory takes precedence)
        with self._lock:
            for task_id, task in self._tasks.items():
                all_tasks[task_id] = task
        
        return all_tasks
    
    def save_checkpoint(
        self,
        task_id: str,
        checkpoint_data: Dict
    ):
        """
        Save task checkpoint to database and file
        
        Args:
            task_id: Task ID
            checkpoint_data: Checkpoint data to save
        """
        # Save to database first
        if self.use_database:
            try:
                with get_db() as db:
                    db_task = db.query(Task).filter_by(id=task_id).first()
                    if db_task:
                        db_task.checkpoint_data = checkpoint_data
                        db.commit()
                        logger.info(f"💾 Saved checkpoint to database for task {task_id}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint to database: {e}")
        
        # Also save to file (backward compatibility)
        if not self.checkpoint_dir:
            logger.warning("No checkpoint directory configured")
            return
        
        checkpoint_file = self.checkpoint_dir / f"task_{task_id}.json"
        
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning(f"Task {task_id} not found in memory for checkpoint")
                # Still save the checkpoint data
                checkpoint = {
                    'task_id': task_id,
                    'timestamp': datetime.now().isoformat(),
                    'data': checkpoint_data
                }
            else:
                checkpoint = {
                    'task_id': task_id,
                    'status': task.status.value,
                    'processed_items': task.processed_items,
                    'total_items': task.total_items,
                    'timestamp': datetime.now().isoformat(),
                    'data': checkpoint_data
                }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"💾 Saved checkpoint file for task {task_id}")
    
    def load_checkpoint(self, task_id: str) -> Optional[Dict]:
        """
        Load task checkpoint from database or file
        
        Args:
            task_id: Task ID
            
        Returns:
            Checkpoint data or None if not found
        """
        # Try database first
        if self.use_database:
            try:
                with get_db() as db:
                    db_task = db.query(Task).filter_by(id=task_id).first()
                    if db_task and db_task.checkpoint_data:
                        logger.info(f"📂 Loaded checkpoint from database for task {task_id}")
                        return db_task.checkpoint_data
            except Exception as e:
                logger.error(f"Failed to load checkpoint from database: {e}")
        
        # Fall back to file
        if not self.checkpoint_dir:
            return None
        
        checkpoint_file = self.checkpoint_dir / f"task_{task_id}.json"
        
        if not checkpoint_file.exists():
            return None
        
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        
        logger.info(f"📂 Loaded checkpoint from file for task {task_id}")
        return checkpoint
    
    def cleanup_task(self, task_id: str):
        """
        Remove task from manager and optionally database
        
        Args:
            task_id: Task ID
        """
        # Remove from database (optional - usually want to keep for audit trail)
        # Uncomment if you want to delete from database:
        # if self.use_database:
        #     try:
        #         with get_db() as db:
        #             db_task = db.query(Task).filter_by(id=task_id).first()
        #             if db_task:
        #                 db.delete(db_task)
        #                 db.commit()
        #                 logger.info(f"🗑️  Deleted task {task_id} from database")
        #     except Exception as e:
        #         logger.error(f"Failed to delete task from database: {e}")
        
        # Remove from in-memory
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
        
        # Clean up checkpoint file
        if self.checkpoint_dir:
            checkpoint_file = self.checkpoint_dir / f"task_{task_id}.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
        
        logger.info(f"🗑️  Cleaned up task {task_id} from memory and checkpoint files")


# Global task manager instance
task_manager = TaskManager(use_database=True)
