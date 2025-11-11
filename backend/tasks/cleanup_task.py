"""
Automatic Data Cleanup Task

Periodically removes orphaned/old data from database:
- Datasets older than 24 hours with no associated tasks
- Completed tasks older than 7 days
- Failed tasks older than 24 hours
"""

import logging
from datetime import datetime, timedelta
from sqlalchemy import and_
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)


class DataCleanupTask:
    """Background task for automatic data cleanup"""
    
    def __init__(
        self,
        dataset_retention_hours: int = 24,
        completed_task_retention_days: int = 7,
        failed_task_retention_hours: int = 24,
        check_interval_hours: int = 6
    ):
        """
        Initialize cleanup task
        
        Args:
            dataset_retention_hours: Delete datasets older than this (hours)
            completed_task_retention_days: Delete completed tasks older than this (days)
            failed_task_retention_hours: Delete failed tasks older than this (hours)
            check_interval_hours: Run cleanup every N hours
        """
        self.dataset_retention_hours = dataset_retention_hours
        self.completed_task_retention_days = completed_task_retention_days
        self.failed_task_retention_hours = failed_task_retention_hours
        self.check_interval_hours = check_interval_hours
        self.running = False
        self.thread = None
    
    def start(self):
        """Start background cleanup task"""
        if self.running:
            logger.warning("Cleanup task already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info(f"🧹 Started automatic data cleanup task (runs every {self.check_interval_hours}h)")
    
    def stop(self):
        """Stop background cleanup task"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("🛑 Stopped automatic data cleanup task")
    
    def _run_loop(self):
        """Main loop for periodic cleanup"""
        while self.running:
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"❌ Cleanup task error: {e}")
            
            # Sleep for check_interval_hours
            for _ in range(self.check_interval_hours * 3600):
                if not self.running:
                    break
                time.sleep(1)
    
    def cleanup(self):
        """
        Perform cleanup of old data
        
        Returns:
            Dict with cleanup statistics
        """
        from ..db import get_db
        from ..models.database import Dataset, Task, ScreeningResult
        
        stats = {
            'datasets_deleted': 0,
            'tasks_deleted': 0,
            'screening_results_deleted': 0
        }
        
        with get_db() as db:
            now = datetime.utcnow()
            
            # 1. Clean up old datasets (no associated tasks)
            dataset_cutoff = now - timedelta(hours=self.dataset_retention_hours)
            
            # Find datasets older than cutoff with no tasks
            old_datasets = db.query(Dataset).filter(
                and_(
                    Dataset.created_at < dataset_cutoff,
                    ~Dataset.tasks.any()  # No associated tasks
                )
            ).all()
            
            for dataset in old_datasets:
                db.delete(dataset)
                stats['datasets_deleted'] += 1
            
            # 2. Clean up completed tasks (and their results)
            completed_cutoff = now - timedelta(days=self.completed_task_retention_days)
            
            old_completed_tasks = db.query(Task).filter(
                and_(
                    Task.status == 'COMPLETED',
                    Task.completed_at < completed_cutoff
                )
            ).all()
            
            for task in old_completed_tasks:
                # Delete associated screening results first
                results = db.query(ScreeningResult).filter_by(task_id=task.id).all()
                for result in results:
                    db.delete(result)
                    stats['screening_results_deleted'] += 1
                
                db.delete(task)
                stats['tasks_deleted'] += 1
            
            # 3. Clean up failed tasks
            failed_cutoff = now - timedelta(hours=self.failed_task_retention_hours)
            
            old_failed_tasks = db.query(Task).filter(
                and_(
                    Task.status == 'FAILED',
                    Task.updated_at < failed_cutoff
                )
            ).all()
            
            for task in old_failed_tasks:
                # Delete associated screening results
                results = db.query(ScreeningResult).filter_by(task_id=task.id).all()
                for result in results:
                    db.delete(result)
                    stats['screening_results_deleted'] += 1
                
                db.delete(task)
                stats['tasks_deleted'] += 1
            
            # Commit all deletions
            db.commit()
        
        if any(stats.values()):
            logger.info(f"🧹 Cleanup complete: {stats}")
        
        return stats


# Global cleanup task instance
cleanup_task = DataCleanupTask(
    dataset_retention_hours=24,
    completed_task_retention_days=7,
    failed_task_retention_hours=24,
    check_interval_hours=6
)
