"""
Database Models - SQLAlchemy ORM Models

Defines database schema for:
- Datasets (uploaded files and processed data)
- Tasks (screening tasks with progress tracking)
- ScreeningResults (individual article decisions)
- DeduplicationRecords (duplicate tracking and review items)
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


# ===== Enums =====

class TaskStatus(str, enum.Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScreeningDecision(str, enum.Enum):
    """Screening decision enumeration"""
    INCLUDE = "include"
    EXCLUDE = "exclude"
    MANUAL_REVIEW = "manual_review"
    RELEVANT = "relevant"  # Legacy support
    IRRELEVANT = "irrelevant"  # Legacy support
    UNCERTAIN = "uncertain"  # Legacy support


class DatasetStatus(str, enum.Enum):
    """Dataset status"""
    UPLOADED = "uploaded"
    PARSED = "parsed"
    MAPPED = "mapped"
    MERGED = "merged"
    DEDUPLICATED = "deduplicated"
    READY = "ready"


# ===== Models =====

class Dataset(Base):
    """
    Dataset model - stores uploaded and processed datasets
    
    Lifecycle:
    1. UPLOADED - File uploaded, not parsed
    2. PARSED - Columns and data extracted
    3. MAPPED - Fields mapped to standard schema
    4. MERGED - Combined from multiple sources (if applicable)
    5. DEDUPLICATED - Duplicates removed
    6. READY - Ready for screening
    """
    __tablename__ = "datasets"
    
    id = Column(String(36), primary_key=True)  # UUID
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum(DatasetStatus), default=DatasetStatus.UPLOADED)
    
    # File info
    original_filename = Column(String(255), nullable=True)
    file_format = Column(String(50), nullable=True)  # xlsx, csv, ris
    file_size = Column(Integer, nullable=True)  # bytes
    
    # Data info
    total_records = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    columns = Column(JSON, nullable=True)  # List of column names
    
    # Field mapping
    field_mappings = Column(JSON, nullable=True)  # Dict of original -> standard mappings
    detected_database = Column(String(100), nullable=True)  # pubmed, scopus, etc.
    mapping_confidence = Column(Float, nullable=True)
    
    # Deduplication info
    duplicates_removed = Column(Integer, default=0)
    dedup_method = Column(String(100), nullable=True)
    
    # Merge info (if merged from multiple datasets)
    is_merged = Column(Boolean, default=False)
    merged_from = Column(JSON, nullable=True)  # List of dataset IDs
    source_name = Column(String(255), nullable=True)  # Source label for merged data
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100), nullable=True)  # Future: user ID
    
    # Data storage (JSON for now, could move to separate table)
    data_json = Column(Text, nullable=True)  # JSON serialized DataFrame
    
    # Relationships
    tasks = relationship("Task", back_populates="dataset")
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, name={self.name}, records={self.total_records})>"


class Task(Base):
    """
    Task model - tracks async screening tasks
    
    Progress tracking, checkpointing, and result storage
    """
    __tablename__ = "tasks"
    
    id = Column(String(36), primary_key=True)  # UUID
    task_type = Column(String(50), default="screening")  # screening, deduplication, etc.
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    
    # Dataset reference
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    dataset = relationship("Dataset", back_populates="tasks")
    
    # Progress tracking
    total_items = Column(Integer, default=0)
    processed_items = Column(Integer, default=0)
    progress_percent = Column(Float, default=0.0)
    
    # Time tracking
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    estimated_completion = Column(DateTime, nullable=True)
    
    # Configuration
    config = Column(JSON, nullable=True)  # Screening criteria, model settings, etc.
    model_name = Column(String(100), nullable=True)
    num_workers = Column(Integer, default=8)
    
    # Results summary
    included_count = Column(Integer, default=0)
    excluded_count = Column(Integer, default=0)
    manual_review_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    
    # Cost tracking
    total_cost = Column(Float, default=0.0)
    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)
    total_reasoning_tokens = Column(Integer, default=0)
    total_cached_tokens = Column(Integer, default=0)
    
    # Error info
    error_message = Column(Text, nullable=True)
    
    # Checkpoint data
    checkpoint_data = Column(JSON, nullable=True)
    last_checkpoint_at = Column(DateTime, nullable=True)
    
    # Metadata
    created_by = Column(String(100), nullable=True)
    
    # Relationships
    screening_results = relationship("ScreeningResult", back_populates="task", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Task(id={self.id}, status={self.status}, progress={self.progress_percent}%)>"


class ScreeningResult(Base):
    """
    Screening result for individual article
    
    Stores AI decision, confidence, reasoning, and cost for each screened article
    """
    __tablename__ = "screening_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Task reference
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False)
    task = relationship("Task", back_populates="screening_results")
    
    # Article info
    title = Column(Text, nullable=True, default="[No Title]")  # Allow NULL, provide default
    abstract = Column(Text, nullable=True)
    authors = Column(Text, nullable=True)
    journal = Column(String(255), nullable=True)
    year = Column(Integer, nullable=True)
    doi = Column(String(255), nullable=True)
    
    # Screening decision
    decision = Column(Enum(ScreeningDecision), nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    reasoning_content = Column(Text, nullable=True)  # Extended reasoning for reasoning models
    needs_manual_review = Column(Boolean, default=False)
    
    # AI metadata
    model_used = Column(String(100), nullable=True)
    provider_used = Column(String(50), nullable=True)
    
    # Token usage
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    reasoning_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    # Cost
    api_cost = Column(Float, default=0.0)
    
    # Processing time
    processing_time = Column(Float, nullable=True)  # seconds
    screened_at = Column(DateTime, default=datetime.utcnow)
    
    # Manual review (if edited by human)
    manual_decision = Column(Enum(ScreeningDecision), nullable=True)
    manual_notes = Column(Text, nullable=True)
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<ScreeningResult(id={self.id}, decision={self.decision}, confidence={self.confidence})>"


class DeduplicationRecord(Base):
    """
    Deduplication tracking - stores duplicate pairs and manual review items
    
    Used by smart deduplicator to track:
    - Confirmed duplicates (removed)
    - Potential duplicates needing manual review
    """
    __tablename__ = "deduplication_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Dataset reference
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    
    # Duplicate info
    record_1_index = Column(Integer, nullable=False)  # Original index in dataset
    record_2_index = Column(Integer, nullable=False)
    
    title_1 = Column(Text, nullable=True)
    title_2 = Column(Text, nullable=True)
    
    # Similarity metrics
    similarity_score = Column(Float, nullable=True)
    similarity_method = Column(String(50), nullable=True)  # doi, title_tfidf, metadata
    
    # Metadata comparison
    author_1 = Column(Text, nullable=True)
    author_2 = Column(Text, nullable=True)
    journal_1 = Column(String(255), nullable=True)
    journal_2 = Column(String(255), nullable=True)
    year_1 = Column(Integer, nullable=True)
    year_2 = Column(Integer, nullable=True)
    
    # Decision
    is_duplicate = Column(Boolean, default=False)  # True = confirmed duplicate
    needs_manual_review = Column(Boolean, default=False)  # True = uncertain, needs review
    reason = Column(Text, nullable=True)
    
    # Manual review
    manual_decision = Column(Boolean, nullable=True)  # True = keep both, False = remove duplicate
    manual_notes = Column(Text, nullable=True)
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<DeduplicationRecord(id={self.id}, is_duplicate={self.is_duplicate}, needs_review={self.needs_manual_review})>"


class SystemLog(Base):
    """
    System activity log - audit trail for all operations
    
    Tracks uploads, screening, deduplication, exports, etc.
    """
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Log info
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR
    category = Column(String(50), nullable=False)  # upload, screening, dedup, export
    message = Column(Text, nullable=False)
    
    # Context
    dataset_id = Column(String(36), nullable=True)
    task_id = Column(String(36), nullable=True)
    user_id = Column(String(100), nullable=True)
    
    # Request info
    endpoint = Column(String(255), nullable=True)
    method = Column(String(10), nullable=True)  # GET, POST, etc.
    status_code = Column(Integer, nullable=True)
    
    # Additional data (renamed from 'metadata' to avoid SQLAlchemy conflict)
    extra_data = Column(JSON, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level={self.level}, category={self.category})>"
