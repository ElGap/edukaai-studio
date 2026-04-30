"""
Database models for EdukaAI Studio.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, 
    String, Text, JSON, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker, Session
from sqlalchemy.sql import func

class Base(DeclarativeBase):
    pass


def generate_uuid():
    """Generate a unique UUID string."""
    return str(uuid.uuid4())


class Dataset(Base):
    """Training/validation dataset."""
    __tablename__ = "datasets"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    format = Column(String(50), nullable=False)  # 'alpaca', 'sharegpt', 'custom'
    file_path = Column(String(500), nullable=False)
    size_bytes = Column(Integer, nullable=False, default=0)
    num_samples = Column(Integer, nullable=False, default=0)  # Valid samples
    total_raw_samples = Column(Integer, nullable=False, default=0)
    
    # Validation report
    validation_report = Column(JSON, nullable=False, default=dict)
    dataset_schema = Column("schema", JSON, nullable=True)
    preview_samples = Column(JSON, nullable=False, default=list)
    
    # Metadata
    is_validation_set = Column(Boolean, default=False, index=True)
    parent_dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True, index=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    training_runs = relationship("TrainingRun", foreign_keys="TrainingRun.training_dataset_id", back_populates="training_dataset")


class ModelRegistry(Base):
    """Curated base models registry."""
    __tablename__ = "base_models"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    huggingface_id = Column(String(255), nullable=False, unique=True)
    name = Column(String(255), nullable=False)
    architecture = Column(String(50), nullable=True)  # 'qwen2', 'llama', 'phi3'
    parameter_count = Column(Integer, nullable=True)
    context_length = Column(Integer, nullable=True, default=4096)
    
    # MLX-specific config
    mlx_config = Column(JSON, nullable=False, default=dict)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_curated = Column(Boolean, default=True)
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class TrainingPreset(Base):
    """Training configuration presets."""
    __tablename__ = "training_presets"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)  # 'Quick', 'Balanced', 'Maximum'
    description = Column(Text, nullable=True)
    is_default = Column(Boolean, default=False)
    
    # Training parameters
    steps = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    lora_rank = Column(Integer, nullable=False)
    lora_alpha = Column(Integer, nullable=False)
    lora_dropout = Column(Float, nullable=False, default=0.05)
    batch_size = Column(Integer, nullable=False)
    warmup_steps = Column(Integer, nullable=False)
    gradient_accumulation_steps = Column(Integer, nullable=False, default=1)
    early_stopping_patience = Column(Integer, nullable=False, default=0)
    
    # Advanced parameters
    weight_decay = Column(Float, nullable=True)
    max_gradient_norm = Column(Float, nullable=True)
    gradient_checkpointing = Column(Boolean, nullable=False, default=False)
    num_lora_layers = Column(Integer, nullable=False, default=16)
    prompt_masking = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class TrainingRun(Base):
    """Training run with complete isolation."""
    __tablename__ = "training_runs"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="pending", index=True)  # pending, running, paused, completed, failed
    
    # Dataset configuration
    training_dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    validation_dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True)

    # Base model
    base_model_id = Column(String(36), ForeignKey("base_models.id"), nullable=False, index=True)
    
    # Training configuration
    preset_id = Column(String(36), ForeignKey("training_presets.id"), nullable=True)
    
    # Hyperparameters (stored explicitly for reproducibility)
    steps = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    lora_rank = Column(Integer, nullable=False)
    lora_alpha = Column(Integer, nullable=False)
    lora_dropout = Column(Float, nullable=False)
    batch_size = Column(Integer, nullable=False)
    warmup_steps = Column(Integer, nullable=False)
    gradient_accumulation_steps = Column(Integer, nullable=False, default=1)
    early_stopping_patience = Column(Integer, nullable=False, default=0)
    max_seq_length = Column(Integer, nullable=False, default=2048)

    # Advanced parameters
    weight_decay = Column(Float, nullable=True)
    max_gradient_norm = Column(Float, nullable=True)
    gradient_checkpointing = Column(Boolean, default=False)
    num_lora_layers = Column(Integer, nullable=False, default=16)
    prompt_masking = Column(Boolean, default=True)
    validation_split_percent = Column(Integer, nullable=False, default=10)  # 0 = disabled, 5, 10, or 15%
    
    # Resource limits
    cpu_cores_limit = Column(Integer, nullable=True)
    gpu_memory_limit_gb = Column(Float, nullable=True)
    ram_limit_gb = Column(Float, nullable=True)
    
    # Progress tracking
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, nullable=False)
    best_loss = Column(Float, nullable=True)
    best_step = Column(Integer, nullable=True)
    validation_loss = Column(Float, nullable=True)  # Track validation loss
    
    # Storage paths (relative to STORAGE_ROOT)
    storage_path = Column(String(500), nullable=False)
    
    # Export status
    adapter_exported = Column(Boolean, default=False)
    fused_exported = Column(Boolean, default=False)
    
    # User notes/description
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # List of tag strings
    notes = Column(Text, nullable=True)  # User notes/thoughts about fine-tuning
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    
    # Live status tracking
    status_message = Column(Text, nullable=True, default="")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    paused_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    training_dataset = relationship("Dataset", foreign_keys=[training_dataset_id], back_populates="training_runs")
    validation_dataset = relationship("Dataset", foreign_keys=[validation_dataset_id])
    base_model = relationship("ModelRegistry")
    preset = relationship("TrainingPreset")
    metrics = relationship("TrainingMetric", back_populates="run", cascade="all, delete-orphan")


class TrainingMetric(Base):
    """Training metrics time-series data."""
    __tablename__ = "training_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), ForeignKey("training_runs.id"), nullable=False)
    
    step = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    
    # Training metrics
    train_loss = Column(Float, nullable=False)
    eval_loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=False)
    gradient_norm = Column(Float, nullable=True)
    
    # Resource metrics
    cpu_percent = Column(Float, nullable=True)
    memory_percent = Column(Float, nullable=True)
    gpu_memory_used_mb = Column(Integer, nullable=True)
    
    # Performance metrics
    samples_per_second = Column(Float, nullable=True)
    tokens_per_second = Column(Float, nullable=True)
    elapsed_seconds = Column(Float, nullable=True)
    
    # Relationship
    run = relationship("TrainingRun", back_populates="metrics")


# Database connection
_engine = None
_SessionLocal = None


def init_db(database_url: str = None, force_recreate: bool = False):
    """Initialize database connection and create tables."""
    from .config import get_settings
    
    global _engine, _SessionLocal
    
    if database_url is None:
        database_url = get_settings().database_url
    
    _engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
        echo=False,
        pool_pre_ping=True,
    )
    
    # Enable WAL mode for SQLite for better concurrent read/write performance
    if database_url.startswith("sqlite"):
        from sqlalchemy import event
        
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()
    
    if force_recreate:
        from sqlalchemy import text
        with _engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS training_runs"))
            conn.commit()
            logging.info("Dropped training_runs table for recreation")
    
    Base.metadata.create_all(bind=_engine)
    
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    
    return _engine


def get_db() -> Session:
    """Get database session (for FastAPI dependency injection)."""
    if _SessionLocal is None:
        init_db()
    
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_thread_safe_session() -> Session:
    """Get a database session safe for use in background threads.
    
    Use this in training callbacks and WebSocket handlers where
    FastAPI's dependency injection is not available. Always use
    in a try/finally block to ensure the session is closed.
    
    Example:
        db = get_thread_safe_session()
        try:
            run = db.query(TrainingRun).filter(...)
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()
    """
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def seed_initial_data():
    """Seed database with initial data (curated models, presets)."""
    from .config import get_settings
    
    if _SessionLocal is None:
        init_db()
    
    with _SessionLocal() as db:
        existing_presets = db.query(TrainingPreset).count()
        if existing_presets > 0:
            return
        
        # Seed training presets
        presets = [
            TrainingPreset(
                name="Quick",
                description="Fast iteration, 100 steps",
                is_default=True,
                steps=100,
                learning_rate=1e-4,
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.05,
                batch_size=4,
                warmup_steps=10,
                gradient_accumulation_steps=1,
                early_stopping_patience=0,
                gradient_checkpointing=False,
                num_lora_layers=8,
                prompt_masking=True
            ),
            TrainingPreset(
                name="Balanced",
                description="Good quality, 500 steps",
                is_default=False,
                steps=500,
                learning_rate=5e-5,
                lora_rank=16,
                lora_alpha=32,
                lora_dropout=0.05,
                batch_size=4,
                warmup_steps=50,
                gradient_accumulation_steps=1,
                early_stopping_patience=10,
                gradient_checkpointing=False,
                num_lora_layers=16,
                prompt_masking=True
            ),
            TrainingPreset(
                name="Maximum",
                description="Best quality, 1000 steps",
                is_default=False,
                steps=1000,
                learning_rate=1e-5,
                lora_rank=32,
                lora_alpha=64,
                lora_dropout=0.05,
                batch_size=2,
                warmup_steps=100,
                gradient_accumulation_steps=2,
                early_stopping_patience=20,
                gradient_checkpointing=True,
                num_lora_layers=16,
                prompt_masking=True
            ),
        ]
        
        for preset in presets:
            db.add(preset)
        
        db.commit()
