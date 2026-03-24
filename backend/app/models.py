"""
Database models for EdukaAI Studio.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, 
    String, Text, JSON, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.sql import func

Base = declarative_base()


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
    schema = Column(JSON, nullable=True)
    preview_samples = Column(JSON, nullable=False, default=list)
    
    # Metadata
    is_validation_set = Column(Boolean, default=False)
    parent_dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    training_runs = relationship("TrainingRun", foreign_keys="TrainingRun.training_dataset_id", back_populates="training_dataset")


class BaseModel(Base):
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
    seed = Column(Integer, nullable=True)
    checkpoint_frequency = Column(Integer, nullable=True)
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
    status = Column(String(50), nullable=False, default="pending")  # pending, running, paused, completed, failed
    
    # Dataset configuration
    training_dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    validation_dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True)
    use_auto_split = Column(Boolean, default=True)
    validation_split_ratio = Column(Float, nullable=True)
    
    # Base model
    base_model_id = Column(String(36), ForeignKey("base_models.id"), nullable=False)
    
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
    seed = Column(Integer, nullable=True)
    
    # Advanced parameters
    weight_decay = Column(Float, nullable=True)
    max_gradient_norm = Column(Float, nullable=True)
    checkpoint_frequency = Column(Integer, nullable=False, default=100)
    gradient_checkpointing = Column(Boolean, default=False)
    num_lora_layers = Column(Integer, nullable=False, default=16)
    prompt_masking = Column(Boolean, default=True)
    validation_split_percent = Column(Integer, nullable=False, default=10)  # 5, 10, or 15%
    
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
    gguf_exported = Column(Boolean, default=False)
    
    # User notes/description
    description = Column(Text, nullable=True)
    tags = Column(String(500), nullable=True)  # Comma-separated tags
    notes = Column(Text, nullable=True)  # User notes/thoughts about fine-tuning
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    paused_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    training_dataset = relationship("Dataset", foreign_keys=[training_dataset_id], back_populates="training_runs")
    validation_dataset = relationship("Dataset", foreign_keys=[validation_dataset_id])
    base_model = relationship("BaseModel")
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
        echo=False
    )
    
    if force_recreate:
        # Drop specific tables that need schema changes
        from sqlalchemy import text
        with _engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS training_runs"))
            conn.commit()
            print("Dropped training_runs table for recreation")
    
    # Create tables
    Base.metadata.create_all(bind=_engine)
    
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    
    return _engine


def get_db() -> Session:
    """Get database session."""
    if _SessionLocal is None:
        init_db()
    
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_initial_data():
    """Seed database with initial data (curated models, presets)."""
    from .config import get_settings
    
    init_db()
    
    with _SessionLocal() as db:
        # Check if already seeded
        existing_models = db.query(BaseModel).count()
        if existing_models > 0:
            return
        
        # Seed curated models with MLX-compatible models
        # All models are 4-bit quantized from mlx-community for optimal Mac performance
        curated_models = [
            # Small models (1-1.5B parameters) - Fast, low memory
            BaseModel(
                huggingface_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
                name="Llama 3.2 1B (4-bit)",
                architecture="llama",
                parameter_count=1_000_000_000,
                context_length=4096,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.v_proj"],
                    "recommended_max_seq_length": 4096,
                    "model_family": "Llama 3.2",
                    "size_category": "small",
                    "use_cases": ["Fast prototyping", "Edge deployment", "Low-latency apps"]
                }
            ),
            BaseModel(
                huggingface_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                name="Qwen 2.5 0.5B (4-bit)",
                architecture="qwen2",
                parameter_count=500_000_000,
                context_length=4096,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
                    "recommended_max_seq_length": 4096,
                    "model_family": "Qwen 2.5",
                    "size_category": "tiny",
                    "use_cases": ["Multilingual tasks", "Resource-constrained environments"]
                }
            ),
            BaseModel(
                huggingface_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
                name="Qwen 2.5 1.5B (4-bit)",
                architecture="qwen2",
                parameter_count=1_500_000_000,
                context_length=4096,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
                    "recommended_max_seq_length": 4096,
                    "model_family": "Qwen 2.5",
                    "size_category": "small",
                    "use_cases": ["Multilingual fine-tuning", "Chinese/English tasks"]
                }
            ),
            
            # Medium models (3-4B parameters) - Balanced
            BaseModel(
                huggingface_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
                name="Llama 3.2 3B (4-bit)",
                architecture="llama",
                parameter_count=3_000_000_000,
                context_length=4096,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.v_proj"],
                    "recommended_max_seq_length": 4096,
                    "model_family": "Llama 3.2",
                    "size_category": "medium",
                    "use_cases": ["General-purpose fine-tuning", "Chatbots", "Content generation"]
                }
            ),
            BaseModel(
                huggingface_id="mlx-community/Phi-3-mini-4k-instruct-4bit",
                name="Phi-3 Mini 4K (4-bit)",
                architecture="phi3",
                parameter_count=3_800_000_000,
                context_length=4096,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                    "recommended_max_seq_length": 4096,
                    "model_family": "Phi-3",
                    "size_category": "medium",
                    "use_cases": ["Coding tasks", "Reasoning", "Instruction following"]
                }
            ),
            BaseModel(
                huggingface_id="mlx-community/Qwen2.5-3B-Instruct-4bit",
                name="Qwen 2.5 3B (4-bit)",
                architecture="qwen2",
                parameter_count=3_000_000_000,
                context_length=4096,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
                    "recommended_max_seq_length": 4096,
                    "model_family": "Qwen 2.5",
                    "size_category": "medium",
                    "use_cases": ["Multilingual applications", "Asian languages", "Translation"]
                }
            ),
            BaseModel(
                huggingface_id="mlx-community/gemma-2-2b-it-4bit",
                name="Gemma 2 2B (4-bit)",
                architecture="gemma",
                parameter_count=2_000_000_000,
                context_length=4096,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
                    "recommended_max_seq_length": 4096,
                    "model_family": "Gemma 2",
                    "size_category": "small",
                    "use_cases": ["Research", "Educational tasks", "General assistant"]
                }
            ),
            
            # Large models (7-8B parameters) - High quality
            BaseModel(
                huggingface_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
                name="Qwen 2.5 7B (4-bit)",
                architecture="qwen2",
                parameter_count=7_000_000_000,
                context_length=4096,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
                    "recommended_max_seq_length": 4096,
                    "model_family": "Qwen 2.5",
                    "size_category": "large",
                    "use_cases": ["High-quality multilingual", "Professional applications", "Complex reasoning"]
                }
            ),
            BaseModel(
                huggingface_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                name="Llama 3.1 8B (4-bit)",
                architecture="llama",
                parameter_count=8_000_000_000,
                context_length=8192,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
                    "recommended_max_seq_length": 8192,
                    "model_family": "Llama 3.1",
                    "size_category": "large",
                    "use_cases": ["Production applications", "Long-context tasks", "Advanced reasoning"]
                }
            ),
            BaseModel(
                huggingface_id="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                name="Mistral 7B v0.3 (4-bit)",
                architecture="mistral",
                parameter_count=7_000_000_000,
                context_length=32768,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
                    "recommended_max_seq_length": 32768,
                    "model_family": "Mistral",
                    "size_category": "large",
                    "use_cases": ["Long-context applications", "Sliding window attention", "Efficient inference"]
                }
            ),
            BaseModel(
                huggingface_id="mlx-community/gemma-2-4b-it-4bit",
                name="Gemma 2 4B (4-bit)",
                architecture="gemma",
                parameter_count=4_000_000_000,
                context_length=4096,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
                    "recommended_max_seq_length": 4096,
                    "model_family": "Gemma 2",
                    "size_category": "medium",
                    "use_cases": ["Research", "Knowledge-intensive tasks", "Google ecosystem"]
                }
            ),
            BaseModel(
                huggingface_id="mlx-community/Phi-3-small-8k-instruct-4bit",
                name="Phi-3 Small 8K (4-bit)",
                architecture="phi3",
                parameter_count=7_000_000_000,
                context_length=8192,
                mlx_config={
                    "supports_lora": True,
                    "lora_target_modules": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                    "recommended_max_seq_length": 8192,
                    "model_family": "Phi-3",
                    "size_category": "large",
                    "use_cases": ["Coding", "Mathematical reasoning", "Logical tasks"]
                }
            )
        ]
        
        for model in curated_models:
            db.add(model)
        
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
