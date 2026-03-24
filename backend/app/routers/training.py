"""
Training router - Steps 2, 3, 4: Configuration, Execution, Management
"""

import os
import json
import shutil
import asyncio
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.exceptions import NotFoundError, ValidationError, TrainingError
from ..core.logging import get_logger
from ..models import get_db, TrainingRun, TrainingPreset, BaseModel as BaseModelDB, Dataset, generate_uuid
from ..config import get_config, get_settings
from ..ml.trainer import training_manager

router = APIRouter()
logger = get_logger(__name__)


def formatParameters(parameter_count: int) -> str:
    """Format parameter count to human-readable string (e.g., 3B, 7B, 13B)."""
    if parameter_count >= 1_000_000_000:
        return f"{parameter_count / 1_000_000_000:.1f}B"
    elif parameter_count >= 1_000_000:
        return f"{parameter_count / 1_000_000:.1f}M"
    else:
        return f"{parameter_count:,}"


def estimate_training_memory(
    model_params: int,
    lora_rank: int,
    lora_layers: int,
    batch_size: int,
    seq_length: int,
    total_params: int
) -> float:
    """
    Estimate training memory requirements in GB.
    
    Rough estimation based on:
    - Base model size (4-bit quantized ~0.5 bytes/param)
    - LoRA parameters (2 bytes/param for FP16)
    - Activations/gradients (depends on batch size and seq length)
    - Optimizer state (2x model size for AdamW)
    
    Returns:
        float: Estimated memory in GB
    """
    # Base model memory (4-bit quantized)
    base_model_gb = (total_params * 0.5) / (1024**3)
    
    # LoRA parameters memory (FP16)
    # Each LoRA layer adds rank * (input_dim + output_dim) * 2 bytes
    # Rough estimate: ~4x rank per layer
    lora_params_per_layer = lora_rank * 4
    lora_total_params = lora_params_per_layer * lora_layers
    lora_memory_gb = (lora_total_params * 2) / (1024**3)
    
    # Activation memory (rough estimate)
    # Depends on batch size, seq length, and hidden dimension
    # Rough: batch_size * seq_length * hidden_dim * 4 bytes
    hidden_dim = 2048  # Rough estimate for 1-3B models
    activation_gb = (batch_size * seq_length * hidden_dim * 4) / (1024**3)
    
    # Optimizer state (AdamW: 2x model size for momentum + variance)
    optimizer_gb = base_model_gb * 0.1  # Only for LoRA params, not base
    
    # Gradients (similar to parameters)
    gradient_gb = lora_memory_gb
    
    # Overhead (20% safety margin)
    total_gb = (base_model_gb + lora_memory_gb + activation_gb + 
                optimizer_gb + gradient_gb) * 1.2
    
    return max(total_gb, 2.0)  # Minimum 2GB


class BaseModelResponse(BaseModel):
    id: str
    huggingface_id: str
    name: str
    architecture: str
    parameter_count: int
    context_length: int
    mlx_config: Optional[Dict] = None
    is_custom: bool = False  # True if user-added custom model


class TrainingPresetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    steps: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    warmup_steps: int
    gradient_accumulation_steps: int
    early_stopping_patience: int
    gradient_checkpointing: bool
    num_lora_layers: int
    prompt_masking: bool


class CreateTrainingRunRequest(BaseModel):
    name: Optional[str] = None
    training_dataset_id: str
    validation_dataset_id: Optional[str] = None
    base_model_id: str
    preset_id: str
    
    # Validation split (percentage for auto-split: 5, 10, or 15)
    validation_split_percent: int = Field(default=10, ge=5, le=15)
    
    # Resource limits
    cpu_cores_limit: Optional[int] = Field(default=None, ge=1, le=32)
    gpu_memory_limit_gb: Optional[float] = None
    ram_limit_gb: Optional[float] = None
    
    # Custom overrides (optional)
    steps: Optional[int] = Field(default=None, ge=10, le=10000)
    learning_rate: Optional[float] = Field(default=None, gt=0)
    lora_rank: Optional[int] = Field(default=None, ge=4, le=128)
    lora_alpha: Optional[int] = Field(default=None, ge=4, le=256)
    batch_size: Optional[int] = Field(default=None, ge=1, le=64)
    max_seq_length: Optional[int] = Field(default=2048, ge=128, le=8192)
    
    # Additional hyperparameters
    lora_dropout: Optional[float] = Field(default=None, ge=0.0, le=0.5)
    warmup_steps: Optional[int] = Field(default=None, ge=0, le=1000)
    gradient_accumulation_steps: Optional[int] = Field(default=None, ge=1, le=32)
    early_stopping_patience: Optional[int] = Field(default=None, ge=0, le=50)
    gradient_checkpointing: Optional[bool] = None
    num_lora_layers: Optional[int] = Field(default=None, ge=4, le=32)
    prompt_masking: Optional[bool] = None


class ExportStatus(BaseModel):
    available: bool
    path: Optional[str] = None
    size_mb: Optional[float] = None
    exported_at: Optional[str] = None


class ExportStatusResponse(BaseModel):
    adapter: ExportStatus
    fused: ExportStatus
    gguf: ExportStatus


class DatasetInfo(BaseModel):
    """Dataset information including PII anonymization report."""
    id: str
    num_samples: int
    use_auto_split: bool
    validation_split_percent: int
    anonymization_report: Optional[Dict[str, Any]] = None


class TrainingConfigResponse(BaseModel):
    """Training configuration details."""
    steps: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    max_seq_length: int
    warmup_steps: int
    gradient_accumulation_steps: int
    early_stopping_patience: int
    gradient_checkpointing: bool
    num_lora_layers: int
    prompt_masking: bool
    validation_split_percent: int = 10  # 5, 10, or 15
    dataset: Optional[DatasetInfo] = None  # Dataset info with PII report


class TrainingRunResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    tags: Optional[str] = None
    notes: Optional[str] = None
    status: str
    current_step: int
    total_steps: int
    best_loss: Optional[float]
    best_step: Optional[int]
    validation_loss: Optional[float]
    completed_at: Optional[str]
    error_message: Optional[str] = None
    base_model: BaseModelResponse
    created_at: str
    adapter_exported: bool = False
    fused_exported: bool = False
    gguf_exported: bool = False
    training_config: Optional[TrainingConfigResponse] = None
    
    class Config:
        from_attributes = True


def build_training_config_response(run) -> TrainingConfigResponse:
    """Build training config response from TrainingRun model."""
    # Build dataset info with anonymization report if available
    dataset_info = None
    if run.training_dataset:
        dataset = run.training_dataset
        anonymization_report = None
        if dataset.validation_report:
            anon_data = dataset.validation_report.get("sanitization", {}).get("anonymization", {})
            anonymization_report = {
                "total_samples": anon_data.get("total_samples", dataset.num_samples),
                "samples_with_pii": anon_data.get("samples_with_pii", 0),
                "total_replacements": anon_data.get("total_replacements", 0),
                "types_found": anon_data.get("types_found", {}),
                "fields_affected": anon_data.get("fields_affected", [])
            }
        
        # Get validation split percent from training config or use default
        val_split = getattr(run, 'validation_split_percent', 10)
        
        dataset_info = DatasetInfo(
            id=dataset.id,
            num_samples=dataset.num_samples,
            use_auto_split=run.validation_dataset_id is None,
            validation_split_percent=val_split,
            anonymization_report=anonymization_report
        )
    
    return TrainingConfigResponse(
        steps=run.steps,
        learning_rate=run.learning_rate,
        lora_rank=run.lora_rank,
        lora_alpha=run.lora_alpha,
        lora_dropout=run.lora_dropout,
        batch_size=run.batch_size,
        max_seq_length=run.max_seq_length,
        warmup_steps=run.warmup_steps,
        gradient_accumulation_steps=run.gradient_accumulation_steps,
        early_stopping_patience=run.early_stopping_patience,
        gradient_checkpointing=run.gradient_checkpointing,
        num_lora_layers=run.num_lora_layers,
        prompt_masking=run.prompt_masking,
        validation_split_percent=getattr(run, 'validation_split_percent', 10),
        dataset=dataset_info
    )



    huggingface_id: str = Field(..., min_length=3, max_length=255)


class ValidateModelResponse(BaseModel):
    is_valid: bool
    message: str
    model_info: Optional[Dict[str, Any]] = None
    suggested_name: Optional[str] = None

# Rebuild models to resolve forward references
ValidateModelResponse.model_rebuild()
DatasetInfo.model_rebuild()
TrainingConfigResponse.model_rebuild()


@router.post("/base-models/validate", response_model=ValidateModelResponse)
async def validate_custom_model(
    request: ValidateModelRequest,
    db: Session = Depends(get_db)
):
    """
    Validate a custom HuggingFace model for MLX compatibility.
    Checks if the model exists and has MLX-compatible format.
    Accepts both model IDs (org/model-name) and full URLs (https://huggingface.co/org/model-name).
    """
    raw_input = request.huggingface_id.strip()
    
    # Handle full HuggingFace URLs - extract model ID
    huggingface_id = raw_input
    if raw_input.startswith('https://huggingface.co/'):
        # Extract model ID from URL: https://huggingface.co/org/model-name -> org/model-name
        huggingface_id = raw_input.replace('https://huggingface.co/', '')
        # Remove trailing slash if present
        huggingface_id = huggingface_id.rstrip('/')
        logger.info(f"Extracted model ID '{huggingface_id}' from URL '{raw_input}'")
    
    # Validate format (org/model-name or just model-name, max 2 levels)
    if not re.match(r'^[\w\-\.]+(/[\w\-\.]+)?$', huggingface_id):
        return ValidateModelResponse(
            is_valid=False,
            message="Invalid HuggingFace model ID format. Expected: 'organization/model-name' or 'model-name'",
            suggested_name=None
        )
    
    # Check if already in database
    existing = db.query(BaseModelDB).filter(
        BaseModelDB.huggingface_id == huggingface_id
    ).first()
    
    if existing:
        return ValidateModelResponse(
            is_valid=True,
            message=f"Model already exists: {existing.name}",
            model_info={
                "id": existing.id,
                "name": existing.name,
                "architecture": existing.architecture,
                "parameter_count": existing.parameter_count,
                "context_length": existing.context_length
            },
            suggested_name=existing.name
        )
    
    # Try to fetch model info from HuggingFace
    try:
        from huggingface_hub import model_info, HfApi
        
        logger.info(f"Validating custom model: {huggingface_id}")
        
        # Get model metadata
        info = model_info(huggingface_id)
        
        # Check for MLX compatibility markers
        tags = info.tags or []
        
        # Check if it's already MLX-formatted (mlx-community, mlx-4bit, etc.)
        is_mlx_formatted = any(tag in tags for tag in ['mlx', 'mlx-community', '4bit', '8bit'])
        
        # Check for common model architectures
        supported_architectures = [
            'llama', 'qwen2', 'mistral', 'mixtral', 'phi', 'gemma', 
            'gemma2', 'qwen2.5', 'llama3', 'phi3'
        ]
        
        architecture = None
        for arch in supported_architectures:
            if arch in huggingface_id.lower() or any(arch in tag.lower() for tag in tags):
                architecture = arch.upper()
                break
        
        if not architecture:
            architecture = "Unknown"
        
        # Extract parameter count from tags or model ID
        param_count = 0
        import re as regex
        
        # Try to find parameter count in model ID (e.g., "3B", "7B", "1.5B")
        param_match = regex.search(r'(\d+\.?\d*)[Bb]', huggingface_id)
        if param_match:
            param_count = int(float(param_match.group(1)) * 1_000_000_000)
        
        # Default parameter counts for common models
        if param_count == 0:
            defaults = {
                '0.5b': 500_000_000, '0.5B': 500_000_000,
                '1b': 1_000_000_000, '1B': 1_000_000_000,
                '1.5b': 1_500_000_000, '1.5B': 1_500_000_000,
                '2b': 2_000_000_000, '2B': 2_000_000_000,
                '3b': 3_000_000_000, '3B': 3_000_000_000,
                '4b': 4_000_000_000, '4B': 4_000_000_000,
                '7b': 7_000_000_000, '7B': 7_000_000_000,
                '8b': 8_000_000_000, '8B': 8_000_000_000,
                '13b': 13_000_000_000, '13B': 13_000_000_000
            }
            for key, value in defaults.items():
                if key in huggingface_id.lower():
                    param_count = value
                    break
        
        # If still no param count, use default
        if param_count == 0:
            param_count = 3_000_000_000  # Assume 3B if unknown
        
        # Extract context length from tags or use defaults
        context_length = 8192  # Default
        
        # Generate suggested name
        parts = huggingface_id.split('/')
        base_name = parts[-1] if len(parts) > 1 else huggingface_id
        suggested_name = base_name.replace('-', ' ').replace('_', ' ').title()
        
        # Build response
        model_info_dict = {
            "huggingface_id": huggingface_id,
            "architecture": architecture,
            "parameter_count": param_count,
            "context_length": context_length,
            "is_mlx_formatted": is_mlx_formatted,
            "tags": tags[:10]  # First 10 tags
        }
        
        if is_mlx_formatted:
            return ValidateModelResponse(
                is_valid=True,
                message=f"Model appears MLX-compatible: {huggingface_id}. Architecture: {architecture}. Estimated parameters: {param_count/1e9:.1f}B",
                model_info=model_info_dict,
                suggested_name=suggested_name
            )
        else:
            return ValidateModelResponse(
                is_valid=True,
                message=f"Model found: {huggingface_id}. Note: This doesn't appear to be MLX-formatted. Consider using a model from mlx-community for best results. Architecture detected: {architecture}. Estimated parameters: {param_count/1e9:.1f}B",
                model_info=model_info_dict,
                suggested_name=suggested_name
            )
            
    except Exception as e:
        logger.error(f"Error validating model {huggingface_id}: {e}")
        return ValidateModelResponse(
            is_valid=False,
            message=f"Could not validate model '{huggingface_id}'. Error: {str(e)}. Please check the model ID and ensure it's publicly available on HuggingFace.",
            suggested_name=None
        )


@router.post("/base-models/custom", response_model=BaseModelResponse)
async def add_custom_model(
    request: ValidateModelRequest,
    db: Session = Depends(get_db)
):
    """
    Add a custom model to the database after validation.
    """
    # First validate
    validation = await validate_custom_model(request, db)
    
    if not validation.is_valid:
        raise ValidationError(validation.message)
    
    huggingface_id = request.huggingface_id.strip()
    
    # Check if already exists
    existing = db.query(BaseModelDB).filter(
        BaseModelDB.huggingface_id == huggingface_id
    ).first()
    
    if existing:
        return BaseModelResponse(
            id=existing.id,
            huggingface_id=existing.huggingface_id,
            name=existing.name,
            architecture=existing.architecture,
            parameter_count=existing.parameter_count,
            context_length=existing.context_length,
            mlx_config={
                **existing.mlx_config,
                "is_curated": existing.is_curated,
                "is_custom": not existing.is_curated
            },
            is_custom=not existing.is_curated
        )
    
    # Create new model entry
    info = validation.model_info
    model_id = generate_uuid()
    
    model = BaseModelDB(
        id=model_id,
        huggingface_id=huggingface_id,
        name=validation.suggested_name or huggingface_id,
        architecture=info.get("architecture", "Unknown"),
        parameter_count=info.get("parameter_count", 3_000_000_000),
        context_length=info.get("context_length", 8192),
        is_active=True,
        is_curated=False,  # Mark as custom
        mlx_config={
            "is_custom": True,
            "validation_info": validation.model_info,
            "added_at": datetime.now().isoformat()
        }
    )
    
    db.add(model)
    db.commit()
    db.refresh(model)
    
    logger.info(f"Added custom model: {huggingface_id} (ID: {model_id})")
    
    return BaseModelResponse(
        id=model.id,
        huggingface_id=model.huggingface_id,
        name=model.name,
        architecture=model.architecture,
        parameter_count=model.parameter_count,
        context_length=model.context_length,
        mlx_config={
            **model.mlx_config,
            "is_curated": model.is_curated,
            "is_custom": not model.is_curated
        },
        is_custom=True  # Mark as custom
    )


@router.get("/base-models", response_model=List[BaseModelResponse])
async def list_base_models(db: Session = Depends(get_db)):
    """List all base models (curated + custom)."""
    # Get curated models first
    curated_models = db.query(BaseModelDB).filter(
        BaseModelDB.is_active == True,
        BaseModelDB.is_curated == True
    ).order_by(BaseModelDB.parameter_count).all()
    
    # Get custom models (added by users)
    custom_models = db.query(BaseModelDB).filter(
        BaseModelDB.is_active == True,
        BaseModelDB.is_curated == False
    ).order_by(BaseModelDB.created_at.desc()).all()
    
    # Combine: curated first, then custom
    all_models = curated_models + custom_models
    
    return [
        BaseModelResponse(
            id=m.id,
            huggingface_id=m.huggingface_id,
            name=m.name,
            architecture=m.architecture,
            parameter_count=m.parameter_count,
            context_length=m.context_length,
            mlx_config={
                **m.mlx_config,
                "is_curated": m.is_curated,
                "is_custom": not m.is_curated
            },
            is_custom=not m.is_curated
        )
        for m in all_models
    ]


@router.delete("/base-models/{model_id}")
async def delete_custom_model(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a custom base model.
    Only custom models (not curated) can be deleted.
    Cannot delete models that are in use by active or pending training runs.
    """
    model = db.query(BaseModelDB).filter(BaseModelDB.id == model_id).first()
    
    if not model:
        raise NotFoundError(f"Model {model_id} not found")
    
    # Only allow deletion of custom models
    if model.is_curated:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete curated models. Only custom models can be deleted."
        )
    
    # Check if model is used by any training runs
    active_runs = db.query(TrainingRun).filter(
        TrainingRun.base_model_id == model_id,
        TrainingRun.status.in_(["pending", "running", "paused"])
    ).count()
    
    if active_runs > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete model: it is currently in use by {active_runs} active training run(s). "
                   "Wait for training to complete or stop the runs first."
        )
    
    # Check if model is used by any completed/failed training runs
    completed_runs = db.query(TrainingRun).filter(
        TrainingRun.base_model_id == model_id
    ).count()
    
    if completed_runs > 0:
        # Soft delete: mark as inactive instead of hard delete
        model.is_active = False
        logger.info(f"Soft-deleted custom model {model_id} (in use by {completed_runs} completed runs)")
        message = f"Custom model '{model.name}' has been removed from the model list. " \
                  f"Note: {completed_runs} existing training runs will continue to work."
    else:
        # Hard delete: no training runs use this model
        db.delete(model)
        logger.info(f"Hard-deleted custom model {model_id}")
        message = f"Custom model '{model.name}' has been permanently deleted."
    
    db.commit()
    
    return {
        "success": True,
        "message": message,
        "model_id": model_id,
        "deleted": completed_runs == 0  # True if hard deleted, False if soft deleted
    }


@router.get("/training-presets", response_model=List[TrainingPresetResponse])
async def list_training_presets(db: Session = Depends(get_db)):
    """List all training presets."""
    presets = db.query(TrainingPreset).all()
    
    return [
        TrainingPresetResponse(
            id=p.id,
            name=p.name,
            description=p.description,
            steps=p.steps,
            learning_rate=p.learning_rate,
            lora_rank=p.lora_rank,
            lora_alpha=p.lora_alpha,
            lora_dropout=p.lora_dropout,
            batch_size=p.batch_size,
            warmup_steps=p.warmup_steps,
            gradient_accumulation_steps=p.gradient_accumulation_steps,
            early_stopping_patience=p.early_stopping_patience,
            gradient_checkpointing=p.gradient_checkpointing,
            num_lora_layers=p.num_lora_layers,
            prompt_masking=p.prompt_masking
        )
        for p in presets
    ]


@router.post("/training/runs", response_model=TrainingRunResponse)
async def create_training_run(
    request: CreateTrainingRunRequest,
    db: Session = Depends(get_db)
):
    """Create a new training run with configuration."""
    
    # Validate dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == request.training_dataset_id).first()
    if not dataset:
        raise NotFoundError(f"Training dataset {request.training_dataset_id} not found")
    
    # Validate base model exists
    base_model = db.query(BaseModelDB).filter(BaseModelDB.id == request.base_model_id).first()
    if not base_model:
        raise NotFoundError(f"Base model {request.base_model_id} not found")
    
    # Validate preset exists
    preset = db.query(TrainingPreset).filter(TrainingPreset.id == request.preset_id).first()
    if not preset:
        raise NotFoundError(f"Training preset {request.preset_id} not found")
    
    # Generate run name if not provided
    if not request.name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        request.name = f"{base_model.name}-{timestamp}"
    
    # Create storage directory for this run
    run_id = generate_uuid()
    storage_path = f"./storage/runs/{run_id}"
    
    # Create directory structure
    for subdir in ["config", "data", "checkpoints", "logs", "exports"]:
        Path(f"{storage_path}/{subdir}").mkdir(parents=True, exist_ok=True)
    
    # Copy dataset to run directory
    training_data_path = f"{storage_path}/data/train.jsonl"
    shutil.copy(dataset.file_path, training_data_path)
    
    # Handle validation dataset
    validation_data_path = None
    validation_dataset_id = None
    
    if request.validation_dataset_id:
        # Use separate validation file
        val_dataset = db.query(Dataset).filter(Dataset.id == request.validation_dataset_id).first()
        if not val_dataset:
            raise NotFoundError(f"Validation dataset {request.validation_dataset_id} not found")
        validation_data_path = f"{storage_path}/data/validation.jsonl"
        shutil.copy(val_dataset.file_path, validation_data_path)
        validation_dataset_id = val_dataset.id
        use_auto_split = False
        validation_split_percent = None
    else:
        # Auto-split from training data using configured percentage
        use_auto_split = True
        validation_split_percent = request.validation_split_percent
    
    # Save configuration
    config = {
        "run_id": run_id,
        "name": request.name,
        "base_model": {
            "id": base_model.id,
            "huggingface_id": base_model.huggingface_id,
            "name": base_model.name
        },
        "dataset": {
            "training_id": dataset.id,
            "training_path": training_data_path,
            "validation_id": validation_dataset_id,
            "validation_path": validation_data_path,
            "use_auto_split": use_auto_split,
            "validation_split_percent": validation_split_percent,
            "anonymization_report": dataset.validation_report.get("sanitization", {}).get("anonymization", {
                "total_samples": dataset.num_samples,
                "samples_with_pii": 0,
                "total_replacements": 0,
                "types_found": {},
                "fields_affected": []
            }) if dataset.validation_report else {
                "total_samples": dataset.num_samples,
                "samples_with_pii": 0,
                "total_replacements": 0,
                "types_found": {},
                "fields_affected": []
            }
        },
        "hyperparameters": {
            "steps": request.steps or preset.steps,
            "learning_rate": request.learning_rate or preset.learning_rate,
            "lora_rank": request.lora_rank or preset.lora_rank,
            "lora_alpha": request.lora_alpha or preset.lora_alpha,
            "lora_dropout": request.lora_dropout or preset.lora_dropout,
            "batch_size": request.batch_size or preset.batch_size,
            "warmup_steps": request.warmup_steps or preset.warmup_steps,
            "gradient_accumulation_steps": request.gradient_accumulation_steps or preset.gradient_accumulation_steps,
            "early_stopping_patience": request.early_stopping_patience or preset.early_stopping_patience,
            "max_seq_length": request.max_seq_length,
            "gradient_checkpointing": request.gradient_checkpointing if request.gradient_checkpointing is not None else preset.gradient_checkpointing,
            "num_lora_layers": request.num_lora_layers or preset.num_lora_layers,
            "prompt_masking": request.prompt_masking if request.prompt_masking is not None else preset.prompt_masking,
            "validation_split_percent": validation_split_percent
        },
        "resource_limits": {
            "cpu_cores": request.cpu_cores_limit,
            "gpu_memory_gb": request.gpu_memory_limit_gb,
            "ram_gb": request.ram_limit_gb
        },
        "created_at": datetime.now().isoformat()
    }
    
    with open(f"{storage_path}/config/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create database record
    run = TrainingRun(
        id=run_id,
        name=request.name,
        status="pending",
        training_dataset_id=dataset.id,
        validation_dataset_id=validation_dataset_id,
        base_model_id=base_model.id,
        preset_id=preset.id,
        
        # Auto-generate description with key training details
        description=f"Fine-tuned {base_model.name} ({formatParameters(base_model.parameter_count)}) "
                   f"on {dataset.num_samples:,} samples from '{dataset.name}'. "
                   f"LoRA rank {request.lora_rank or preset.lora_rank}, {request.steps or preset.steps} steps. "
                   f"Base model: {base_model.huggingface_id}",
        
        # Hyperparameters
        steps=request.steps or preset.steps,
        learning_rate=request.learning_rate or preset.learning_rate,
        lora_rank=request.lora_rank or preset.lora_rank,
        lora_alpha=request.lora_alpha or preset.lora_alpha,
        lora_dropout=request.lora_dropout or preset.lora_dropout,
        batch_size=request.batch_size or preset.batch_size,
        warmup_steps=request.warmup_steps or preset.warmup_steps,
        gradient_accumulation_steps=request.gradient_accumulation_steps or preset.gradient_accumulation_steps,
        early_stopping_patience=request.early_stopping_patience or preset.early_stopping_patience,
        max_seq_length=request.max_seq_length,
        gradient_checkpointing=request.gradient_checkpointing if request.gradient_checkpointing is not None else preset.gradient_checkpointing,
        num_lora_layers=request.num_lora_layers or preset.num_lora_layers,
        prompt_masking=request.prompt_masking if request.prompt_masking is not None else preset.prompt_masking,
        validation_split_percent=validation_split_percent or 10,
        
        # Resource limits
        cpu_cores_limit=request.cpu_cores_limit,
        gpu_memory_limit_gb=request.gpu_memory_limit_gb,
        ram_limit_gb=request.ram_limit_gb,
        
        total_steps=request.steps or preset.steps,
        storage_path=storage_path
    )
    
    db.add(run)
    db.commit()
    db.refresh(run)
    
    return TrainingRunResponse(
        id=run.id,
        name=run.name,
        description=run.description,
        tags=run.tags,
        status=run.status,
        current_step=run.current_step,
        total_steps=run.total_steps,
        best_loss=run.best_loss,
        best_step=run.best_step,
        validation_loss=run.validation_loss,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        error_message=run.error_message,
        adapter_exported=run.adapter_exported,
        fused_exported=run.fused_exported,
        gguf_exported=run.gguf_exported,
        training_config=build_training_config_response(run),
        base_model=BaseModelResponse(
            id=base_model.id,
            huggingface_id=base_model.huggingface_id,
            name=base_model.name,
            architecture=base_model.architecture,
            parameter_count=base_model.parameter_count,
            context_length=base_model.context_length,
            mlx_config=base_model.mlx_config,
            is_custom=not getattr(base_model, 'is_curated', True)
        ),
        created_at=run.created_at.isoformat()
    )


@router.get("/training/runs", response_model=List[TrainingRunResponse])
async def list_training_runs(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List training runs with optional status filter."""
    query = db.query(TrainingRun)
    
    if status:
        query = query.filter(TrainingRun.status == status)
    
    runs = query.order_by(TrainingRun.created_at.desc()).all()
    
    return [
        TrainingRunResponse(
            id=r.id,
            name=r.name,
            description=r.description,
            tags=r.tags,
            notes=r.notes,
            status=r.status,
            current_step=r.current_step,
            total_steps=r.total_steps,
            best_loss=r.best_loss,
            best_step=r.best_step,
            validation_loss=r.validation_loss,
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
            error_message=r.error_message,
            adapter_exported=r.adapter_exported,
            fused_exported=r.fused_exported,
            gguf_exported=r.gguf_exported,
            training_config=build_training_config_response(r),
            base_model=BaseModelResponse(
                id=r.base_model.id,
                huggingface_id=r.base_model.huggingface_id,
                name=r.base_model.name,
                architecture=r.base_model.architecture,
                parameter_count=r.base_model.parameter_count,
                context_length=r.base_model.context_length,
                mlx_config=r.base_model.mlx_config,
                is_custom=not getattr(r.base_model, 'is_curated', True)
            ),
            created_at=r.created_at.isoformat()
        )
        for r in runs
    ]


@router.get("/training/runs/{run_id}", response_model=TrainingRunResponse)
async def get_training_run(run_id: str, db: Session = Depends(get_db)):
    """Get training run details."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    return TrainingRunResponse(
        id=run.id,
        name=run.name,
        description=run.description,
        tags=run.tags,
        notes=run.notes,
        status=run.status,
        current_step=run.current_step,
        total_steps=run.total_steps,
        best_loss=run.best_loss,
        best_step=run.best_step,
        validation_loss=run.validation_loss,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        error_message=run.error_message,
        adapter_exported=run.adapter_exported,
        fused_exported=run.fused_exported,
        gguf_exported=run.gguf_exported,
        training_config=build_training_config_response(run),
        base_model=BaseModelResponse(
            id=run.base_model.id,
            huggingface_id=run.base_model.huggingface_id,
            name=run.base_model.name,
            architecture=run.base_model.architecture,
            parameter_count=run.base_model.parameter_count,
            context_length=run.base_model.context_length,
            mlx_config=run.base_model.mlx_config,
            is_custom=not getattr(run.base_model, 'is_curated', True)
        ),
        created_at=run.created_at.isoformat()
    )


# Note: Duplicate list_training_runs removed - kept the first one above


@router.get("/training/runs/{run_id}/checkpoints")
async def list_checkpoints(run_id: str, db: Session = Depends(get_db)):
    """List checkpoints for a training run."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    # MLX-LM saves checkpoints in the main storage directory, not in checkpoints/ subdirectory
    # Check both locations for backwards compatibility
    storage_path = run.storage_path
    checkpoints_subdir = f"{storage_path}/checkpoints"
    
    checkpoint_files = []
    
    # Look in main storage directory
    if os.path.exists(storage_path):
        for filename in os.listdir(storage_path):
            if filename.endswith('_adapters.safetensors') or filename.endswith('_adapters.npz'):
                checkpoint_files.append((storage_path, filename))
    
    # Also look in checkpoints subdirectory if it exists
    if os.path.exists(checkpoints_subdir):
        for filename in os.listdir(checkpoints_subdir):
            if filename.endswith('.safetensors') or filename.endswith('.npz'):
                checkpoint_files.append((checkpoints_subdir, filename))
    
    checkpoints = []
    for dir_path, filename in checkpoint_files:
        try:
            # Extract step number from filename (e.g., 0000100_adapters.safetensors)
            if '_' in filename:
                step_match = filename.split('_')[0]
                step = int(step_match)
                
                # Get file size
                file_path = os.path.join(dir_path, filename)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                file_size_mb = round(file_size / (1024 * 1024), 2)
                
                checkpoints.append({
                    "step": step,
                    "filename": filename,
                    "is_best": step == run.best_step,
                    "path": f"{dir_path}/{filename}",
                    "size_mb": file_size_mb
                })
        except (ValueError, IndexError):
            continue
    
    return sorted(checkpoints, key=lambda x: x["step"])


@router.get("/training/runs/{run_id}/checkpoints/{step}/download")
async def download_checkpoint(run_id: str, step: int, db: Session = Depends(get_db)):
    """Download a specific checkpoint file."""
    from fastapi.responses import FileResponse
    
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    # Look for checkpoint in main storage directory
    storage_path = run.storage_path
    checkpoint_filename = f"{step:08d}_adapters.safetensors"
    checkpoint_path = os.path.join(storage_path, checkpoint_filename)
    
    # Also check checkpoints subdirectory
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(storage_path, "checkpoints", checkpoint_filename)
    
    if not os.path.exists(checkpoint_path):
        raise NotFoundError(f"Checkpoint at step {step} not found")
    
    return FileResponse(
        checkpoint_path,
        filename=checkpoint_filename,
        media_type="application/octet-stream"
    )


@router.delete("/training/runs/{run_id}")
async def delete_training_run(run_id: str, db: Session = Depends(get_db)):
    """Delete a training run and all associated data."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    # Delete storage directory
    try:
        if os.path.exists(run.storage_path):
            shutil.rmtree(run.storage_path)
    except Exception as e:
        # Log error but continue with DB deletion
        print(f"Warning: Failed to delete storage for run {run_id}: {e}")
    
    # Delete database record
    db.delete(run)
    db.commit()
    
    return {"message": "Training run deleted successfully"}


class UpdateRunRequest(BaseModel):
    """Request to update training run metadata."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=5000)
    tags: Optional[str] = Field(None, max_length=500)
    notes: Optional[str] = Field(None, max_length=10000)  # Notes can be longer


@router.patch("/training/runs/{run_id}", response_model=TrainingRunResponse)
async def update_training_run(
    run_id: str,
    request: UpdateRunRequest,
    db: Session = Depends(get_db)
):
    """
    Update training run metadata (name, description, tags).
    Does not affect training logic or exports.
    """
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    # Update fields if provided
    if request.name is not None:
        run.name = request.name
        logger.info(f"Updated run {run_id} name to: {request.name}")
    
    if request.description is not None:
        run.description = request.description
        logger.info(f"Updated run {run_id} description")
    
    if request.tags is not None:
        run.tags = request.tags
        logger.info(f"Updated run {run_id} tags to: {request.tags}")
    
    if request.notes is not None:
        run.notes = request.notes
        logger.info(f"Updated run {run_id} notes")
    
    # Update timestamp
    run.updated_at = datetime.now()
    
    db.commit()
    db.refresh(run)
    
    # Build response
    base_model = run.base_model
    
    return TrainingRunResponse(
        id=run.id,
        name=run.name,
        description=run.description,
        tags=run.tags,
        notes=run.notes,
        status=run.status,
        current_step=run.current_step,
        total_steps=run.total_steps,
        best_loss=run.best_loss,
        best_step=run.best_step,
        validation_loss=run.validation_loss,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        error_message=run.error_message,
        adapter_exported=run.adapter_exported,
        fused_exported=run.fused_exported,
        gguf_exported=run.gguf_exported,
        training_config=build_training_config_response(run),
        base_model=BaseModelResponse(
            id=base_model.id,
            huggingface_id=base_model.huggingface_id,
            name=base_model.name,
            architecture=base_model.architecture,
            parameter_count=base_model.parameter_count,
            context_length=base_model.context_length,
            mlx_config=base_model.mlx_config,
            is_custom=not getattr(base_model, 'is_curated', True)
        ),
        created_at=run.created_at.isoformat()
    )


@router.post("/training/runs/{run_id}/start")
async def start_training(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Start training for a run."""
    from ..ml.trainer import training_manager, TrainingConfig
    from ..core.exceptions import ResourceLimitError
    
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    if run.status != "pending":
        raise ValidationError(f"Cannot start run with status: {run.status}")
    
    # SECURITY CHECK #1: Prevent concurrent training runs
    active_runs = db.query(TrainingRun).filter(
        TrainingRun.status.in_(['running', 'downloading', 'loading_model', 'paused'])
    ).count()
    
    if active_runs > 0:
        logger.warning(f"Attempted to start training {run_id} while {active_runs} runs are active")
        raise ResourceLimitError(
            "Training already in progress. Only one training can run at a time. "
            "Please wait for the current training to complete or stop it."
        )
    
    # Load config
    config_path = f"{run.storage_path}/config/training_config.json"
    with open(config_path) as f:
        saved_config = json.load(f)
    
    # SECURITY CHECK #2: Resource limit validation
    # Estimate memory requirements and validate against safe limits
    estimated_memory_gb = estimate_training_memory(
        model_params=saved_config["base_model"].get("parameter_count", 1_000_000_000),
        lora_rank=run.lora_rank,
        lora_layers=run.num_lora_layers,
        batch_size=run.batch_size,
        seq_length=run.max_seq_length,
        total_params=saved_config["base_model"].get("parameter_count", 1_000_000_000)
    )
    
    MAX_ALLOWED_MEMORY_GB = 16  # 16GB RAM limit for safety
    if estimated_memory_gb > MAX_ALLOWED_MEMORY_GB:
        logger.error(
            f"Training {run_id} requires {estimated_memory_gb:.1f}GB RAM, "
            f"exceeds limit of {MAX_ALLOWED_MEMORY_GB}GB"
        )
        raise ResourceLimitError(
            f"Configuration requires {estimated_memory_gb:.1f}GB RAM. "
            f"Maximum allowed: {MAX_ALLOWED_MEMORY_GB}GB. "
            f"Please reduce batch_size, max_seq_length, or lora_rank."
        )
    
    logger.info(
        f"Starting training {run_id}: estimated memory {estimated_memory_gb:.1f}GB, "
        f"model {saved_config['base_model']['name']}, "
        f"steps={run.steps}, rank={run.lora_rank}, batch={run.batch_size}"
    )
    
    # Create training config
    # Extract model ID from URL if needed (handle https://huggingface.co/org/model format)
    raw_model_id = saved_config["base_model"]["huggingface_id"]
    if raw_model_id.startswith('https://huggingface.co/'):
        model_id = raw_model_id.replace('https://huggingface.co/', '').rstrip('/')
        logger.info(f"Extracted model ID '{model_id}' from URL '{raw_model_id}'")
    else:
        model_id = raw_model_id
    
    training_config = TrainingConfig(
        model_id=model_id,
        data_path=saved_config["dataset"]["training_path"],
        output_path=run.storage_path,
        steps=run.steps,
        learning_rate=run.learning_rate,
        lora_rank=run.lora_rank,
        lora_alpha=run.lora_alpha,
        lora_dropout=run.lora_dropout,
        batch_size=run.batch_size,
        max_seq_length=run.max_seq_length,
        warmup_steps=run.warmup_steps,
        gradient_accumulation_steps=run.gradient_accumulation_steps,
        early_stopping_patience=run.early_stopping_patience,
        gradient_checkpointing=run.gradient_checkpointing,
        num_lora_layers=run.num_lora_layers,
        prompt_masking=run.prompt_masking,
        cpu_cores_limit=run.cpu_cores_limit,
        gpu_memory_limit_gb=run.gpu_memory_limit_gb,
        ram_limit_gb=run.ram_limit_gb
    )
    
    # Callbacks for updating database (each creates its own session)
    def on_step_complete(data):
        """Update run progress in database and save step metrics."""
        from ..models import get_db, TrainingMetric
        db = next(get_db())
        try:
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                # Update run progress
                run.current_step = data["step"]
                if data.get("best_loss") is not None:
                    run.best_loss = data["best_loss"]
                    run.best_step = data["best_step"]
                run.status = "running"
                
                # Save step metrics to TrainingMetric table for historical curves
                metric = TrainingMetric(
                    run_id=run_id,
                    step=data["step"],
                    train_loss=data.get("loss"),
                    eval_loss=data.get("validation_loss"),  # Save validation loss if available
                    learning_rate=data.get("learning_rate", run.learning_rate),
                    samples_per_second=data.get("it_per_second"),
                    tokens_per_second=data.get("tokens_per_second"),
                    elapsed_seconds=data.get("elapsed_seconds")
                )
                db.add(metric)
                
                db.commit()
                logger.info(f"Updated run {run_id}: step={data['step']}, loss={data.get('best_loss')}, saved metric")
        except Exception as e:
            logger.error(f"Error in on_step_complete: {e}")
            db.rollback()
        finally:
            db.close()
    
    def on_training_complete():
        """Mark run as completed."""
        from ..models import get_db
        db = next(get_db())
        try:
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                run.status = "completed"
                run.completed_at = datetime.now()
                run.current_step = run.total_steps
                if run.best_loss is None:
                    run.best_loss = 0  # Default if not set
                db.commit()
                logger.info(f"Training run {run_id} marked as completed")
                training_manager.cleanup(run_id)
        except Exception as e:
            logger.error(f"Error in on_training_complete: {e}", exc_info=True)
        finally:
            db.close()
    
    def on_error(error_msg):
        """Mark run as failed."""
        from ..models import get_db
        db = next(get_db())
        try:
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                run.status = "failed"
                run.error_message = error_msg
                db.commit()
                logger.error(f"Training run {run_id} marked as failed: {error_msg}")
                training_manager.cleanup(run_id)
        except Exception as e:
            logger.error(f"Error in on_error: {e}", exc_info=True)
        finally:
            db.close()
    
    def on_status_change(status: str, message: str):
        """Handle status changes like downloading, loading_model, etc."""
        # Just log it - the WebSocket will pick up status changes via get_stats()
        logger.info(f"Training {run_id} status changed: {status} - {message}")
    
    # Start training
    try:
        await training_manager.create_training(
            run_id=run_id,
            config=training_config,
            step_callback=on_step_complete,
            complete_callback=on_training_complete,
            error_callback=on_error,
            status_callback=on_status_change
        )
        
        return {"message": "Training started", "run_id": run_id}
        
    except Exception as e:
        run.status = "failed"
        run.error_message = str(e)
        db.commit()
        raise TrainingError(f"Failed to start training: {str(e)}")


@router.post("/training/runs/{run_id}/pause")
async def pause_training(run_id: str, db: Session = Depends(get_db)):
    """Pause a running training."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    if run.status != "running":
        raise ValidationError(f"Cannot pause run with status: {run.status}")
    
    training_manager.pause_training(run_id)
    
    run.status = "paused"
    run.paused_at = datetime.now()
    db.commit()
    
    return {"message": "Training paused", "run_id": run_id}


@router.post("/training/runs/{run_id}/resume")
async def resume_training(run_id: str, db: Session = Depends(get_db)):
    """Resume a paused training."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    if run.status != "paused":
        raise ValidationError(f"Cannot resume run with status: {run.status}")
    
    training_manager.resume_training(run_id)
    
    run.status = "running"
    run.paused_at = None
    db.commit()
    
    return {"message": "Training resumed", "run_id": run_id}


@router.post("/training/runs/{run_id}/stop")
async def stop_training(run_id: str, db: Session = Depends(get_db)):
    """Stop a running training."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    if run.status not in ["running", "paused"]:
        raise ValidationError(f"Cannot stop run with status: {run.status}")
    
    training_manager.stop_training(run_id)
    
    # Note: The process will update the status to "stopped" via callback
    return {"message": "Training stop requested", "run_id": run_id}


@router.get("/training/runs/{run_id}/stats")
async def get_training_stats(run_id: str):
    """Get real-time training statistics."""
    process = training_manager.get_process(run_id)
    
    if not process:
        raise NotFoundError(f"No active training process for run {run_id}")
    
    return process.get_stats()
@router.websocket("/ws/training/runs/{run_id}")
async def training_websocket(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time training updates."""
    from ..core.logging import get_logger
    from ..config import get_settings
    
    logger = get_logger(__name__)
    
    # Security check: Only accept WebSocket from localhost
    settings = get_settings()
    if not getattr(settings, 'allow_remote', False):
        client_host = websocket.client.host if websocket.client else None
        allowed_hosts = ["127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"]
        
        if client_host and client_host not in allowed_hosts:
            logger.warning(f"Rejected WebSocket connection from non-localhost: {client_host}")
            await websocket.close(code=1008, reason="Only localhost connections allowed")
            return
    
    await websocket.accept()
    logger.info(f"WebSocket connected for run {run_id}")
    
    try:
        # Wait a moment for training to initialize
        import asyncio
        await asyncio.sleep(0.3)
        
        # Send initial connection confirmation with current status
        process = training_manager.get_process(run_id)
        if process:
            initial_stats = process.get_stats()
            await websocket.send_json({
                "type": "training_update",
                "data": initial_stats,
                "timestamp": datetime.now().isoformat()
            })
            logger.info(f"Sent initial status for run {run_id}: {initial_stats['status']}")
        else:
            # Process not active yet, send connected message
            await websocket.send_json({
                "type": "connected",
                "run_id": run_id,
                "timestamp": datetime.now().isoformat()
            })
        
        last_sent_step = -1
        last_status = None
        update_count = 0
        
        while True:
            # Check for client messages (commands)
            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.5
                )
                
                # Handle client commands
                data = json.loads(message)
                if data.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("action") == "pause":
                    training_manager.pause_training(run_id)
                    logger.info(f"Pause requested via WebSocket for run {run_id}")
                elif data.get("action") == "resume":
                    training_manager.resume_training(run_id)
                    logger.info(f"Resume requested via WebSocket for run {run_id}")
                elif data.get("action") == "stop":
                    training_manager.stop_training(run_id)
                    logger.info(f"Stop requested via WebSocket for run {run_id}")
                    
            except asyncio.TimeoutError:
                # No message from client, continue
                pass
            except WebSocketDisconnect:
                break
            
            # Get training process and send updates
            process = training_manager.get_process(run_id)
            
            if process:
                stats = process.get_stats()
                
                # Send update if:
                # 1. Step changed
                # 2. Status changed
                # 3. Every 10th poll (heartbeat with data)
                should_send = (
                    stats["current_step"] != last_sent_step or
                    stats["status"] != last_status or
                    update_count % 4 == 0  # Every 2 seconds (0.5s * 4)
                )
                
                if should_send:
                    last_sent_step = stats["current_step"]
                    last_status = stats["status"]
                    
                    try:
                        await websocket.send_json({
                            "type": "training_update",
                            "data": stats,
                            "timestamp": datetime.now().isoformat()
                        })
                        logger.debug(f"Sent update for run {run_id}: step={stats['current_step']}, status={stats['status']}")
                    except Exception as e:
                        logger.warning(f"Failed to send WebSocket update for run {run_id}: {e}")
                        break
                    
                    # If training finished, break the loop
                    if stats["status"] in ["completed", "failed", "stopped"]:
                        logger.info(f"Training {run_id} finished with status: {stats['status']}")
                        break
                
                update_count += 1
            else:
                # Process no longer active, check database for final status
                logger.info(f"Training process {run_id} no longer active, checking database")
                
                # Get final status from database
                from ..models import get_db, TrainingRun
                db = next(get_db())
                try:
                    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                    if run:
                        final_stats = {
                            "run_id": run_id,
                            "status": run.status,
                            "current_step": run.current_step,
                            "total_steps": run.total_steps,
                            "best_loss": run.best_loss,
                            "best_step": run.best_step,
                            "current_loss": None,
                            "validation_loss": None,
                            "error_message": run.error_message,
                            "peak_memory_mb": 0,
                            "peak_cpu_percent": 0,
                            "tokens_per_second": 0,
                            "it_per_second": 0
                        }
                        await websocket.send_json({
                            "type": "training_update",
                            "data": final_stats,
                            "timestamp": datetime.now().isoformat()
                        })
                        logger.info(f"Sent final status from database for run {run_id}")
                except Exception as e:
                    logger.error(f"Error fetching final status from database: {e}")
                finally:
                    db.close()
                
                break
            
            # Small delay to prevent spam
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
    except Exception as e:
        logger.error(f"WebSocket error for run {run_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
        await websocket.close()


# Export endpoints
class ExportRequest(BaseModel):
    format: str  # "adapter", "fused", "gguf"


def _get_export_info(run: TrainingRun, format: str) -> Optional[Dict]:
    """Get export file info if it exists."""
    export_dir = f"{run.storage_path}/exports/{format}"
    
    if not os.path.exists(export_dir):
        return None
    
    files = os.listdir(export_dir)
    if not files:
        return None
    
    # Get the first (main) file
    file_path = os.path.join(export_dir, files[0])
    file_size = os.path.getsize(file_path)
    
    return {
        "path": file_path,
        "size_mb": round(file_size / (1024 * 1024), 2),
        "filename": files[0]
    }


@router.get("/training/runs/{run_id}/exports/status", response_model=ExportStatusResponse)
async def get_export_status(run_id: str, db: Session = Depends(get_db)):
    """Get export status for all formats."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    # Check actual file existence, not just flags
    adapter_info = _get_export_info(run, "adapter")
    fused_info = _get_export_info(run, "fused")
    gguf_info = _get_export_info(run, "gguf")
    
    return ExportStatusResponse(
        adapter=ExportStatus(
            available=adapter_info is not None,
            path=adapter_info["path"] if adapter_info else None,
            size_mb=adapter_info["size_mb"] if adapter_info else None,
            exported_at=datetime.fromtimestamp(os.path.getmtime(adapter_info["path"])).isoformat() if adapter_info else None
        ),
        fused=ExportStatus(
            available=fused_info is not None,
            path=fused_info["path"] if fused_info else None,
            size_mb=fused_info["size_mb"] if fused_info else None,
            exported_at=datetime.fromtimestamp(os.path.getmtime(fused_info["path"])).isoformat() if fused_info else None
        ),
        gguf=ExportStatus(
            available=gguf_info is not None,
            path=gguf_info["path"] if gguf_info else None,
            size_mb=gguf_info["size_mb"] if gguf_info else None,
            exported_at=datetime.fromtimestamp(os.path.getmtime(gguf_info["path"])).isoformat() if gguf_info else None
        )
    )


@router.post("/training/runs/{run_id}/exports")
async def export_model_endpoint(
    run_id: str,
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Export a trained model in various formats."""
    from ..ml.trainer import export_model
    
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    if run.status != "completed":
        raise ValidationError(f"Cannot export run with status: {run.status}. Must be completed.")
    
    # Check if already exported
    existing_export = _get_export_info(run, request.format)
    if existing_export:
        # Update flag if not set
        if request.format == "adapter" and not run.adapter_exported:
            run.adapter_exported = True
            db.commit()
        elif request.format == "fused" and not run.fused_exported:
            run.fused_exported = True
            db.commit()
        elif request.format == "gguf" and not run.gguf_exported:
            run.gguf_exported = True
            db.commit()
        
        return {
            "message": f"Model already exported as {request.format}",
            "run_id": run_id,
            "format": request.format,
            "path": existing_export["path"],
            "size_mb": existing_export["size_mb"],
            "already_exported": True
        }
    
    # Load config to get model info
    config_path = f"{run.storage_path}/config/training_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    base_model_id = config["base_model"]["huggingface_id"]
    adapter_path = f"{run.storage_path}/adapters.safetensors"
    hyperparameters = config.get("hyperparameters", {})
    
    # Determine output path
    export_dir = f"{run.storage_path}/exports/{request.format}"
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        # Run export
        output_path = await export_model(
            model_path=base_model_id,
            adapter_path=adapter_path,
            export_format=request.format,
            output_path=export_dir,
            hyperparameters=hyperparameters
        )
        
        # Update export status
        if request.format == "adapter":
            run.adapter_exported = True
        elif request.format == "fused":
            run.fused_exported = True
        elif request.format == "gguf":
            run.gguf_exported = True
        
        db.commit()
        
        # Get file info
        file_size_mb = None
        if os.path.exists(output_path):
            file_size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
        
        return {
            "message": f"Model exported as {request.format}",
            "run_id": run_id,
            "format": request.format,
            "path": output_path,
            "size_mb": file_size_mb,
            "already_exported": False
        }
        
    except Exception as e:
        raise TrainingError(f"Export failed: {str(e)}")


@router.get("/training/runs/{run_id}/exports/{format}/download")
async def download_export(
    run_id: str,
    format: str,
    db: Session = Depends(get_db)
):
    """Download an exported model."""
    from fastapi.responses import FileResponse
    
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    export_path = f"{run.storage_path}/exports/{format}"
    
    if not os.path.exists(export_path):
        raise NotFoundError(f"Export not found for format: {format}")
    
    # Find the exported file
    files = os.listdir(export_path)
    if not files:
        raise NotFoundError(f"No files in export directory")
    
    # Return the first file (assuming single export per format)
    file_path = os.path.join(export_path, files[0])
    
    return FileResponse(
        path=file_path,
        filename=f"{run.name}-{format}.{'safetensors' if format == 'adapter' else 'gguf' if format == 'gguf' else 'bin'}",
        media_type='application/octet-stream'
    )


@router.get("/training/runs/{run_id}/logs/detailed")
async def get_detailed_log(
    run_id: str,
    format: str = "json",
    db: Session = Depends(get_db)
):
    """Get detailed training log."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    log_path = f"{run.storage_path}/logs/detailed_training.log"
    
    if not os.path.exists(log_path):
        raise NotFoundError(f"Detailed log not found for run {run_id}")
    
    if format == "csv":
        # Return raw CSV
        from fastapi.responses import FileResponse
        return FileResponse(
            path=log_path,
            filename=f"{run.name}-detailed-log.csv",
            media_type='text/csv'
        )
    else:
        # Return as JSON
        import csv
        entries = []
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append({
                    "timestamp": row["timestamp"],
                    "step": int(row["step"]),
                    "loss": float(row["loss"]),
                    "learning_rate": float(row["learning_rate"]),
                    "tokens_per_second": float(row["tokens_per_second"]),
                    "it_per_second": float(row["it_per_second"]),
                    "cpu_percent": float(row["cpu_percent"]),
                    "memory_mb": float(row["memory_mb"]),
                    "peak_memory_mb": float(row["peak_memory_mb"])
                })
        
        return {
            "run_id": run_id,
            "total_entries": len(entries),
            "entries": entries
        }


@router.get("/training/runs/{run_id}/metrics")
async def get_training_metrics(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Get training metrics time-series data for visualization."""
    from ..models import TrainingMetric
    
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    # Fetch all metrics ordered by step
    metrics = db.query(TrainingMetric).filter(
        TrainingMetric.run_id == run_id
    ).order_by(TrainingMetric.step).all()
    
    return {
        "run_id": run_id,
        "total_metrics": len(metrics),
        "metrics": [
            {
                "step": m.step,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                "train_loss": m.train_loss,
                "eval_loss": m.eval_loss,
                "learning_rate": m.learning_rate,
                "gradient_norm": m.gradient_norm,
                "cpu_percent": m.cpu_percent,
                "memory_percent": m.memory_percent,
                "gpu_memory_used_mb": m.gpu_memory_used_mb,
                "samples_per_second": m.samples_per_second,
                "tokens_per_second": m.tokens_per_second,
                "elapsed_seconds": m.elapsed_seconds
            }
            for m in metrics
        ]
    }


# Update imports
from fastapi import BackgroundTasks
