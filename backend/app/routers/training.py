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
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from ..core.exceptions import NotFoundError, ValidationError, TrainingError
from ..core.logging import get_logger
from ..core import sanitize_dataset_content, assert_safe_path
from ..models import get_db, get_thread_safe_session, TrainingRun, TrainingPreset, ModelRegistry, Dataset, generate_uuid
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
    is_custom: bool = False
    is_downloaded: bool = False
    download_status: str = "missing"  # "complete" | "incomplete" | "missing"


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
    weight_decay: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_gradient_norm: Optional[float] = Field(default=None, gt=0)
    gradient_checkpointing: Optional[bool] = None
    num_lora_layers: Optional[int] = Field(default=None, ge=4, le=32)
    prompt_masking: Optional[bool] = None
    
    # PII Detection (Experimental)
    enable_pii_detection: Optional[bool] = Field(default=False, description="Enable experimental PII detection and anonymization")


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
    weight_decay: Optional[float] = None
    max_gradient_norm: Optional[float] = None
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
    status_message: Optional[str] = None
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
        weight_decay=getattr(run, 'weight_decay', None),
        max_gradient_norm=getattr(run, 'max_gradient_norm', None),
        gradient_checkpointing=run.gradient_checkpointing,
        num_lora_layers=run.num_lora_layers,
        prompt_masking=run.prompt_masking,
        validation_split_percent=getattr(run, 'validation_split_percent', 10),
        dataset=dataset_info
    )


class ValidateModelRequest(BaseModel):
    huggingface_id: str = Field(..., min_length=3, max_length=255)


class ValidateModelResponse(BaseModel):
    is_valid: bool
    message: str
    model_info: Optional[Dict[str, Any]] = None
    suggested_name: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

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
    Validate a custom HuggingFace model for MLX fine-tuning compatibility.
    Uses HuggingFace model_info with expand to extract full metadata:
    config.json (model_type, max_position_embeddings, num_hidden_layers),
    safetensors (accurate param count, quantization mix), card_data (license, base_model).
    Runs a verification checklist before allowing download.
    """
    raw_input = request.huggingface_id.strip()

    huggingface_id = raw_input
    for prefix in ('https://huggingface.co/', 'http://huggingface.co/', 'https://www.huggingface.co/', 'www.huggingface.co/', 'huggingface.co/'):
        if raw_input.startswith(prefix):
            huggingface_id = raw_input[len(prefix):].rstrip('/')
            logger.info(f"Extracted model ID '{huggingface_id}' from URL '{raw_input}'")
            break
    if '/' in huggingface_id:
        parts = huggingface_id.split('/')
        if len(parts) > 2 and parts[-1] == '':
            huggingface_id = '/'.join(parts[:2])
        elif len(parts) > 2:
            if any(p.startswith('tree') or p.startswith('blob') or p.startswith('resolve') or p.startswith('models') for p in parts[2:]):
                huggingface_id = '/'.join(parts[:2])

    if not re.match(r'^[\w\-\.]+(/[\w\-\.]+)?$', huggingface_id):
        return ValidateModelResponse(
            is_valid=False,
            message="Invalid HuggingFace model ID format. Expected: 'organization/model-name' or 'model-name'",
            errors=["Invalid format"]
        )

    existing = db.query(ModelRegistry).filter(
        ModelRegistry.huggingface_id == huggingface_id
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
                "context_length": existing.context_length,
                "already_exists": True,
            },
            suggested_name=existing.name,
            warnings=["This model is already in your registry"]
        )

    try:
        from huggingface_hub import model_info as hf_model_info
        from ..core.model_architectures import (
            resolve_architecture, get_arch_config, get_size_category,
            get_param_count_from_name, detect_architecture_from_id,
            is_mlx_supported, ARCH_MAP, ARCHITECTURE_CONFIG,
            FALLBACK_PARAM_COUNT, CONTEXT_LENGTH_CAP, DEFAULT_CONTEXT_LENGTH,
            DEFAULT_ARCH,
        )

        settings = get_settings()

        info = hf_model_info(
            huggingface_id,
            expand=["config", "safetensors", "cardData"],
            token=settings.hf_token
        )

        download_size_gb = 0.0
        file_siblings = []
        try:
            info_files = hf_model_info(
                huggingface_id,
                files_metadata=True,
                token=settings.hf_token
            )
            file_siblings = [s.rfilename for s in (info_files.siblings or [])]
            for s in (info_files.siblings or []):
                if hasattr(s, 'size') and s.size:
                    download_size_gb += s.size
        except Exception:
            pass

        warnings = []
        errors = []

        tags = info.tags or getattr(info_files, 'tags', None) or []
        siblings = [s.rfilename for s in (info.siblings or [])] or file_siblings
        pipeline_tag = info.pipeline_tag
        is_gated = info.gated or False
        library_name = info.library_name
        downloads = info.downloads or 0

        is_mlx_formatted = (
            any(t.lower().replace("-", "") in ('mlx', 'mlxcommunity', '4bit', '8bit') for t in tags)
            or huggingface_id.lower().startswith("mlx-community/")
        )

        has_safetensors = any(s.endswith(".safetensors") for s in siblings)
        has_tokenizer = any(s in siblings for s in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"])
        has_chat_template = any(s in siblings for s in ["tokenizer_config.json", "chat_template.jinja"])

        raw_model_type = None
        num_hidden_layers = None
        context_length = DEFAULT_CONTEXT_LENGTH
        hidden_size = None

        if info.config:
            raw_model_type = info.config.get("model_type")
            ctx = info.config.get("max_position_embeddings")
            if ctx and ctx > 0:
                context_length = min(ctx, CONTEXT_LENGTH_CAP)
            num_hidden_layers = info.config.get("num_hidden_layers")
            hidden_size = info.config.get("hidden_size")

        if raw_model_type:
            architecture = resolve_architecture(raw_model_type)
            if raw_model_type.lower().replace("-", "").replace("_", "") not in ARCH_MAP and architecture == DEFAULT_ARCH:
                warnings.append(f"Model type '{raw_model_type}' is not in the known architecture map — assuming {DEFAULT_ARCH}-compatible. May need verification.")
        else:
            raw = detect_architecture_from_id(huggingface_id, tags)
            architecture = resolve_architecture(raw)
            if not raw:
                warnings.append(f"Could not detect architecture from config or model ID — defaulting to {DEFAULT_ARCH}")

        param_count = 0
        quantization = None

        if info.safetensors:
            param_count = info.safetensors.total or 0
            if info.safetensors.parameters:
                quantization = dict(info.safetensors.parameters)

        if param_count == 0:
            param_count = get_param_count_from_name(huggingface_id)

        if param_count == 0:
            param_count = FALLBACK_PARAM_COUNT
            warnings.append(f"Parameter count unknown — assuming {FALLBACK_PARAM_COUNT // 1_000_000_000}B. Actual count will be determined during download.")

        base_model = None
        license_name = None
        languages = None
        if info.card_data:
            base_model = getattr(info.card_data, 'base_model', None) or (info.card_data.get('base_model') if isinstance(info.card_data, dict) else None)
            license_name = getattr(info.card_data, 'license', None) or (info.card_data.get('license') if isinstance(info.card_data, dict) else None)
            languages = getattr(info.card_data, 'language', None) or (info.card_data.get('language') if isinstance(info.card_data, dict) else None)

        download_size_gb = round(download_size_gb / (1024 ** 3), 2) if download_size_gb else round(param_count * 0.5 / (1024**3), 2)

        # === VERIFICATION CHECKLIST ===
        # Hard blocks (errors) — only for models that CANNOT work with MLX

        if pipeline_tag and pipeline_tag not in ("text-generation", "text2text-generation", None):
            errors.append(f"Model pipeline is '{pipeline_tag}', not text-generation. This model may not be suitable for fine-tuning.")

        if not has_safetensors:
            errors.append("No safetensors files found. MLX requires safetensors format.")

        if not has_tokenizer:
            errors.append("No tokenizer files found. A tokenizer is required for training.")

        if is_gated and not settings.hf_token:
            errors.append("This is a gated model. Set EDUKAAI_HF_TOKEN to access it.")

        if raw_model_type and not is_mlx_supported(raw_model_type):
            arch_config_raw = get_arch_config(architecture)
            mlx_module = arch_config_raw.get("mlx_module", architecture)
            errors.append(
                f"Model type '{raw_model_type}' maps to MLX module '{mlx_module}' which is not available. "
                f"This model architecture is not supported by your MLX installation."
            )

        # Soft warnings — model works but user should be aware

        arch_config = get_arch_config(architecture)

        if not arch_config.get("is_supported", True):
            warnings.append(
                f"Architecture '{architecture}' is supported by MLX but has not been fully tested "
                f"for LoRA fine-tuning in EdukaAI Studio. Training may produce suboptimal results."
            )

        if arch_config.get("is_moe", False):
            warnings.append(
                "This is a Mixture-of-Experts model. LoRA will target attention layers only, "
                "not individual expert MLPs. This is safe but may limit fine-tuning effectiveness."
            )

        min_ram = arch_config.get("min_ram_gb", 8)
        try:
            import subprocess
            ram_bytes = int(subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            ).stdout.strip())
            ram_gb = ram_bytes / (1024 ** 3)
            if ram_gb < min_ram:
                warnings.append(
                    f"This model requires at least {min_ram}GB RAM for training. "
                    f"Your Mac has {ram_gb:.0f}GB."
                )
        except Exception:
            pass

        if not has_chat_template:
            warnings.append("No chat template found — will use architecture-specific fallback for inference.")

        if not is_mlx_formatted:
            warnings.append("This model doesn't appear to be MLX-formatted. Consider using a model from mlx-community for best results on Apple Silicon.")

        is_valid = len(errors) == 0

        parts = huggingface_id.split('/')
        base_name = parts[-1] if len(parts) > 1 else huggingface_id
        suggested_name = base_name.replace('-', ' ').replace('_', ' ').title()

        size_cat = get_size_category(param_count)

        model_info_dict = {
            "huggingface_id": huggingface_id,
            "architecture": architecture,
            "raw_model_type": raw_model_type,
            "parameter_count": param_count,
            "context_length": context_length,
            "num_hidden_layers": num_hidden_layers,
            "hidden_size": hidden_size,
            "quantization": quantization,
            "is_mlx_formatted": is_mlx_formatted,
            "has_safetensors": has_safetensors,
            "has_tokenizer": has_tokenizer,
            "has_chat_template": has_chat_template,
            "is_gated": is_gated,
            "base_model": base_model,
            "license": license_name,
            "languages": languages,
            "pipeline_tag": pipeline_tag,
            "library_name": library_name,
            "downloads": downloads,
            "tags": tags[:10],
            "estimated_download_size_gb": download_size_gb,
            "size_category": size_cat,
            "lora_target_modules": arch_config["lora_keys"],
            "stop_strings": arch_config["stop_strings"],
            "eos_token": arch_config["eos_token"],
            "is_moe": arch_config.get("is_moe", False),
            "is_supported": arch_config.get("is_supported", True),
            "min_ram_gb": arch_config.get("min_ram_gb", 8),
        }

        if is_valid:
            msg = f"Model verified: {huggingface_id}. Architecture: {architecture}, ~{param_count/1e9:.1f}B params, context: {context_length}"
        else:
            msg = f"Model not compatible: {huggingface_id}. Issues: {'; '.join(errors)}"

        return ValidateModelResponse(
            is_valid=is_valid,
            message=msg,
            model_info=model_info_dict,
            suggested_name=suggested_name,
            warnings=warnings,
            errors=errors
        )

    except Exception as e:
        logger.error(f"Error validating model {huggingface_id}: {e}")
        return ValidateModelResponse(
            is_valid=False,
            message=f"Could not validate model '{huggingface_id}': {str(e)}",
            errors=[f"HuggingFace API error: {str(e)}"]
        )


@router.post("/base-models/custom", response_model=BaseModelResponse)
async def add_custom_model(
    request: ValidateModelRequest,
    db: Session = Depends(get_db)
):
    """
    Add a custom model to the database after validation.
    Stores full HuggingFace metadata in mlx_config for fine-tuning capability.
    """
    validation = await validate_custom_model(request, db)

    if not validation.is_valid:
        raise ValidationError(validation.message)

    huggingface_id = request.huggingface_id.strip()
    for prefix in ('https://huggingface.co/', 'http://huggingface.co/', 'https://www.huggingface.co/', 'www.huggingface.co/', 'huggingface.co/'):
        if huggingface_id.startswith(prefix):
            huggingface_id = huggingface_id[len(prefix):].rstrip('/')
            break

    existing = db.query(ModelRegistry).filter(
        ModelRegistry.huggingface_id == huggingface_id
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

    from ..core.model_architectures import resolve_architecture, get_arch_config, get_size_category, FALLBACK_PARAM_COUNT, DEFAULT_CONTEXT_LENGTH, DEFAULT_ARCH

    info = validation.model_info
    model_id = generate_uuid()

    raw_arch = info.get("architecture", DEFAULT_ARCH)
    architecture = resolve_architecture(raw_arch)
    arch_config = get_arch_config(architecture)
    param_count = info.get("parameter_count", FALLBACK_PARAM_COUNT)
    size_cat = get_size_category(param_count)
    context_length = info.get("context_length", DEFAULT_CONTEXT_LENGTH)

    model = ModelRegistry(
        id=model_id,
        huggingface_id=huggingface_id,
        name=validation.suggested_name or huggingface_id,
        architecture=architecture,
        parameter_count=param_count,
        context_length=context_length,
        is_active=True,
        is_curated=False,
        mlx_config={
            "is_custom": True,
            "supports_lora": True,
            "lora_target_modules": arch_config["lora_keys"],
            "recommended_max_seq_length": context_length,
            "model_family": architecture,
            "size_category": size_cat,
            "stop_strings": arch_config["stop_strings"],
            "eos_token": arch_config["eos_token"],
            "chat_template_fallback": arch_config["chat_template_fallback"],
            "num_hidden_layers": info.get("num_hidden_layers"),
            "hidden_size": info.get("hidden_size"),
            "quantization": info.get("quantization"),
            "is_mlx_formatted": info.get("is_mlx_formatted", False),
            "has_chat_template": info.get("has_chat_template", False),
            "is_gated": info.get("is_gated", False),
            "is_moe": info.get("is_moe", arch_config.get("is_moe", False)),
            "is_supported": info.get("is_supported", arch_config.get("is_supported", True)),
            "min_ram_gb": info.get("min_ram_gb", arch_config.get("min_ram_gb", 8)),
            "base_model": info.get("base_model"),
            "license": info.get("license"),
            "languages": info.get("languages"),
            "downloads": info.get("downloads", 0),
            "estimated_download_size_gb": info.get("estimated_download_size_gb", 0),
            "validation_warnings": validation.warnings,
            "raw_model_type": info.get("raw_model_type"),
            "pipeline_tag": info.get("pipeline_tag"),
            "library_name": info.get("library_name"),
            "added_at": datetime.now().isoformat(),
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
        is_custom=True
    )


@router.get("/base-models/{model_id}/recommended-config")
async def get_recommended_config(model_id: str, db: Session = Depends(get_db)):
    """Get architecture-aware recommended training config for a model."""
    from ..core.model_architectures import get_arch_config, get_size_category
    
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise NotFoundError(f"Model {model_id} not found")
    
    architecture = model.architecture or "qwen2"
    arch_config = get_arch_config(architecture)
    size_cat = get_size_category(model.parameter_count or 3_000_000_000)
    
    preset = arch_config.get("recommended_presets", {}).get(size_cat)
    if not preset:
        for cat in ["small", "medium", "tiny"]:
            preset = arch_config.get("recommended_presets", {}).get(cat)
            if preset:
                break
    
    if not preset:
        preset = {"lora_rank": 8, "lora_layers": 16, "learning_rate": 5e-5, "batch_size": 4}
    
    return {
        "model_id": model_id,
        "architecture": architecture,
        "size_category": size_cat,
        "context_length": model.context_length or 4096,
        "recommended": {
            "lora_rank": preset.get("lora_rank", 8),
            "num_lora_layers": preset.get("lora_layers", 16),
            "learning_rate": preset.get("learning_rate", 5e-5),
            "batch_size": preset.get("batch_size", 4),
            "max_seq_length": min(model.context_length or 4096, 4096),
            "gradient_checkpointing": preset.get("gradient_checkpointing", False),
        },
        "lora_target_modules": arch_config.get("lora_keys", []),
    }


@router.get("/base-models", response_model=List[BaseModelResponse])
async def list_base_models(db: Session = Depends(get_db)):
    """List all active base models."""
    import asyncio
    from ..core.model_architectures import _is_model_complete_sync, _get_hf_cache_root
    from pathlib import Path

    models = db.query(ModelRegistry).filter(
        ModelRegistry.is_active == True
    ).order_by(ModelRegistry.parameter_count).all()

    def _check_status(hf_id: str):
        """Return (is_complete, has_any_files)."""
        is_complete = _is_model_complete_sync(hf_id)
        if is_complete:
            return True, True
        # Check if ANY snapshot dir exists (metadata without weights)
        cache_root = _get_hf_cache_root()
        model_cache = cache_root / f"models--{hf_id.replace('/', '--')}"
        snapshots = model_cache / "snapshots"
        has_any = snapshots.exists() and any(d.is_dir() for d in snapshots.iterdir())
        return False, has_any

    async def _check_one(m):
        is_complete, has_any = await asyncio.to_thread(_check_status, m.huggingface_id)
        return m, is_complete, has_any

    checked = await asyncio.gather(*[_check_one(m) for m in models])

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
            is_custom=not m.is_curated,
            is_downloaded=is_complete,
            download_status=("complete" if is_complete else ("incomplete" if has_any else "missing")),
        )
        for m, is_complete, has_any in checked
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
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    
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


@router.get("/base-models/{model_id}/download-status")
async def get_model_download_status(model_id: str, db: Session = Depends(get_db)):
    """
    Check whether a model's weight files are fully present in the local cache.
    Returns cache-health metadata so the UI can show 'Download incomplete' badges.
    """
    from ..core.model_architectures import _is_model_complete_sync, _get_hf_cache_root
    from pathlib import Path

    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise NotFoundError(f"Model {model_id} not found")

    is_complete = await asyncio.to_thread(_is_model_complete_sync, model.huggingface_id)

    cache_root = _get_hf_cache_root()
    model_cache = cache_root / f"models--{model.huggingface_id.replace('/', '--')}"
    snapshots = model_cache / "snapshots"

    snapshot_dir = None
    if snapshots.exists():
        for snapshot in snapshots.iterdir():
            if snapshot.is_dir():
                snapshot_dir = str(snapshot)
                break

    return {
        "model_id": model_id,
        "huggingface_id": model.huggingface_id,
        "is_complete": is_complete,
        "cache_snapshot_dir": snapshot_dir,
        "cache_root": str(cache_root),
        "message": (
            "Model weights fully cached."
            if is_complete
            else "Model metadata cached but weight files (*.safetensors) are missing. "
                 "Please delete and re-add the model to trigger a fresh download."
        ),
    }


@router.post("/training/runs", response_model=TrainingRunResponse)
async def create_training_run(
    request: CreateTrainingRunRequest,
    db: Session = Depends(get_db)
):
    """Create a new training run with configuration."""
    
    # Log PII detection status (experimental feature)
    if request.enable_pii_detection:
        logger.info(f"[EXPERIMENTAL] PII detection enabled for training run. Dataset will be scanned for PII.")
    
    # Validate dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == request.training_dataset_id).first()
    if not dataset:
        raise NotFoundError(f"Training dataset {request.training_dataset_id} not found")
    
    # Validate base model exists
    base_model = db.query(ModelRegistry).filter(ModelRegistry.id == request.base_model_id).first()
    if not base_model:
        raise NotFoundError(f"Base model {request.base_model_id} not found")
    
    if request.max_seq_length > (base_model.context_length or 4096):
        raise ValidationError(
            f"max_seq_length ({request.max_seq_length}) exceeds model context length "
            f"({base_model.context_length or 4096})"
        )
    
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
    
    # Validate dataset file path before accessing it
    if dataset.file_path:
        assert_safe_path(dataset.file_path, ["./storage/datasets", str(Path("./storage/datasets").resolve())])

    # Apply PII detection if enabled (experimental feature)
    if request.enable_pii_detection:
        logger.info(f"[EXPERIMENTAL] Applying PII detection to dataset for run {run_id}")
        try:
            # Read original dataset
            with open(dataset.file_path, 'r') as f:
                original_content = f.read()
            
            # Apply PII sanitization
            sanitized_content, warnings, anonymization_report = sanitize_dataset_content(original_content)
            
            # Write sanitized version to run directory
            with open(training_data_path, 'w') as f:
                f.write(sanitized_content)
            
            logger.info(f"[EXPERIMENTAL] PII detection complete: {anonymization_report.get('total_replacements', 0)} replacements made")
            
            # Store report in config for reference
            pii_report = anonymization_report
        except Exception as e:
            logger.error(f"[EXPERIMENTAL] PII detection failed: {e}")
            # Fallback: copy original
            shutil.copy(dataset.file_path, training_data_path)
            pii_report = {"error": str(e), "skipped": True}
    else:
        # Normal flow: just copy the dataset
        shutil.copy(dataset.file_path, training_data_path)
        pii_report = {"skipped": True, "reason": "PII detection not enabled"}
    
    # Handle validation dataset
    validation_data_path = None
    validation_dataset_id = None
    
    if request.validation_dataset_id:
        # Use separate validation file
        val_dataset = db.query(Dataset).filter(Dataset.id == request.validation_dataset_id).first()
        if not val_dataset:
            raise NotFoundError(f"Validation dataset {request.validation_dataset_id} not found")
        if val_dataset.file_path:
            assert_safe_path(val_dataset.file_path, ["./storage/datasets", str(Path("./storage/datasets").resolve())])
        validation_data_path = f"{storage_path}/data/validation.jsonl"
        
        # Apply PII detection to validation dataset if enabled
        if request.enable_pii_detection:
            try:
                with open(val_dataset.file_path, 'r') as f:
                    val_content = f.read()
                val_sanitized, _, _ = sanitize_dataset_content(val_content)
                with open(validation_data_path, 'w') as f:
                    f.write(val_sanitized)
            except Exception as e:
                logger.error(f"[EXPERIMENTAL] PII detection failed for validation set: {e}")
                shutil.copy(val_dataset.file_path, validation_data_path)
        else:
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
            },
            "enable_pii_detection": request.enable_pii_detection or False
        },
        "hyperparameters": {
            "steps": request.steps if request.steps is not None else preset.steps,
            "learning_rate": request.learning_rate if request.learning_rate is not None else preset.learning_rate,
            "lora_rank": request.lora_rank if request.lora_rank is not None else preset.lora_rank,
            "lora_alpha": request.lora_alpha if request.lora_alpha is not None else preset.lora_alpha,
            "lora_dropout": request.lora_dropout if request.lora_dropout is not None else preset.lora_dropout,
            "batch_size": request.batch_size if request.batch_size is not None else preset.batch_size,
            "warmup_steps": request.warmup_steps if request.warmup_steps is not None else preset.warmup_steps,
            "gradient_accumulation_steps": request.gradient_accumulation_steps if request.gradient_accumulation_steps is not None else preset.gradient_accumulation_steps,
            "early_stopping_patience": request.early_stopping_patience if request.early_stopping_patience is not None else preset.early_stopping_patience,
            "weight_decay": request.weight_decay,
            "max_gradient_norm": request.max_gradient_norm,
            "max_seq_length": request.max_seq_length,
            "gradient_checkpointing": request.gradient_checkpointing if request.gradient_checkpointing is not None else preset.gradient_checkpointing,
            "num_lora_layers": request.num_lora_layers or preset.num_lora_layers,
            "prompt_masking": request.prompt_masking if request.prompt_masking is not None else preset.prompt_masking,
            "validation_split_percent": validation_split_percent,
            "architecture": base_model.architecture or "qwen2",
            "lora_target_modules": (base_model.mlx_config or {}).get("lora_target_modules")
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
                   f"LoRA rank {request.lora_rank if request.lora_rank is not None else preset.lora_rank}, {request.steps if request.steps is not None else preset.steps} steps. "
                   f"Base model: {base_model.huggingface_id}",
        
        # Hyperparameters
        steps=request.steps if request.steps is not None else preset.steps,
        learning_rate=request.learning_rate if request.learning_rate is not None else preset.learning_rate,
        lora_rank=request.lora_rank if request.lora_rank is not None else preset.lora_rank,
        lora_alpha=request.lora_alpha if request.lora_alpha is not None else preset.lora_alpha,
        lora_dropout=request.lora_dropout if request.lora_dropout is not None else preset.lora_dropout,
        batch_size=request.batch_size if request.batch_size is not None else preset.batch_size,
        warmup_steps=request.warmup_steps if request.warmup_steps is not None else preset.warmup_steps,
        gradient_accumulation_steps=request.gradient_accumulation_steps if request.gradient_accumulation_steps is not None else preset.gradient_accumulation_steps,
        early_stopping_patience=request.early_stopping_patience if request.early_stopping_patience is not None else preset.early_stopping_patience,
        weight_decay=request.weight_decay,
        max_gradient_norm=request.max_gradient_norm,
        max_seq_length=request.max_seq_length,
        gradient_checkpointing=request.gradient_checkpointing if request.gradient_checkpointing is not None else preset.gradient_checkpointing,
        num_lora_layers=request.num_lora_layers if request.num_lora_layers is not None else preset.num_lora_layers,
        prompt_masking=request.prompt_masking if request.prompt_masking is not None else preset.prompt_masking,
        validation_split_percent=validation_split_percent if validation_split_percent is not None else request.validation_split_percent,
        
        # Resource limits
        cpu_cores_limit=request.cpu_cores_limit,
        gpu_memory_limit_gb=request.gpu_memory_limit_gb,
        ram_limit_gb=request.ram_limit_gb,
        
        total_steps=request.steps if request.steps is not None else preset.steps,
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
        status_message=run.status_message or "",
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
            status_message=r.status_message or "",
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
        status_message=run.status_message or "",
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
    if storage_path:
        assert_safe_path(storage_path, ["./storage/runs", str(Path("./storage/runs").resolve())])
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
    if storage_path:
        assert_safe_path(storage_path, ["./storage/runs", str(Path("./storage/runs").resolve())])
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
    
    if run.status in ("running", "paused"):
        raise ValidationError(f"Cannot delete a {run.status} training run. Stop it first.")
    
    # Stop any active training process
    process = training_manager.get_process(run_id)
    if process:
        training_manager.stop_training(run_id)
        logger.info(f"Stopped active training process for run {run_id} before deletion")
    
    # Delete storage directory
    try:
        if run.storage_path:
            assert_safe_path(run.storage_path, ["./storage/runs", str(Path("./storage/runs").resolve())])
            if os.path.exists(run.storage_path):
                shutil.rmtree(run.storage_path)
    except Exception as e:
        # Log error but continue with DB deletion
        logger.warning(f"Failed to delete storage for run {run_id}: {e}")
    
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
        status_message=run.status_message or "",
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
    model_id = raw_model_id
    for prefix in ('https://huggingface.co/', 'http://huggingface.co/', 'https://www.huggingface.co/', 'www.huggingface.co/', 'huggingface.co/'):
        if raw_model_id.startswith(prefix):
            model_id = raw_model_id[len(prefix):].rstrip('/')
            logger.info(f"Extracted model ID '{model_id}' from URL '{raw_model_id}'")
            break
    else:
        model_id = raw_model_id
    
    hp = saved_config.get("hyperparameters", {})
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
        weight_decay=run.weight_decay,
        max_gradient_norm=run.max_gradient_norm,
        architecture=hp.get("architecture", run.base_model.architecture if run.base_model else "qwen2"),
        lora_target_modules=hp.get("lora_target_modules"),
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
        from ..models import get_thread_safe_session, TrainingMetric
        db = get_thread_safe_session()
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
        from ..models import get_thread_safe_session
        db = get_thread_safe_session()
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
        """Mark run as failed or stopped."""
        from ..models import get_thread_safe_session
        db = get_thread_safe_session()
        try:
            run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run:
                if error_msg.startswith("stopped:"):
                    run.status = "stopped"
                    run.error_message = error_msg[len("stopped:"):]
                else:
                    run.status = "failed"
                    run.error_message = error_msg
                run.completed_at = datetime.now()
                db.commit()
                logger.info(f"Training run {run_id} marked as {run.status}: {error_msg}")
                training_manager.cleanup(run_id)
        except Exception as e:
            logger.error(f"Error in on_error: {e}", exc_info=True)
        finally:
            db.close()
    
    def on_status_change(status: str, message: str):
        """Handle status changes like downloading, loading_model, etc."""
        logger.info(f"Training {run_id} status changed: {status} - {message}")
        try:
            run_obj = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if run_obj:
                run_obj.status = status
                run_obj.status_message = message or ""
                if status == "running" and not run_obj.started_at:
                    run_obj.started_at = datetime.now()
                db.commit()
        except Exception as e:
            logger.warning(f"Failed to persist status change for {run_id}: {e}")
    
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
    
    if run.status not in ["running", "paused", "downloading", "loading_model"]:
        raise ValidationError(f"Cannot stop run with status: {run.status}")
    
    training_manager.stop_training(run_id)
    
    run.status = "stopped"
    run.status_message = "Stopped by user"
    run.completed_at = datetime.now()
    db.commit()
    
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
        last_sent_message = ""
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
                    stats.get("status_message", "") != last_sent_message or
                    update_count % 4 == 0
                )
                
                if should_send:
                    last_sent_step = stats["current_step"]
                    last_status = stats["status"]
                    last_sent_message = stats.get("status_message", "")
                    
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
                db = get_thread_safe_session()
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
ALLOWED_EXPORT_FORMATS = {"adapter", "fused", "gguf"}

class ExportRequest(BaseModel):
    format: str  # "adapter", "fused", "gguf"

    @field_validator('format')
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in ALLOWED_EXPORT_FORMATS:
            raise ValueError(f"Invalid format. Allowed: {', '.join(sorted(ALLOWED_EXPORT_FORMATS))}")
        return v


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
    
    # Get architecture-specific LoRA target modules from base model
    base_model_db = db.query(ModelRegistry).filter(ModelRegistry.id == run.base_model_id).first()
    lora_target_modules = None
    if base_model_db and base_model_db.mlx_config:
        lora_target_modules = base_model_db.mlx_config.get("lora_target_modules")
    
    # Resolve base model to local cache path if available
    from ..core.model_architectures import get_cached_snapshot_path_sync
    cached_snapshot = get_cached_snapshot_path_sync(base_model_id)
    model_path = cached_snapshot if cached_snapshot else base_model_id
    if cached_snapshot:
        logger.info(f"Using locally cached model for export: {model_path}")
    else:
        logger.info(f"Using HuggingFace model ID for export: {model_path}")
    
    # Determine output path
    export_dir = f"{run.storage_path}/exports/{request.format}"
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        # Run export
        try:
            output_path = await export_model(
                model_path=model_path,
                adapter_path=adapter_path,
                export_format=request.format,
                output_path=export_dir,
                hyperparameters=hyperparameters,
                lora_target_modules=lora_target_modules
            )
        except NotImplementedError as e:
            raise ExportError(str(e))
        
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
    
    if format not in ALLOWED_EXPORT_FORMATS:
        raise ValidationError(f"Invalid export format '{format}'. Allowed: {', '.join(sorted(ALLOWED_EXPORT_FORMATS))}")
    
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if not run:
        raise NotFoundError(f"Training run {run_id} not found")
    
    export_path = os.path.normpath(f"{run.storage_path}/exports/{format}")
    if not export_path.startswith(os.path.normpath(run.storage_path)):
        raise ValidationError("Invalid export path")
    
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

