"""
Chat router - Dual Chat for model comparison with real inference
Security: Implements comprehensive input validation and sanitization
"""

import os
import json
import asyncio
import re
import html
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from ..core.exceptions import NotFoundError, ValidationError
from ..core.logging import get_logger
from ..models import get_db, TrainingRun
from ..ml.trainer import load_model_for_inference, generate_response
from mlx_lm import generate as mlx_generate

router = APIRouter()
logger = get_logger(__name__)

# In-memory cache for loaded models (per-session)
loaded_models: Dict[str, tuple] = {}

# Security: Blocked patterns for XSS and injection prevention
BLOCKED_PATTERNS = [
    r'<script[^>]*>.*?</script>',
    r'<iframe[^>]*>.*?</iframe>',
    r'<object[^>]*>.*?</object>',
    r'<embed[^>]*>.*?</embed>',
    r'javascript:',
    r'data:text/html',
    r'on\w+\s*=',
    r'<img[^>]+on\w+\s*=',
    r'<[^>]+on\w+\s*=',
]

# Security: Prompt injection patterns to block
PROMPT_INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)',
    r'disregard\s+(all\s+)?(system\s+)?prompts?',
    r'new\s+system\s+prompt',
    r'override\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)',
    r'forget\s+(all\s+)?(previous|prior)\s+(instructions?|context)',
    r'system\s*:\s*you\s+are\s+now',
    r'you\s+are\s+now\s+(in\s+)?\w+\s+mode',
]

BLOCKED_COMPILED = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in BLOCKED_PATTERNS]
INJECTION_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS]


def sanitize_input(text: str, max_length: int = 4000, allow_html: bool = False) -> str:
    """
    Comprehensive input sanitization to prevent XSS and injection attacks.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length
        allow_html: If False, strips all HTML tags
    
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Check length
    if len(text) > max_length:
        text = text[:max_length]
    
    # Block dangerous patterns
    for pattern in BLOCKED_COMPILED:
        text = pattern.sub('', text)
    
    # If HTML not allowed, escape it
    if not allow_html:
        text = html.escape(text)
    
    return text.strip()


def validate_system_prompt(prompt: str) -> tuple[bool, str]:
    """
    Validate system prompt for injection attempts.
    
    Returns:
        (is_valid, error_message)
    """
    if not prompt:
        return True, ""
    
    # Check length
    if len(prompt) > 2000:
        return False, "System prompt exceeds maximum length of 2000 characters"
    
    # Check for injection patterns
    for pattern in INJECTION_COMPILED:
        if pattern.search(prompt):
            return False, "System prompt contains potentially dangerous patterns"
    
    return True, ""


def validate_message(message: str) -> tuple[bool, str]:
    """
    Validate user message for security.
    
    Returns:
        (is_valid, error_message)
    """
    if not message:
        return False, "Message cannot be empty"
    
    if len(message) > 4000:
        return False, "Message exceeds maximum length of 4000 characters"
    
    # Check for blocked patterns
    for pattern in BLOCKED_COMPILED:
        if pattern.search(message):
            return False, "Message contains prohibited content"
    
    return True, ""


class LoadModelRequest(BaseModel):
    run_id: str
    use_fine_tuned: bool = True


class GenerateRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    system_prompt: str = Field(default="You are a helpful assistant.", max_length=2000)
    max_tokens: int = Field(default=512, ge=16, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    
    @validator('message')
    def validate_message_content(cls, v):
        is_valid, error = validate_message(v)
        if not is_valid:
            raise ValueError(error)
        return sanitize_input(v, max_length=4000, allow_html=False)
    
    @validator('system_prompt')
    def validate_system_content(cls, v):
        is_valid, error = validate_system_prompt(v)
        if not is_valid:
            raise ValueError(error)
        return sanitize_input(v, max_length=2000, allow_html=False)


class GenerateResponse(BaseModel):
    text: str
    tokens: int
    response_time: float
    tokens_per_second: float


@router.post("/chat/load-model")
async def load_model(
    request: LoadModelRequest,
    db: Session = Depends(get_db)
):
    """Load a model (base or fine-tuned) for inference."""
    
    run = db.query(TrainingRun).filter(TrainingRun.id == request.run_id).first()
    if not run:
        raise NotFoundError(f"Training run {request.run_id} not found")
    
    # Load config
    config_path = f"{run.storage_path}/config/training_config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    base_model_id = config["base_model"]["huggingface_id"]
    
    # Check if model was downloaded during training (custom models)
    # Use downloaded path if available, otherwise use HF ID
    # Priority: 1) Custom download dir, 2) HF cache dir, 3) HF ID (will download)
    from pathlib import Path
    
    # 1. Check our custom download directory
    download_dir = Path(run.storage_path).parent / "downloaded_models" / base_model_id.replace("/", "--")
    if download_dir.exists() and any(download_dir.glob("model*.safetensors")):
        model_path = str(download_dir)
        logger.info(f"Using downloaded model for chat: {model_path}")
    else:
        # 2. Check HF cache directory
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_name = base_model_id.replace("/", "--")
        model_cache_path = cache_dir / f"models--{model_name}"
        
        if model_cache_path.exists():
            snapshots_dir = model_cache_path / "snapshots"
            if snapshots_dir.exists():
                # Find the first snapshot with required files
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir() and any(snapshot.glob("*.safetensors")):
                        model_path = str(snapshot)
                        logger.info(f"Using HF cached model for chat: {model_path}")
                        break
                else:
                    model_path = base_model_id
                    logger.info(f"Using HuggingFace model ID for chat: {model_path}")
            else:
                model_path = base_model_id
                logger.info(f"Using HuggingFace model ID for chat: {model_path}")
        else:
            model_path = base_model_id
            logger.info(f"Using HuggingFace model ID for chat: {model_path}")
    
    if request.use_fine_tuned:
        # Look for adapter file directly in run directory
        adapter_file = f"{run.storage_path}/adapters.safetensors"
        if not os.path.exists(adapter_file):
            # Fallback to numbered checkpoint
            adapter_file = f"{run.storage_path}/0000100_adapters.safetensors"
        if not os.path.exists(adapter_file):
            raise NotFoundError(f"Fine-tuned adapter not found at {adapter_file}")
        adapter_path = run.storage_path  # Directory containing the adapter file
    else:
        adapter_path = None
    
    try:
        # Load model
        model, tokenizer = await load_model_for_inference(
            model_path=model_path,
            adapter_path=adapter_path
        )
        
        # Cache the loaded model
        cache_key = f"{request.run_id}_{request.use_fine_tuned}"
        loaded_models[cache_key] = (model, tokenizer, run.name)
        
        return {
            "message": "Model loaded successfully",
            "run_id": request.run_id,
            "model_type": "fine_tuned" if request.use_fine_tuned else "base",
            "model_name": run.name
        }
        
    except Exception as e:
        raise ValidationError(f"Failed to load model: {str(e)}")


@router.post("/chat/generate", response_model=GenerateResponse)
async def generate_chat_response(
    request: GenerateRequest,
    run_id: str,
    use_fine_tuned: bool = True,
    db: Session = Depends(get_db)
):
    """Generate a response from a loaded model."""
    
    cache_key = f"{run_id}_{use_fine_tuned}"
    
    if cache_key not in loaded_models:
        # Auto-load if not in cache
        load_request = LoadModelRequest(run_id=run_id, use_fine_tuned=use_fine_tuned)
        await load_model(load_request, db)
    
    model, tokenizer, model_name = loaded_models[cache_key]
    
    try:
        result = await generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=request.message,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return GenerateResponse(
            text=result["text"],
            tokens=result["tokens"],
            response_time=result["response_time"],
            tokens_per_second=result["tokens_per_second"]
        )
        
    except Exception as e:
        raise ValidationError(f"Generation failed: {str(e)}")


@router.websocket("/ws/chat/{run_id}")
async def chat_websocket(
    websocket: WebSocket,
    run_id: str,
    use_fine_tuned: bool = True
):
    """WebSocket for streaming chat responses with security validation."""
    
    # Security: Validate WebSocket origin
    from ..config import get_settings
    settings = get_settings()
    if not getattr(settings, 'allow_remote', False):
        client_host = websocket.client.host if websocket.client else None
        allowed_hosts = ["127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"]
        if client_host and client_host not in allowed_hosts:
            await websocket.close(code=1008, reason="Access denied")
            return
    
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Security: Extract and validate inputs
            raw_message = data.get("message", "")
            raw_system_prompt = data.get("system_prompt", "You are a helpful assistant.")
            max_tokens = data.get("max_tokens", 512)
            temperature = data.get("temperature", 0.7)
            
            # Security: Validate message
            is_valid, error_msg = validate_message(raw_message)
            if not is_valid:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Message validation failed: {error_msg}"
                })
                continue
            
            # Security: Sanitize inputs
            message = sanitize_input(raw_message, max_length=4000, allow_html=False)
            system_prompt = sanitize_input(raw_system_prompt, max_length=2000, allow_html=False)
            
            # Security: Validate system prompt
            is_valid, error_msg = validate_system_prompt(raw_system_prompt)
            if not is_valid:
                await websocket.send_json({
                    "type": "error",
                    "message": f"System prompt validation failed: {error_msg}"
                })
                continue
            
            # Security: Validate numeric parameters
            try:
                max_tokens = int(max_tokens)
                temperature = float(temperature)
                if not (16 <= max_tokens <= 4096):
                    raise ValueError("max_tokens out of range")
                if not (0.0 <= temperature <= 2.0):
                    raise ValueError("temperature out of range")
            except (ValueError, TypeError) as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Invalid parameter: {e}"
                })
                continue
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "start",
                "timestamp": datetime.now().isoformat()
            })
            
            # Get cached model
            cache_key = f"{run_id}_{use_fine_tuned}"
            
            if cache_key not in loaded_models:
                await websocket.send_json({
                    "type": "error",
                    "message": "Model not loaded"
                })
                continue
            
            model, tokenizer, model_name = loaded_models[cache_key]
            
            # Generate response (streaming)
            try:
                response_text = ""
                token_count = 0
                
                full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
                
                for token in mlx_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=full_prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                    verbose=False
                ):
                    response_text += token
                    token_count += 1
                    
                    # Send token update (throttled)
                    if token_count % 5 == 0:
                        await websocket.send_json({
                            "type": "token",
                            "text": token,
                            "tokens_so_far": token_count
                        })
                
                # Send final response
                await websocket.send_json({
                    "type": "complete",
                    "text": response_text.strip(),
                    "tokens": token_count,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        print(f"Chat WebSocket disconnected for run {run_id}")
    except Exception as e:
        print(f"Chat WebSocket error: {e}")
        await websocket.close()


@router.post("/chat/unload-model")
async def unload_model(run_id: str, use_fine_tuned: bool = True):
    """Unload a model from memory."""
    cache_key = f"{run_id}_{use_fine_tuned}"
    
    if cache_key in loaded_models:
        del loaded_models[cache_key]
        return {"message": "Model unloaded"}
    
    return {"message": "Model not in cache"}
