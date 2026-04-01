"""
MLX Training Service - Core training engine using mlx_lm
"""

import os
import sys
import json
import time
import signal
import psutil
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import threading
import queue

# MLX imports
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx_lm import load, generate
from mlx_lm.lora import train_model
from mlx_lm.tuner.trainer import train, TrainingArgs, TrainingCallback, evaluate
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters, print_trainable_parameters
from mlx_lm.tuner.datasets import load_dataset, CacheDataset

# Use centralized logging
from ..core.logging import get_logger
import re

# Logging
logger = get_logger(__name__)

# Import config for persistent paths
from ..config import get_model_cache_dir

# Custom dataset loader for Alpaca format
def load_alpaca_dataset(data_dir: str, tokenizer, max_seq_length: int = 2048):
    """Load Alpaca format dataset and convert to mlx_lm compatible format."""
    from mlx_lm.tuner.datasets import TextDataset
    
    train_file = os.path.join(data_dir, "train.jsonl")
    valid_file = os.path.join(data_dir, "valid.jsonl")
    test_file = os.path.join(data_dir, "test.jsonl")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    # Read and parse the training dataset
    train_samples = []
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Convert Alpaca format to text format
                if 'instruction' in data and 'output' in data:
                    # Alpaca format
                    instruction = data.get('instruction', '')
                    input_text = data.get('input', '')
                    output = data.get('output', '')
                    
                    # Convert Alpaca format to text format
                    # If tokenizer supports chat_template, convert to chat format for consistency
                    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                        # Convert Alpaca to chat messages format
                        messages = []
                        if input_text:
                            messages.append({"role": "user", "content": f"{instruction}\n\nInput: {input_text}"})
                        else:
                            messages.append({"role": "user", "content": instruction})
                        messages.append({"role": "assistant", "content": output})
                        # Use chat template (without generation prompt since response is included)
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    else:
                        # Fallback to Alpaca format for non-chat models
                        if input_text:
                            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                        else:
                            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    
                    train_samples.append({'text': prompt})
                elif 'messages' in data:
                    # Chat format - convert to text
                    messages = data['messages']
                    # Ensure messages format is correct for training
                    # For training, we want the full conversation including assistant response
                    # but WITHOUT the generation prompt since the response is already there
                    if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None):
                        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    else:
                        # Fallback: concatenate messages manually
                        text = ""
                        for msg in messages:
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                            if role == 'system':
                                text += f"System: {content}\n\n"
                            elif role == 'user':
                                text += f"User: {content}\n\n"
                            elif role == 'assistant':
                                text += f"Assistant: {content}\n\n"
                    train_samples.append({'text': text})
                elif 'text' in data:
                    # Already in text format
                    train_samples.append(data)
                else:
                    # Unknown format, try to use as-is
                    train_samples.append({'text': json.dumps(data)})
            except json.JSONDecodeError:
                continue
    
    if not train_samples:
        raise ValueError("No valid samples found in dataset")
    
    logger.info(f"Loaded {len(train_samples)} samples from Alpaca format dataset")
    
    # Create TextDataset objects
    train_dataset = TextDataset(train_samples, tokenizer, text_key="text")
    
    # Load validation set if exists
    valid_dataset = None
    if os.path.exists(valid_file):
        valid_samples = []
        with open(valid_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'instruction' in data and 'output' in data:
                        instruction = data.get('instruction', '')
                        input_text = data.get('input', '')
                        output = data.get('output', '')
                        # Convert to chat format if tokenizer supports it
                        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                            messages = []
                            if input_text:
                                messages.append({"role": "user", "content": f"{instruction}\n\nInput: {input_text}"})
                            else:
                                messages.append({"role": "user", "content": instruction})
                            messages.append({"role": "assistant", "content": output})
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                        else:
                            if input_text:
                                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                            else:
                                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                        valid_samples.append({'text': prompt})
                    elif 'messages' in data:
                        # Chat format validation data
                        messages = data['messages']
                        if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None):
                            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                        else:
                            # Fallback: concatenate messages manually
                            text = ""
                            for msg in messages:
                                role = msg.get('role', '')
                                content = msg.get('content', '')
                                if role == 'system':
                                    text += f"System: {content}\n\n"
                                elif role == 'user':
                                    text += f"User: {content}\n\n"
                                elif role == 'assistant':
                                    text += f"Assistant: {content}\n\n"
                        valid_samples.append({'text': text})
                    elif 'text' in data:
                        valid_samples.append(data)
                except json.JSONDecodeError:
                    continue
        if valid_samples:
            valid_dataset = TextDataset(valid_samples, tokenizer, text_key="text")
            logger.info(f"Loaded {len(valid_samples)} validation samples")
    
    # Load test set if exists
    test_dataset = None
    if os.path.exists(test_file):
        test_samples = []
        with open(test_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'instruction' in data and 'output' in data:
                        instruction = data.get('instruction', '')
                        input_text = data.get('input', '')
                        output = data.get('output', '')
                        # Convert to chat format if tokenizer supports it
                        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                            messages = []
                            if input_text:
                                messages.append({"role": "user", "content": f"{instruction}\n\nInput: {input_text}"})
                            else:
                                messages.append({"role": "user", "content": instruction})
                            messages.append({"role": "assistant", "content": output})
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                        else:
                            if input_text:
                                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                            else:
                                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                        test_samples.append({'text': prompt})
                    elif 'messages' in data:
                        # Chat format test data
                        messages = data['messages']
                        if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None):
                            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                        else:
                            # Fallback: concatenate messages manually
                            text = ""
                            for msg in messages:
                                role = msg.get('role', '')
                                content = msg.get('content', '')
                                if role == 'system':
                                    text += f"System: {content}\n\n"
                                elif role == 'user':
                                    text += f"User: {content}\n\n"
                                elif role == 'assistant':
                                    text += f"Assistant: {content}\n\n"
                        test_samples.append({'text': text})
                    elif 'text' in data:
                        test_samples.append(data)
                except json.JSONDecodeError:
                    continue
        if test_samples:
            test_dataset = TextDataset(test_samples, tokenizer, text_key="text")
            logger.info(f"Loaded {len(test_samples)} test samples")
    
    return train_dataset, valid_dataset, test_dataset
    
    # Create a simple dataset that yields the samples
    return samples, [], []


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    model_id: str
    data_path: str
    output_path: str
    
    # Training params
    steps: int = 100
    learning_rate: float = 1e-4
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    batch_size: int = 4
    max_seq_length: int = 2048
    warmup_steps: int = 10
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 0
    
    # Advanced params
    gradient_checkpointing: bool = False
    num_lora_layers: int = 16
    prompt_masking: bool = True
    validation_split_percent: int = 10  # 5, 10, or 15
    
    # Resource limits
    cpu_cores_limit: Optional[int] = None
    gpu_memory_limit_gb: Optional[float] = None
    ram_limit_gb: Optional[float] = None


class MLXTrainingCallback(TrainingCallback):
    """Custom training callback to intercept steps for monitoring and control."""
    
    def __init__(self, training_process: 'TrainingProcess'):
        self.training_process = training_process
        self.iteration_count = 0
    
    def on_train_loss_report(self, train_info: Dict[str, Any]):
        """Called after training loss report."""
        self.iteration_count += 1
        
        # Check for stop/pause
        if self.training_process._check_should_stop():
            raise InterruptedError("Training stopped by user")
        
        while self.training_process._check_should_pause():
            time.sleep(0.5)
            if self.training_process._check_should_stop():
                raise InterruptedError("Training stopped while paused")
        
        # Get training info
        # mlx_lm calls this every steps_per_report iterations
        # So actual step = iteration_count * steps_per_report
        actual_step = self.iteration_count * 10  # steps_per_report is 10
        loss = train_info.get("train_loss", train_info.get("loss", 0))
        
        # Update training process state
        self.training_process.current_step = actual_step
        
        # Capture performance metrics
        self.training_process.current_loss = loss
        self.training_process.it_per_second = train_info.get("iterations_per_second", 0)
        self.training_process.tokens_per_second = train_info.get("tokens_per_second", 0)
        
        # Track best loss
        if self.training_process.best_loss is None or loss < self.training_process.best_loss:
            self.training_process.best_loss = loss
            self.training_process.best_step = actual_step
        
        # Calculate actual learning rate with warmup
        target_lr = self.training_process.config.learning_rate
        warmup_steps = self.training_process.config.warmup_steps
        if actual_step <= warmup_steps and warmup_steps > 0:
            # Linear warmup: ramp from 0 to target_lr over warmup_steps
            current_lr = target_lr * (actual_step / warmup_steps)
        else:
            current_lr = target_lr
        
        # Monitor resources
        resources = self.training_process._monitor_resources()
        
        # Write detailed log entry (granular logging)
        self.training_process._write_detailed_log_entry(
            actual_step,
            loss,
            current_lr,  # Use actual LR instead of target
            self.training_process.tokens_per_second,
            self.training_process.it_per_second,
            resources
        )
        
        # Call external callback (store data for async processing)
        if self.training_process.on_step_complete:
            try:
                # Just call synchronously - the caller can handle async if needed
                callback_data = {
                    "step": actual_step,
                    "loss": loss,
                    "learning_rate": current_lr,  # Report actual LR with warmup
                    "best_loss": self.training_process.best_loss,
                    "best_step": self.training_process.best_step,
                    "tokens_per_second": self.training_process.tokens_per_second,
                    "it_per_second": self.training_process.it_per_second,
                    "resources": resources,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Include validation loss if available
                if self.training_process.validation_loss is not None:
                    callback_data["validation_loss"] = self.training_process.validation_loss
                
                self.training_process.on_step_complete(callback_data)
                logger.info(f"Step {actual_step}: loss={loss:.4f}, lr={current_lr:.2e}, best={self.training_process.best_loss:.4f}")
            except Exception as e:
                logger.warning(f"Error in step complete callback: {e}")
    
    def on_val_loss_report(self, val_info: Dict[str, Any]):
        """Called after validation loss report."""
        # Capture validation loss
        val_loss = val_info.get("loss", val_info.get("val_loss", 0))
        self.training_process.validation_loss = val_loss
        actual_step = self.iteration_count * 10
        
        logger.info(f"VALIDATION CALLBACK TRIGGERED - Step {actual_step}: loss={val_loss:.4f}")
        
        # Also update the training process current step to match
        self.training_process.current_step = actual_step
        
        # Store validation step separately
        self.training_process.last_validation_step = actual_step
        
        # Save validation loss to database via the step callback
        # This ensures validation loss appears in the metrics table
        if self.training_process.on_step_complete:
            try:
                callback_data = {
                    "step": actual_step,
                    "loss": self.training_process.current_loss if self.training_process.current_loss is not None else 0,
                    "validation_loss": val_loss,
                    "learning_rate": self.training_process.config.learning_rate,
                    "best_loss": self.training_process.best_loss,
                    "best_step": self.training_process.best_step,
                    "tokens_per_second": self.training_process.tokens_per_second,
                    "it_per_second": self.training_process.it_per_second,
                    "timestamp": datetime.now().isoformat()
                }
                self.training_process.on_step_complete(callback_data)
                logger.info(f"Saved validation loss to database: step={actual_step}, loss={val_loss:.4f}")
            except Exception as e:
                logger.error(f"Error saving validation loss: {e}")
                logger.exception(e)


class TrainingProcess:
    """
    Manages a single training process with MLX.
    Runs in isolated environment with resource limits.
    """
    
    def __init__(self, run_id: str, config: TrainingConfig):
        self.run_id = run_id
        self.config = config
        self.status = "pending"
        self.current_step = 0
        self.total_steps = config.steps
        self.best_loss = None
        self.best_step = None
        self.error_message = None
        self.start_time = None
        self.end_time = None
        
        # Resource monitoring
        self.process = None
        self.peak_memory_mb = 0
        self.peak_cpu_percent = 0
        
        # Training metrics (updated during training)
        self.current_loss = None
        self.validation_loss = None
        self.last_validation_step = None
        self.tokens_per_second = 0
        self.it_per_second = 0
        
        # Callbacks
        self.on_step_complete: Optional[Callable[[Dict], None]] = None
        self.on_checkpoint_saved: Optional[Callable[[int, float], None]] = None
        self.on_training_complete: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_status_change: Optional[Callable[[str, str], None]] = None  # (status, message)
        
        # Control flags
        self._should_stop = False
        self._should_pause = False
        self._is_paused = False
        
        # MLX objects
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        # Detailed logging
        self.detailed_log_path = os.path.join(config.output_path, "logs", "detailed_training.log")
        self._detailed_log_file = None
        self._write_detailed_log_header()
        
        logger.info(f"TrainingProcess initialized for run {run_id}")
    
    def _validate_cached_model(self, download_dir: Path, expected_model_id: str) -> bool:
        """Validate that cached model matches the expected model ID.
        
        Args:
            download_dir: Path to the downloaded model directory
            expected_model_id: The HuggingFace model ID we expect
            
        Returns:
            True if model is valid and matches expected ID, False otherwise
        """
        try:
            config_file = download_dir / "config.json"
            if not config_file.exists():
                logger.warning(f"[VALIDATION] No config.json found in {download_dir}")
                return False
            
            # Read the config to get the model identifier
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Get model identifier from config
            config_model_id = config.get('_name_or_path', '')
            model_type = config.get('model_type', 'unknown')
            
            logger.info(f"[VALIDATION] Cached model config: _name_or_path='{config_model_id}', model_type='{model_type}'")
            logger.info(f"[VALIDATION] Expected model ID: '{expected_model_id}'")
            
            # Validate by checking if expected model ID is in the config
            # Handle various formats: mlx-community/Llama-3.2-1B, meta-llama/Llama-3.2-1B, etc.
            expected_name = expected_model_id.split('/')[-1] if '/' in expected_model_id else expected_model_id
            
            if config_model_id and expected_model_id in config_model_id:
                logger.info(f"[VALIDATION] ✓ Model matches expected ID")
                return True
            elif config_model_id and expected_name in config_model_id:
                logger.info(f"[VALIDATION] ✓ Model name '{expected_name}' found in config")
                return True
            elif config_model_id == expected_model_id:
                logger.info(f"[VALIDATION] ✓ Exact match")
                return True
            else:
                logger.warning(f"[VALIDATION] ✗ MISMATCH DETECTED!")
                logger.warning(f"[VALIDATION]   Expected: {expected_model_id}")
                logger.warning(f"[VALIDATION]   Found in cache: {config_model_id}")
                logger.warning(f"[VALIDATION]   Will clear cache and re-download")
                return False
                
        except Exception as e:
            logger.error(f"[VALIDATION] Error validating cached model: {e}")
            return False

    def _check_model_cached(self, model_id: str) -> bool:
        """Check if model is already cached locally with all required files.
        
        Checks in order:
        1. Our custom download directory (storage/runs/downloaded_models/)
        2. HuggingFace cache directory
        
        Includes validation to ensure cached model matches expected model_id.
        """
        from pathlib import Path
        
        logger.info(f"[CACHE CHECK] Looking for model: {model_id}")
        
        # First check our custom download directory (now persistent via EDUKAAI_MODEL_CACHE_DIR)
        try:
            cache_base_dir = get_model_cache_dir()
            download_dir = cache_base_dir / model_id.replace("/", "--")
            logger.info(f"[CACHE CHECK] Checking custom directory: {download_dir}")
            logger.info(f"[CACHE CHECK] Cache base directory: {cache_base_dir}")
            
            if download_dir.exists():
                config_file = download_dir / "config.json"
                safetensors_files = list(download_dir.glob("model*.safetensors"))
                
                if config_file.exists() and safetensors_files:
                    logger.info(f"[CACHE CHECK] Found files in custom directory: {len(safetensors_files)} safetensors files")
                    
                    # VALIDATE the cached model
                    if self._validate_cached_model(download_dir, model_id):
                        logger.info(f"[CACHE CHECK] ✓ Model {model_id} validated and ready to use")
                        return True
                    else:
                        # Model doesn't match - clear it
                        logger.warning(f"[CACHE CHECK] Cached model doesn't match {model_id}, clearing...")
                        import shutil
                        shutil.rmtree(download_dir)
                        logger.info(f"[CACHE CHECK] Cleared mismatched cache directory")
                else:
                    logger.info(f"[CACHE CHECK] Directory exists but missing files: config={config_file.exists()}, weights={len(safetensors_files)}")
            else:
                logger.info(f"[CACHE CHECK] Custom download directory does not exist")
        except Exception as e:
            logger.error(f"[CACHE CHECK] Error checking custom download directory: {e}")
        
        # Then check HuggingFace cache directory
        try:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_name = model_id.replace("/", "--")
            model_cache_path = cache_dir / f"models--{model_name}"
            
            logger.info(f"[CACHE CHECK] Checking HF cache at: {model_cache_path}")
            
            if not model_cache_path.exists():
                logger.info(f"[CACHE CHECK] HF cache directory does not exist")
                return False
            
            snapshots_dir = model_cache_path / "snapshots"
            if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
                logger.info(f"[CACHE CHECK] HF snapshots directory empty")
                return False
            
            # Check if config.json exists in any snapshot
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    config_file = snapshot / "config.json"
                    safetensors_files = list(snapshot.glob("*.safetensors"))
                    
                    if config_file.exists() and safetensors_files:
                        logger.info(f"[CACHE CHECK] Found valid HF cache at {snapshot}")
                        return True
                    elif config_file.exists():
                        logger.warning(f"[CACHE CHECK] HF snapshot missing weights: {snapshot}")
                    elif safetensors_files:
                        logger.warning(f"[CACHE CHECK] HF snapshot missing config: {snapshot}")
            
            logger.info(f"[CACHE CHECK] No valid HF cache found")
            return False
            
        except Exception as e:
            logger.error(f"[CACHE CHECK] Error checking HF cache: {e}")
            return False
    
    def _update_status(self, status: str, message: str = ""):
        """Update status and notify via callback."""
        self.status = status
        self.status_message = message  # Store message for display in UI
        if self.on_status_change:
            try:
                self.on_status_change(status, message)
            except Exception as e:
                logger.error(f"Error in status change callback: {e}")
    
    def _apply_resource_limits(self):
        """Apply CPU and memory limits to the process."""
        if self.config.cpu_cores_limit:
            try:
                # Set CPU affinity (limit to specific cores)
                process = psutil.Process()
                available_cores = list(range(psutil.cpu_count()))
                limited_cores = available_cores[:self.config.cpu_cores_limit]
                process.cpu_affinity(limited_cores)
                logger.info(f"Limited CPU to cores: {limited_cores}")
            except Exception as e:
                logger.warning(f"Could not set CPU affinity: {e}")
    
    def _monitor_resources(self) -> Dict[str, Any]:
        """Monitor current resource usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Track peaks
            self.peak_memory_mb = max(self.peak_memory_mb, memory_info.rss / 1024 / 1024)
            self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
            
            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / 1024 / 1024,
                "peak_memory_mb": self.peak_memory_mb,
                "peak_cpu_percent": self.peak_cpu_percent
            }
        except Exception as e:
            logger.warning(f"Resource monitoring error: {e}")
            return {}
    
    def _write_detailed_log_header(self):
        """Write CSV header to detailed log file."""
        try:
            os.makedirs(os.path.dirname(self.detailed_log_path), exist_ok=True)
            with open(self.detailed_log_path, 'w') as f:
                f.write("timestamp,step,loss,learning_rate,tokens_per_second,it_per_second,cpu_percent,memory_mb,peak_memory_mb\n")
        except Exception as e:
            logger.warning(f"Could not create detailed log file: {e}")
    
    def _write_detailed_log_entry(self, step: int, loss: float, learning_rate: float, tokens_per_sec: float, it_per_sec: float, resources: Dict):
        """Write detailed log entry."""
        try:
            timestamp = datetime.now().isoformat()
            cpu_percent = resources.get("cpu_percent", 0)
            memory_mb = resources.get("memory_mb", 0)
            peak_memory_mb = resources.get("peak_memory_mb", 0)
            
            with open(self.detailed_log_path, 'a') as f:
                f.write(f"{timestamp},{step},{loss:.6f},{learning_rate:.2e},{tokens_per_sec:.2f},{it_per_sec:.2f},{cpu_percent:.1f},{memory_mb:.1f},{peak_memory_mb:.1f}\n")
        except Exception as e:
            logger.warning(f"Could not write to detailed log: {e}")
    
    def _check_should_stop(self) -> bool:
        """Check if training should stop."""
        return self._should_stop
    
    def _check_should_pause(self) -> bool:
        """Check if training should pause."""
        if self._should_pause and not self._is_paused:
            self._is_paused = True
            self.status = "paused"
            logger.info(f"Training {self.run_id} paused")
            return True
        return False
    
    def _resume_from_pause(self):
        """Resume training from pause."""
        self._should_pause = False
        self._is_paused = False
        self.status = "running"
        logger.info(f"Training {self.run_id} resumed")
    
    def _download_model(self, model_id: str) -> bool:
        """Download model files from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download, HfFileSystem, get_hf_file_metadata
            from huggingface_hub import login as hf_login
            import shutil
            from ..config import get_settings
            from pathlib import Path
            
            # IMMEDIATE STOP CHECK at function entry
            if self._check_should_stop():
                logger.info("[STOP CHECK] Stop signal detected at download start - aborting immediately")
                self._update_status("stopped", "Download stopped by user")
                return False
            
            # Check if model already exists in our persistent cache directory
            cache_base_dir = get_model_cache_dir()
            download_dir = cache_base_dir / model_id.replace("/", "--")
            logger.info(f"[DOWNLOAD] Using persistent cache directory: {cache_base_dir}")
            
            if download_dir.exists():
                config_file = download_dir / "config.json"
                safetensors_files = list(download_dir.glob("model*.safetensors"))
                
                if config_file.exists() and safetensors_files:
                    logger.info(f"[SKIP DOWNLOAD] Model already exists in {download_dir}")
                    logger.info(f"[SKIP DOWNLOAD] This cache survives app reinstalls!")
                    self._update_status("downloading", f"Using cached model from {download_dir}")
                    return True
            
            # Check for HF_TOKEN and login if available
            settings = get_settings()
            hf_token = settings.hf_token
            if hf_token:
                logger.info("[AUTH] HF_TOKEN found, logging into HuggingFace Hub...")
                try:
                    hf_login(token=hf_token)
                    logger.info("[AUTH] Successfully authenticated with HuggingFace Hub")
                except Exception as auth_error:
                    logger.warning(f"[AUTH] Failed to authenticate: {auth_error}")
                    logger.info("[AUTH] Proceeding with unauthenticated download")
            else:
                logger.info("[AUTH] No HF_TOKEN found, using unauthenticated download")
            
            logger.info(f"[DOWNLOAD START] Model: {model_id}")
            self._update_status("downloading", f"Starting download of {model_id}...")
            
            # Use persistent cache directory (survives app reinstalls)
            cache_base_dir = get_model_cache_dir()
            download_dir = cache_base_dir / model_id.replace("/", "--")
            download_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"[DOWNLOAD DIR] {download_dir}")
            logger.info(f"[DOWNLOAD DIR] Persistent cache location (survives reinstalls)")
            
            # Use HfFileSystem to list files in the repo
            fs = HfFileSystem()
            
            # Get list of files in the repo
            try:
                repo_files = fs.ls(model_id, detail=False)
                logger.info(f"[REPO FILES] Found {len(repo_files)} files in repo {model_id}")
            except Exception as e:
                logger.error(f"[REPO ERROR] Could not list files in repo {model_id}: {e}")
                repo_files = []
            
            downloaded_files = []
            
            # Check for stop signal before listing
            if self._check_should_stop():
                logger.info("[STOP CHECK] Stop detected after file listing - aborting")
                self._update_status("stopped", "Download stopped by user")
                return False
            
            # List of essential files to download
            essential_files = [
                "config.json",
                "tokenizer.json", 
                "tokenizer_config.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "chat_template.json",
                "vocab.json",
                "merges.txt",
                "tokenizer.model"
            ]
            
            # Find and download all safetensors files
            safetensors_files = [f for f in repo_files if f.endswith('.safetensors')]
            if safetensors_files:
                logger.info(f"[WEIGHTS] Found {len(safetensors_files)} model weight files to download")
                logger.info("[IMPORTANT] Each file download cannot be interrupted once started. Stop signal is checked between files.")
                
                for i, safetensors_file in enumerate(safetensors_files):
                    # Check for stop signal before each file
                    if self._check_should_stop():
                        logger.info(f"[STOP CHECK] File {i+1}/{len(safetensors_files)}: Stop signal detected - aborting download")
                        self._update_status("stopped", "Download stopped by user")
                        return False
                    
                    filename = safetensors_file.split('/')[-1]
                    file_size_mb = 0
                    
                    # Try to get file size info
                    try:
                        metadata = get_hf_file_metadata(filename=filename, repo_id=model_id, token=hf_token)
                        file_size_mb = metadata.size / (1024 * 1024) if metadata.size else 0
                        size_info = f" ({file_size_mb:.1f} MB)" if file_size_mb > 0 else ""
                        logger.info(f"[DOWNLOAD] File {i+1}/{len(safetensors_files)}: {filename}{size_info}")
                    except:
                        logger.info(f"[DOWNLOAD] File {i+1}/{len(safetensors_files)}: {filename}")
                    
                    try:
                        self._update_status("downloading", f"Downloading {filename} ({i+1}/{len(safetensors_files)})...")
                        downloaded_path = hf_hub_download(
                            repo_id=model_id,
                            filename=filename,
                            local_dir=str(download_dir),
                            local_dir_use_symlinks=False,
                            resume_download=True,
                            token=hf_token  # Pass token explicitly
                        )
                        downloaded_files.append(downloaded_path)
                        logger.info(f"[DOWNLOAD COMPLETE] File {i+1}/{len(safetensors_files)}: {filename}")
                    except Exception as e:
                        logger.error(f"[DOWNLOAD FAILED] File {i+1}/{len(safetensors_files)}: {filename} - {e}")
            else:
                # Try common weight file patterns if we couldn't list
                logger.warning("[WEIGHTS] No safetensors files found in listing, trying common patterns...")
                weight_patterns = [
                    "model.safetensors",
                    "model-00001-of-00001.safetensors",
                    "model-00001-of-00002.safetensors",
                    "pytorch_model.bin",
                    "model.bin"
                ]
                for pattern in weight_patterns:
                    if self._check_should_stop():
                        logger.info("[STOP CHECK] Stop signal received during download, aborting...")
                        self._update_status("stopped", "Download stopped by user")
                        return False
                    
                    try:
                        logger.info(f"[DOWNLOAD] Trying to download: {pattern}...")
                        self._update_status("downloading", f"Downloading {pattern}...")
                        downloaded_path = hf_hub_download(
                            repo_id=model_id,
                            filename=pattern,
                            local_dir=str(download_dir),
                            local_dir_use_symlinks=False,
                            resume_download=True,
                            token=hf_token
                        )
                        downloaded_files.append(downloaded_path)
                        logger.info(f"[DOWNLOAD COMPLETE] {pattern}")
                        break  # Stop after first successful weight download
                    except Exception as e:
                        logger.debug(f"[DOWNLOAD FAILED] {pattern}: {e}")
            
            # Check for stop signal before config files
            if self._check_should_stop():
                logger.info("[STOP CHECK] Stop detected before config files - aborting")
                self._update_status("stopped", "Download stopped by user")
                return False
            
            # Download essential config/tokenizer files
            logger.info(f"[CONFIG] Downloading configuration files...")
            for i, filename in enumerate(essential_files):
                if self._check_should_stop():
                    logger.info(f"[STOP CHECK] Config file {i+1}/{len(essential_files)}: Stop detected - aborting")
                    self._update_status("stopped", "Download stopped by user")
                    return False
                
                try:
                    logger.info(f"[DOWNLOAD] Config file {i+1}/{len(essential_files)}: {filename}...")
                    self._update_status("downloading", f"Downloading {filename}...")
                    downloaded_path = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        local_dir=str(download_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                        token=hf_token
                    )
                    downloaded_files.append(downloaded_path)
                    logger.info(f"[DOWNLOAD COMPLETE] Config file {i+1}/{len(essential_files)}: {filename}")
                except Exception as e:
                    logger.debug(f"[DOWNLOAD FAILED] Config file {i+1}/{len(essential_files)}: {filename} - {e}")
            
            # Verify we have the essential files
            config_file = download_dir / "config.json"
            safetensors_files = list(download_dir.glob("*.safetensors"))
            bin_files = list(download_dir.glob("*.bin"))
            
            if not config_file.exists():
                logger.error("[VERIFY FAILED] Download failed: config.json not found")
                return False
            
            if not safetensors_files and not bin_files:
                logger.error("[VERIFY FAILED] Download failed: No model weights found")
                return False
            
            # MLX models expect 'model*.safetensors' naming convention
            # Rename if downloaded file uses different naming (e.g., 'weights.00.safetensors')
            # Only rename files that don't already have 'model' prefix AND don't create duplicates
            if safetensors_files:
                # Check if we already have properly named model files
                existing_model_files = list(download_dir.glob("model*.safetensors"))
                
                if existing_model_files:
                    logger.info(f"[MLX COMPAT] Found {len(existing_model_files)} properly named model files, skipping rename")
                else:
                    # No model files exist, need to rename
                    weights_files = [f for f in safetensors_files if not f.name.startswith('model')]
                    total_shards = len(weights_files)
                    
                    for idx, weights_file in enumerate(sorted(weights_files)):
                        # Determine new name
                        if total_shards == 1:
                            new_name = weights_file.parent / 'model.safetensors'
                        else:
                            # For sharded models, extract shard number
                            shard_match = re.search(r'(\d+)', weights_file.name)
                            if shard_match:
                                shard_num = int(shard_match.group(1)) + 1
                                new_name = weights_file.parent / f'model-{shard_num:05d}-of-{total_shards:05d}.safetensors'
                            else:
                                new_name = weights_file.parent / f'model-{idx+1:05d}-of-{total_shards:05d}.safetensors'
                        
                        # Only rename if target doesn't already exist
                        if not new_name.exists():
                            logger.info(f"[RENAME] {weights_file.name} -> {new_name.name} (MLX compatibility)")
                            try:
                                weights_file.rename(new_name)
                            except Exception as e:
                                logger.warning(f"[RENAME FAILED] {weights_file.name}: {e}")
                        else:
                            logger.info(f"[SKIP RENAME] {weights_file.name} -> {new_name.name} (target already exists)")
            
            logger.info(f"[DOWNLOAD SUCCESS] Model downloaded to {download_dir}")
            logger.info(f"[DOWNLOAD SUCCESS] Total files downloaded: {len(downloaded_files)}")
            self._update_status("downloading", f"Model download complete! {len(downloaded_files)} files downloaded")
            return True
            
        except Exception as e:
            logger.error(f"[DOWNLOAD ERROR] Failed to download model {model_id}: {e}")
            logger.exception(e)
            self.error_message = f"Failed to download model: {str(e)}"
            return False
    
    async def train(self):
        """
        Main training loop using mlx_lm.
        """
        try:
            self.status = "running"
            self.start_time = datetime.now()
            
            # Apply resource limits
            self._apply_resource_limits()
            
            logger.info("=" * 70)
            logger.info(f"[TRAINING START] Run ID: {self.run_id}")
            logger.info(f"[TRAINING CONFIG] Requested Model: {self.config.model_id}")
            logger.info(f"[TRAINING CONFIG] Steps: {self.config.steps}")
            logger.info(f"[TRAINING CONFIG] Data path: {self.config.data_path}")
            logger.info(f"[TRAINING CONFIG] Output path: {self.config.output_path}")
            logger.info("=" * 70)
            
            # Determine model path with detailed logging
            # Use persistent cache directory (survives app reinstalls)
            cache_base_dir = get_model_cache_dir()
            download_dir = cache_base_dir / self.config.model_id.replace("/", "--")
            logger.info(f"[MODEL RESOLUTION] Expected model: {self.config.model_id}")
            logger.info(f"[MODEL RESOLUTION] Cache base dir: {cache_base_dir}")
            logger.info(f"[MODEL RESOLUTION] Model specific dir: {download_dir}")
            logger.info(f"[MODEL RESOLUTION] Note: Cache survives app reinstalls")
            
            # Check if model needs to be downloaded
            # First check our custom download directory
            if download_dir.exists() and any(download_dir.glob("model*.safetensors")):
                # Use downloaded model - but VALIDATE first
                logger.info(f"[MODEL RESOLUTION] Found files in custom directory, validating...")
                if self._validate_cached_model(download_dir, self.config.model_id):
                    self.model_path = str(download_dir)
                    logger.info(f"[MODEL RESOLUTION] ✓ Using validated cached model at: {self.model_path}")
                else:
                    logger.warning(f"[MODEL RESOLUTION] ✗ Cached model validation failed, will re-download")
                    import shutil
                    shutil.rmtree(download_dir)
                    logger.info(f"[MODEL RESOLUTION] Cleared invalid cache, downloading...")
                    download_success = self._download_model(self.config.model_id)
                    if not download_success:
                        raise FileNotFoundError(f"Failed to download model {self.config.model_id}")
                    self.model_path = str(download_dir)
                    logger.info(f"[MODEL RESOLUTION] Using freshly downloaded model at: {self.model_path}")
            elif self._check_model_cached(self.config.model_id):
                # Model exists in HF cache, use HF ID
                self.model_path = self.config.model_id
                logger.info(f"[MODEL RESOLUTION] Using HuggingFace cache for: {self.model_path}")
            else:
                # Model not found anywhere, download it
                logger.info(f"[MODEL RESOLUTION] Model {self.config.model_id} not found, downloading...")
                download_success = self._download_model(self.config.model_id)
                if not download_success:
                    raise FileNotFoundError(f"Failed to download model {self.config.model_id}")
                # Use downloaded model path
                self.model_path = str(download_dir)
                logger.info(f"[MODEL RESOLUTION] Using downloaded model at: {self.model_path}")
            
            # Log final model path before loading
            logger.info("=" * 70)
            logger.info(f"[MODEL LOADING] Final model_path: {self.model_path}")
            logger.info(f"[MODEL LOADING] Config model_id: {self.config.model_id}")
            logger.info("=" * 70)
            
            # Load model
            logger.info("[MODEL LOADING] Loading model into memory...")
            self._update_status("loading_model", f"Loading {self.config.model_id} into memory...")
            self.model, self.tokenizer = load(
                self.model_path,
                tokenizer_config={"trust_remote_code": True}
            )
            
            # Model loaded successfully - verify by checking config
            logger.info("[MODEL LOADING] ✓ Model loaded successfully")
            self._update_status("model_loaded", "Model loaded successfully")
            
            # Additional verification - log model info if available
            if hasattr(self.model, 'config'):
                model_config = self.model.config
                loaded_model_type = getattr(model_config, 'model_type', 'unknown')
                loaded_vocab_size = getattr(model_config, 'vocab_size', 'unknown')
                logger.info(f"[MODEL VERIFICATION] Loaded model type: {loaded_model_type}")
                logger.info(f"[MODEL VERIFICATION] Vocab size: {loaded_vocab_size}")
            
            # Create adapter output directory
            adapter_path = Path(self.config.output_path)
            adapter_path.mkdir(parents=True, exist_ok=True)
            adapter_file = adapter_path / "adapters.safetensors"
            
            # Prepare training arguments
            logger.info("Setting up training...")
            
            # Create args object similar to command-line args
            class Args:
                pass
            
            args = Args()
            args.model = self.model_path  # Use local path instead of HF ID
            args.train = True
            # We'll set args.data below after determining the directory
            args.fine_tune_type = "lora"
            args.optimizer = "adam"
            args.optimizer_config = {"adam": {}, "adamw": {}, "sgd": {}, "adafactor": {}}
            args.num_layers = self.config.num_lora_layers
            args.batch_size = self.config.batch_size
            args.iters = self.config.steps
            args.val_batches = 25
            args.learning_rate = self.config.learning_rate
            args.steps_per_report = 10
            args.steps_per_eval = 25  # Run validation every 25 steps (was 100)
            args.resume_adapter_file = None
            args.adapter_path = str(adapter_path)
            args.save_every = 100
            args.test = False
            args.test_batches = 500
            args.max_seq_length = self.config.max_seq_length
            args.seed = 0
            args.grad_checkpoint = self.config.gradient_checkpointing
            args.grad_accumulation_steps = self.config.gradient_accumulation_steps
            args.clear_cache_threshold = 0
            args.lr_schedule = None  # Disable mlx_lm's built-in schedule, we handle it manually in callback
            args.lora_parameters = {
                "rank": self.config.lora_rank,
                "dropout": self.config.lora_dropout,
                "scale": self.config.lora_alpha / self.config.lora_rank  # scale = alpha / rank
            }
            args.mask_prompt = self.config.prompt_masking
            args.report_to = None
            args.project_name = None
            
            # Prepare dataset path - mlx_lm expects a directory, not a file
            import os
            data_dir = os.path.dirname(self.config.data_path)
            if not data_dir:
                data_dir = "."
            args.data = data_dir
            
            logger.info(f"Dataset directory: {data_dir}")
            
            # Verify dataset files exist
            train_file = os.path.join(data_dir, "train.jsonl")
            if not os.path.exists(train_file):
                logger.error(f"Training file not found: {train_file}")
                raise FileNotFoundError(f"Training dataset not found at {train_file}")
            
            # Count samples in dataset
            with open(train_file, 'r') as f:
                sample_count = sum(1 for line in f if line.strip())
            logger.info(f"Found {sample_count} training samples")
            
            if sample_count == 0:
                raise ValueError("Dataset contains no valid samples")
            
            # Adjust batch size if needed
            if sample_count < args.batch_size:
                logger.warning(f"Batch size ({args.batch_size}) > samples ({sample_count}), reducing to {sample_count}")
                args.batch_size = max(1, sample_count)
            
            # Load dataset using custom Alpaca loader
            logger.info("Loading datasets...")
            try:
                train_set, valid_set, test_set = load_alpaca_dataset(data_dir, self.tokenizer, args.max_seq_length)
                
                if not train_set or len(train_set) == 0:
                    raise ValueError("Dataset loaded but contains 0 examples")
                
                logger.info(f"Successfully loaded {len(train_set)} training examples")
                
                # Log validation approach
                validation_file = os.path.join(data_dir, "valid.jsonl")
                if os.path.exists(validation_file):
                    with open(validation_file, 'r') as f:
                        val_count = sum(1 for line in f if line.strip())
                    logger.info(f"Using custom validation set: {val_count} samples from valid.jsonl")
                else:
                    val_percent = getattr(self.config, 'validation_split_percent', 10)
                    logger.info(f"Using auto-split validation: {val_percent}% of training data")
                
                # Create validation split (configurable % of training data) for validation curve
                if valid_set is None or len(valid_set) == 0:
                    from mlx_lm.tuner.datasets import TextDataset
                    import random
                    # Use configurable split percentage (default 10%)
                    val_percent = getattr(self.config, 'validation_split_percent', 10)
                    train_percent = 100 - val_percent
                    
                    logger.info(f"DEBUG: train_set type={type(train_set)}, len={len(train_set)}")
                    split_point = max(1, int(len(train_set) * (train_percent / 100)))
                    logger.info(f"DEBUG: split_point={split_point}, train_percent={train_percent}, val_percent={val_percent}")
                    
                    # Try to access underlying data - TextDataset uses _data attribute
                    train_samples = None
                    if hasattr(train_set, '_data'):
                        train_samples = train_set._data
                        logger.info(f"DEBUG: Found _data attribute with {len(train_samples)} samples")
                    elif hasattr(train_set, 'samples'):
                        train_samples = train_set.samples
                        logger.info(f"DEBUG: Found samples attribute with {len(train_samples)} samples")
                    else:
                        # Fallback: iterate through dataset to collect samples
                        logger.info("DEBUG: No _data or samples attribute, iterating to collect samples...")
                        train_samples = []
                        for i, item in enumerate(train_set):
                            train_samples.append(item)
                            if i >= len(train_set) - 1:  # Stop after collecting all
                                break
                        logger.info(f"DEBUG: Collected {len(train_samples)} samples by iteration")
                    
                    if train_samples and len(train_samples) > 0:
                        valid_samples = train_samples[split_point:]
                        train_samples = train_samples[:split_point]
                        logger.info(f"DEBUG: Split into {len(train_samples)} train + {len(valid_samples)} validation samples")
                        
                        if len(valid_samples) > 0:
                            valid_set = TextDataset(valid_samples, self.tokenizer, text_key="text")
                            logger.info(f"Created validation set with {len(valid_samples)} samples ({val_percent}% auto-split)")
                        else:
                            valid_set = TextDataset([], self.tokenizer, text_key="text")
                            logger.warning("Could not create validation set - no samples available for split")
                    else:
                        logger.error("DEBUG: Could not extract samples from train_set")
                        valid_set = TextDataset([], self.tokenizer, text_key="text")
                        logger.warning("Could not create validation set - no samples available")
                
                if test_set is None:
                    from mlx_lm.tuner.datasets import TextDataset
                    test_set = TextDataset([], self.tokenizer, text_key="text")
                
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                logger.exception(e)
                raise ValueError(f"Failed to load dataset: {str(e)}")
            
            # Create training callback
            training_callback = MLXTrainingCallback(self)
            
            # Log validation dataset info
            logger.info(f"DEBUG: Before train_model - valid_set type={type(valid_set)}, len={len(valid_set) if valid_set else 'None'}")
            if valid_set and len(valid_set) > 0:
                logger.info(f"Validation dataset ready: {len(valid_set)} samples")
            else:
                logger.warning("No validation dataset - validation loss will not be calculated")
            
            # Log training parameters
            logger.info(f"DEBUG: Training args - steps_per_eval={args.steps_per_eval}, val_batches={args.val_batches}")
            
            # Run training
            logger.info("Starting MLX LoRA training...")
            
            # Run in executor to not block async event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: train_model(args, self.model, train_set, valid_set, training_callback)
            )
            
            # Training complete
            self.status = "completed"
            self.end_time = datetime.now()
            
            logger.info(f"Training {self.run_id} completed successfully")
            logger.info(f"Best loss: {self.best_loss} at step {self.best_step}")
            
            if self.on_training_complete:
                try:
                    self.on_training_complete()
                except Exception as callback_error:
                    logger.error(f"Error in complete callback: {callback_error}")
                
        except InterruptedError:
            self.status = "stopped"
            self.end_time = datetime.now()
            logger.info(f"Training {self.run_id} stopped by user")
            
        except FileNotFoundError as e:
            self.status = "failed"
            self.end_time = datetime.now()
            self.error_message = f"File not found: {str(e)}"
            logger.error(f"Training {self.run_id} failed: {self.error_message}")
            
            if self.on_error:
                try:
                    self.on_error(self.error_message)
                except Exception as callback_error:
                    logger.error(f"Error in error callback: {callback_error}")
                    
        except ValueError as e:
            self.status = "failed"
            self.end_time = datetime.now()
            self.error_message = f"Invalid value: {str(e)}"
            logger.error(f"Training {self.run_id} failed: {self.error_message}")
            
            if self.on_error:
                try:
                    self.on_error(self.error_message)
                except Exception as callback_error:
                    logger.error(f"Error in error callback: {callback_error}")
                    
        except Exception as e:
            self.status = "failed"
            self.end_time = datetime.now()
            error_type = type(e).__name__
            self.error_message = f"{error_type}: {str(e)}"
            logger.error(f"Training {self.run_id} failed: {self.error_message}")
            logger.exception(e)
            
            if self.on_error:
                # Call the callback (synchronous)
                try:
                    self.on_error(self.error_message)
                except Exception as callback_error:
                    logger.error(f"Error in error callback: {callback_error}")
    
    def pause(self):
        """Pause training."""
        self._should_pause = True
        logger.info(f"Pause requested for training {self.run_id}")
    
    def resume(self):
        """Resume training from pause."""
        self._should_pause = False
        if self._is_paused:
            self._resume_from_pause()
        logger.info(f"Resume requested for training {self.run_id}")
    
    def stop(self):
        """Stop training."""
        self._should_stop = True
        logger.info(f"Stop requested for training {self.run_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "status_message": getattr(self, 'status_message', ''),  # Include status message for UI display
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_loss": self.current_loss,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "validation_loss": self.validation_loss,
            "error_message": getattr(self, 'error_message', None),
            "progress": (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_cpu_percent": self.peak_cpu_percent,
            "tokens_per_second": self.tokens_per_second,
            "it_per_second": self.it_per_second,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class TrainingManager:
    """
    Manages multiple training processes.
    Ensures resource isolation and proper cleanup.
    """
    
    def __init__(self):
        self.active_processes: Dict[str, TrainingProcess] = {}
        self._lock = threading.Lock()
        logger.info("TrainingManager initialized")
    
    async def create_training(
        self, 
        run_id: str, 
        config: TrainingConfig,
        step_callback: Optional[Callable[[Dict], None]] = None,
        complete_callback: Optional[Callable[[], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str, str], None]] = None
    ) -> TrainingProcess:
        """Create and start a new training process."""
        
        with self._lock:
            if run_id in self.active_processes:
                raise ValueError(f"Training {run_id} already exists")
            
            # Create training process
            process = TrainingProcess(run_id, config)
            process.on_step_complete = step_callback
            process.on_training_complete = complete_callback
            process.on_error = error_callback
            process.on_status_change = status_callback
            
            self.active_processes[run_id] = process
        
        # Start training in background
        asyncio.create_task(process.train())
        
        logger.info(f"Created training {run_id}")
        return process
    
    def get_process(self, run_id: str) -> Optional[TrainingProcess]:
        """Get an active training process."""
        return self.active_processes.get(run_id)
    
    def pause_training(self, run_id: str):
        """Pause a training process."""
        process = self.get_process(run_id)
        if process:
            process.pause()
    
    def resume_training(self, run_id: str):
        """Resume a training process."""
        process = self.get_process(run_id)
        if process:
            process.resume()
    
    def stop_training(self, run_id: str):
        """Stop a training process."""
        process = self.get_process(run_id)
        if process:
            process.stop()
    
    def cleanup(self, run_id: str):
        """Remove a completed training process from active list."""
        with self._lock:
            if run_id in self.active_processes:
                process = self.active_processes[run_id]
                if process.status in ["completed", "stopped", "failed"]:
                    del self.active_processes[run_id]
                    logger.info(f"Cleaned up training {run_id}")
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all active processes."""
        return {
            run_id: process.get_stats()
            for run_id, process in self.active_processes.items()
        }


# Global training manager instance
training_manager = TrainingManager()


# Export helper functions
async def export_model(
    model_path: str,
    adapter_path: str,
    export_format: str,  # "adapter", "fused", "gguf"
    output_path: str,
    hyperparameters: Optional[Dict] = None
) -> str:
    """
    Export a trained model in various formats.
    
    Args:
        model_path: Path to base model
        adapter_path: Path to LoRA adapters
        export_format: One of "adapter", "fused", "gguf"
        output_path: Where to save the export
        hyperparameters: Training hyperparameters for adapter config
    
    Returns:
        Path to exported model
    """
    try:
        logger.info(f"Exporting model from {model_path} with adapters {adapter_path}")
        logger.info(f"Format: {export_format}")
        
        if export_format == "adapter":
            # Copy adapters to export directory
            import shutil
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # adapter_path is the full path to adapters.safetensors
            adapter_file = Path(adapter_path)
            
            if not adapter_file.exists():
                raise FileNotFoundError(f"Adapter file not found at {adapter_file}")
            
            # Copy to exports directory
            export_file = output_dir / "adapters.safetensors"
            shutil.copy2(adapter_file, export_file)
            
            logger.info(f"Adapter exported to {export_file}")
            return str(export_file)
        
        elif export_format == "fused":
            # Fuse adapters with base model
            logger.info("Fusing adapters...")
            
            # Create proper adapters directory structure for mlx_lm
            # mlx_lm.load_adapters expects a directory with adapter_config.json
            adapters_dir = Path(adapter_path).parent / "adapters"
            adapters_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy adapters.safetensors to the directory
            adapter_source = Path(adapter_path)
            adapter_dest = adapters_dir / "adapters.safetensors"
            if not adapter_dest.exists():
                import shutil
                shutil.copy2(adapter_source, adapter_dest)
                logger.info(f"Copied adapters to {adapter_dest}")
            
            # Create/overwrite adapter_config.json with proper lora_parameters
            config_path = adapters_dir / "adapter_config.json"
            # Always create/overwrite to ensure lora_parameters exists
            lora_rank = hyperparameters.get('lora_rank', 8) if hyperparameters else 8
            lora_alpha = hyperparameters.get('lora_alpha', 16) if hyperparameters else 16
            lora_dropout = hyperparameters.get('lora_dropout', 0.05) if hyperparameters else 0.05
            
            adapter_config = {
                "adapter_type": "lora",
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_parameters": {
                    "rank": lora_rank,
                    "alpha": lora_alpha,
                    "dropout": lora_dropout,
                    "scale": lora_alpha / lora_rank if lora_rank > 0 else 1.0
                },
                "target_modules": ["q_proj", "v_proj"],
                "num_layers": hyperparameters.get('num_lora_layers', 8) if hyperparameters else 8,
                "base_model_name_or_path": model_path
            }
            import json
            with open(config_path, 'w') as f:
                json.dump(adapter_config, f, indent=2)
            logger.info(f"Created/updated adapter config at {config_path}")
            
            # Load base model with adapters
            logger.info("Loading base model with adapters...")
            model, tokenizer = load(
                model_path, 
                tokenizer_config={"trust_remote_code": True},
                adapter_path=str(adapters_dir)
            )
            
            # Fuse the adapters into the base model
            logger.info("Fusing LoRA adapters into base model...")
            from mlx.utils import tree_flatten, tree_unflatten
            
            # Get all modules that can be fused (LoRA layers)
            fused_modules = []
            for name, module in model.named_modules():
                if hasattr(module, "fuse"):
                    fused_modules.append((name, module.fuse()))
            
            if fused_modules:
                logger.info(f"Fusing {len(fused_modules)} LoRA layers...")
                model.update_modules(tree_unflatten(fused_modules))
                logger.info("Fusion complete")
            else:
                logger.warning("No LoRA layers found to fuse")
            
            # Save fused model using mlx_lm's save function
            logger.info("Saving fused model...")
            from mlx_lm.utils import save
            
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get model config
            from mlx_lm.utils import load as mlx_load
            _, _, model_config = mlx_load(model_path, return_config=True)
            
            # Save using mlx_lm's save function
            save(
                output_dir,
                model_path,
                model,
                tokenizer,
                model_config,
                donate_model=False
            )
            
            logger.info(f"Fused model saved to {output_path}")
            return str(output_dir)
        
        elif export_format == "gguf":
            # Convert to GGUF format
            logger.info("Converting to GGUF...")
            logger.warning("GGUF export requires llama.cpp. Please convert manually:")
            logger.warning(f"  1. Export fused model: {output_path}")
            logger.warning(f"  2. Use llama.cpp convert.py to convert to GGUF")
            
            # For now, return adapter path
            # TODO: Implement GGUF conversion when llama.cpp is available
            return adapter_path
        
        else:
            raise ValueError(f"Unknown export format: {export_format}")
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        logger.exception(e)
        raise


async def load_model_for_inference(
    model_path: str,
    adapter_path: Optional[str] = None
) -> tuple[Any, Any]:
    """
    Load a model for inference (chat/generation).
    
    Args:
        model_path: Path to model
        adapter_path: Optional path to LoRA adapters
    
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        logger.info(f"Loading model for inference: {model_path}")
        
        # Load base model
        model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
        
        # Load adapters if provided
        if adapter_path and os.path.exists(adapter_path):
            logger.info(f"Loading adapters from {adapter_path}")
            load_adapters(model, adapter_path)
        
        logger.info("Model loaded successfully for inference")
        return (model, tokenizer)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.exception(e)
        raise


async def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """
    Generate a response from the model with proper tokenizer handling.
    
    Returns:
        Dict with response text and metrics
    """
    import time
    
    start_time = time.time()
    
    try:
        # Get special tokens from tokenizer
        eos_token_id = None
        stop_strings = []
        
        # Try to get EOS token from tokenizer
        if hasattr(tokenizer, 'eos_token_id'):
            eos_token_id = tokenizer.eos_token_id
        elif hasattr(tokenizer, 'eos_token'):
            # Convert token string to ID if needed
            if hasattr(tokenizer, 'encode'):
                try:
                    eos_token_id = tokenizer.encode(tokenizer.eos_token)[-1]
                except:
                    pass
        
        # Get stop strings based on model type
        if hasattr(tokenizer, 'apply_chat_template'):
            # Modern chat models
            stop_strings = ["<|end|>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>", "<|assistant|>"]
        else:
            # Generic stop strings
            stop_strings = ["\nUser:", "\nHuman:", "<|end|>", "<|endoftext|>"]
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and getattr(tokenizer, 'chat_template', None):
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # Generate
        logger.info(f"Generating response with max_tokens={max_tokens}, eos_token_id={eos_token_id}")
        
        # Note: mlx_lm.generate doesn't support eos_token_id in this version
        # We'll rely on stop string cleaning instead
        response_text = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        
        # Clean up response - remove special tokens and stop strings
        cleaned_response = response_text
        
        # Remove stop strings
        for stop_str in stop_strings:
            if stop_str in cleaned_response:
                cleaned_response = cleaned_response.split(stop_str)[0]
        
        # Remove <unk> tokens
        cleaned_response = cleaned_response.replace("<unk>", "").strip()
        
        # Remove any remaining special token patterns
        import re
        cleaned_response = re.sub(r'<\|[^|]+\|>', '', cleaned_response).strip()
        
        # Remove the prompt if it got echoed back
        if cleaned_response.startswith(formatted_prompt):
            cleaned_response = cleaned_response[len(formatted_prompt):].strip()
        
        # Count tokens in the cleaned response only
        tokens = tokenizer.encode(cleaned_response) if hasattr(tokenizer, 'encode') else []
        token_count = len(tokens)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "text": cleaned_response,
            "tokens": token_count,
            "response_time": response_time,
            "tokens_per_second": token_count / response_time if response_time > 0 else 0,
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.exception(e)
        raise
