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
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import threading
import queue

from mlx_lm import load, generate
from mlx_lm.lora import train_model
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters, print_trainable_parameters

from ..core.logging import get_logger
from ..config import get_settings
import re

# Logging
logger = get_logger(__name__)

# Minimum free disk space (in bytes) required before downloading a model
# Rough estimate: 8GB for 4B params + overhead
_MIN_FREE_SPACE_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB

# Maximum retries for transient network errors during download
_MAX_DOWNLOAD_RETRIES = 3


def _check_disk_space(path: Path, min_bytes: int = _MIN_FREE_SPACE_BYTES) -> bool:
    """Return True if at least min_bytes of free space exist on the filesystem."""
    try:
        stat = shutil.disk_usage(path)
        return stat.free >= min_bytes
    except OSError:
        return False


def _with_retry(func, retries: int = _MAX_DOWNLOAD_RETRIES, backoff: float = 2.0):
    """Call func() with simple exponential-backoff retry on transient errors."""
    import requests  # noqa: F401 — only used for exception matching
    last_exc = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            exc_name = type(exc).__name__
            is_transient = any(
                pat in exc_name.lower() or pat in str(exc).lower()
                for pat in ("connection", "timeout", "reset", "temporary", "network")
            )
            if not is_transient or attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            logger.warning(
                f"[MODEL RESOLUTION] Download attempt {attempt + 1} failed ({exc_name}). "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)
    raise last_exc

# (HF_HUB_CACHE is set in main.py before any huggingface_hub import)

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
    weight_decay: Optional[float] = None
    max_gradient_norm: Optional[float] = None
    architecture: str = "qwen2"
    lora_target_modules: Optional[List[str]] = None
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
    
    def __init__(self, training_process: 'TrainingProcess', steps_per_report: int = 10):
        self.training_process = training_process
        self.iteration_count = 0
        self.steps_per_report = steps_per_report
    
    def on_train_loss_report(self, train_info: Dict[str, Any]):
        """Called after training loss report."""
        self.iteration_count += 1
        
        # Check for stop/pause
        if self.training_process._check_should_stop():
            raise InterruptedError("Training stopped by user")
        
        while self.training_process._pause_event.is_set():
            if not self.training_process._is_paused:
                self.training_process._is_paused = True
                self.training_process.status = "paused"
                logger.info(f"Training {self.training_process.run_id} paused")
            time.sleep(0.5)
            if self.training_process._stop_event.is_set():
                raise InterruptedError("Training stopped while paused")
        
        # Get training info
        # mlx_lm calls this every steps_per_report iterations
        # So actual step = iteration_count * steps_per_report
        actual_step = self.iteration_count * self.steps_per_report
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
        
        # Dataset / run context surfaced to UI live logs
        self.dataset_context: Optional[Dict[str, Any]] = None
        
        # Callbacks
        self.on_step_complete: Optional[Callable[[Dict], None]] = None
        self.on_checkpoint_saved: Optional[Callable[[int, float], None]] = None
        self.on_training_complete: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_status_change: Optional[Callable[[str, str], None]] = None  # (status, message)
        
        # Control flags (thread-safe)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
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
                f.write("timestamp,step,loss,learning_rate,tokens_per_second,it_per_second,cpu_percent,memory_mb,peak_memory_mb,validation_loss\n")
        except Exception as e:
            logger.warning(f"Could not create detailed log file: {e}")

    def _write_detailed_log_entry(self, step: int, loss: float, learning_rate: float, tokens_per_sec: float, it_per_sec: float, resources: Dict):
        """Write detailed log entry."""
        try:
            timestamp = datetime.now().isoformat()
            cpu_percent = resources.get("cpu_percent", 0)
            memory_mb = resources.get("memory_mb", 0)
            peak_memory_mb = resources.get("peak_memory_mb", 0)
            val_loss = self.validation_loss if self.validation_loss is not None else ""

            with open(self.detailed_log_path, 'a') as f:
                f.write(f"{timestamp},{step},{loss:.6f},{learning_rate:.2e},{tokens_per_sec:.2f},{it_per_sec:.2f},{cpu_percent:.1f},{memory_mb:.1f},{peak_memory_mb:.1f},{val_loss}\n")
        except Exception as e:
            logger.warning(f"Could not write to detailed log: {e}")

    def _check_should_stop(self) -> bool:
        """Check if training should stop."""
        return self._stop_event.is_set()

    def _check_should_pause(self) -> bool:
        """Check if training should pause."""
        if self._pause_event.is_set() and not self._is_paused:
            self._is_paused = True
            self.status = "paused"
            logger.info(f"Training {self.run_id} paused")
            return True
        return False

    def _resume_from_pause(self):
        """Resume training from pause."""
        self._pause_event.clear()
        self._is_paused = False
        self.status = "running"
        logger.info(f"Training {self.run_id} resumed")

    DOWNLOAD_ALLOW_PATTERNS = [
        "*.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "generation_config.json",
        "model.safetensors.index.json",
    ]

    def _resolve_model_path(self, model_id: str) -> str:
        """Return local snapshot path for model_id, downloading if needed.

        Uses the native HuggingFace cache (controlled via HF_HUB_CACHE env var).
        Delegates all caching, deduplication, resume, and validation to
        huggingface_hub.snapshot_download. Returns the snapshot directory path
        that can be passed directly to mlx_lm.load().
        """
        from huggingface_hub import snapshot_download, login as hf_login
        from huggingface_hub.utils import LocalEntryNotFoundError
        from ..config import get_settings
        from ..core.model_architectures import _is_model_complete_sync, _get_hf_cache_root

        settings = get_settings()
        hf_token = settings.hf_token or None

        if hf_token:
            try:
                hf_login(token=hf_token)
            except Exception:
                pass

        # 1. Check if already cached AND complete (weight files present, not just metadata)
        try:
            cached_path = snapshot_download(
                repo_id=model_id,
                allow_patterns=self.DOWNLOAD_ALLOW_PATTERNS,
                local_files_only=True,
                token=hf_token,
            )
            if _is_model_complete_sync(model_id):
                logger.info(f"[MODEL RESOLUTION] Using cached model: {cached_path}")
                return cached_path
            else:
                logger.warning(
                    f"[MODEL RESOLUTION] Model snapshot exists but weight files are incomplete "
                    f"for {model_id}. Will re-download..."
                )
        except LocalEntryNotFoundError:
            logger.info(f"[MODEL RESOLUTION] Model not in local cache, will download...")

        # Stop check before downloading
        if self._check_should_stop():
            raise RuntimeError("Download stopped by user")

        # --- Phase 4: Robustness ---
        # 2. Disk-space guard
        cache_root = _get_hf_cache_root()
        if not _check_disk_space(cache_root):
            raise RuntimeError(
                f"Not enough free disk space to download {model_id}. "
                f"At least {_MIN_FREE_SPACE_BYTES / (1024**3):.0f} GB is required. "
                f"Please free up space and try again."
            )

        self._update_status("downloading", f"Downloading {model_id}...")

        # 3. Download with retry for transient network errors
        def _do_download():
            return snapshot_download(
                repo_id=model_id,
                allow_patterns=self.DOWNLOAD_ALLOW_PATTERNS,
                resume_download=True,
                token=hf_token,
            )

        try:
            downloaded_path = _with_retry(_do_download, retries=_MAX_DOWNLOAD_RETRIES)
        except Exception as exc:
            # If retries exhausted, surface a helpful message
            exc_name = type(exc).__name__
            if any(p in exc_name.lower() or p in str(exc).lower() for p in ("connection", "timeout", "reset")):
                raise RuntimeError(
                    f"Network error while downloading {model_id}: {exc}. "
                    f"Please check your internet connection and try again."
                )
            if "disk" in str(exc).lower() or "space" in str(exc).lower():
                raise RuntimeError(
                    f"Disk full or write error while downloading {model_id}: {exc}. "
                    f"Please free up disk space and try again."
                )
            raise

        if self._check_should_stop():
            raise RuntimeError("Download stopped by user")

        # 4. Verify download actually produced complete weight files
        if not _is_model_complete_sync(model_id):
            logger.error(
                f"[MODEL RESOLUTION] Download returned path {downloaded_path} but "
                f"weight files are still missing for {model_id}."
            )
            raise RuntimeError(
                f"Model download incomplete for {model_id}. The weight files are missing. "
                f"Please remove the model from My Models and re-add it to trigger a fresh download. "
                f"You can also manually delete the cache directory: "
                f"{cache_root}/models--{model_id.replace('/', '--')}"
            )

        logger.info(f"[MODEL RESOLUTION] Downloaded model to: {downloaded_path}")
        return downloaded_path

    async def train(self):
        """
        Main training loop using mlx_lm.
        """
        try:
            self.status = "running"
            self.start_time = datetime.now()
            
            # Apply resource limits
            await asyncio.to_thread(self._apply_resource_limits)
            
            logger.info("=" * 70)
            logger.info(f"[TRAINING START] Run ID: {self.run_id}")
            logger.info(f"[TRAINING CONFIG] Requested Model: {self.config.model_id}")
            logger.info(f"[TRAINING CONFIG] Steps: {self.config.steps}")
            logger.info(f"[TRAINING CONFIG] Data path: {self.config.data_path}")
            logger.info(f"[TRAINING CONFIG] Output path: {self.config.output_path}")
            logger.info("=" * 70)
            
            # Resolve model path via native HuggingFace cache
            logger.info(f"[MODEL RESOLUTION] Expected model: {self.config.model_id}")
            self.model_path = await asyncio.to_thread(
                self._resolve_model_path, self.config.model_id
            )
            logger.info(f"[MODEL RESOLUTION] Using model at: {self.model_path}")
            
            # Log final model path before loading
            logger.info("=" * 70)
            logger.info(f"[MODEL LOADING] Final model_path: {self.model_path}")
            logger.info(f"[MODEL LOADING] Config model_id: {self.config.model_id}")
            logger.info("=" * 70)
            
            # Load model
            logger.info("[MODEL LOADING] Loading model into memory...")
            self._update_status("loading_model", f"Loading {self.config.model_id} into memory...")
            settings = get_settings()
            if settings.allow_remote_code:
                logger.warning(
                    "SECURITY: allow_remote_code is enabled. "
                    "Custom tokenizer code from the model repository will be executed."
                )
            try:
                self.model, self.tokenizer = await asyncio.to_thread(
                    load,
                    self.model_path,
                    tokenizer_config={"trust_remote_code": settings.allow_remote_code}
                )
            except (FileNotFoundError, OSError) as load_exc:
                # Common cause: snapshot exists but *.safetensors weight files are missing
                # (interrupted download, or snapshot_download returned path to incomplete cache)
                err_msg = str(load_exc)
                if "safetensors" in err_msg.lower() or "no such file" in err_msg.lower():
                    logger.error(
                        f"[MODEL LOADING] Model weight files missing for {self.config.model_id} "
                        f"at {self.model_path}. Error: {err_msg}"
                    )
                    raise RuntimeError(
                        f"Model download appears incomplete for '{self.config.model_id}'. "
                        f"The weight files (*.safetensors) are missing. "
                        f"Please remove the model from 'My Models' and re-add it to trigger a fresh download."
                    ) from load_exc
                raise
            
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
            if self.config.weight_decay is not None:
                args.optimizer_config["adam"]["weight_decay"] = self.config.weight_decay
                args.optimizer_config["adamw"]["weight_decay"] = self.config.weight_decay
            args.max_grad_norm = self.config.max_gradient_norm
            args.clear_cache_threshold = 0
            args.lr_schedule = None  # Disable mlx_lm's built-in schedule, we handle it manually in callback
            from ..core.model_architectures import get_lora_keys, validate_lora_keys_against_model

            lora_keys = self.config.lora_target_modules or get_lora_keys(self.config.architecture)
            lora_keys = validate_lora_keys_against_model(self.model, lora_keys)
            logger.info(f"LoRA target keys (validated against model): {lora_keys}")
            args.lora_parameters = {
                "keys": lora_keys,
                "rank": self.config.lora_rank,
                "dropout": self.config.lora_dropout,
                "scale": self.config.lora_alpha / self.config.lora_rank
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
                
                # Build dataset context for UI live logs
                val_file = os.path.join(data_dir, "valid.jsonl")
                val_count = 0
                if os.path.exists(val_file):
                    with open(val_file, 'r') as vf:
                        val_count = sum(1 for line in vf if line.strip())
                
                self.dataset_context = {
                    "samples": len(train_set),
                    "batch_size": args.batch_size,
                    "format": "chat" if hasattr(train_set, '_data') and train_set._data and 'messages' in str(train_set._data[0]) else "alpaca",
                    "validation_samples": val_count,
                    "data_dir": data_dir,
                }
                logger.info(f"[DATASET CONTEXT] {self.dataset_context}")
                
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
            training_callback = MLXTrainingCallback(self, steps_per_report=args.steps_per_report)
            
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
            
            if self.on_error:
                try:
                    self.on_error("stopped:Training stopped by user")
                except Exception as callback_error:
                    logger.error(f"Error in stopped callback: {callback_error}")
            
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
        self._pause_event.set()
        logger.info(f"Pause requested for training {self.run_id}")
    
    def resume(self):
        """Resume training from pause."""
        self._pause_event.clear()
        if self._is_paused:
            self._resume_from_pause()
        logger.info(f"Resume requested for training {self.run_id}")
    
    def stop(self):
        """Stop training."""
        self._stop_event.set()
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
            "dataset_context": self.dataset_context,
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
    export_format: str,
    output_path: str,
    hyperparameters: Optional[Dict] = None,
    lora_target_modules: Optional[list] = None
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
                "target_modules": lora_target_modules or ["q_proj", "v_proj"],
                "num_layers": hyperparameters.get('num_lora_layers', 8) if hyperparameters else 8,
                "base_model_name_or_path": model_path
            }
            import json
            with open(config_path, 'w') as f:
                json.dump(adapter_config, f, indent=2)
            logger.info(f"Created/updated adapter config at {config_path}")
            
            # Load base model with adapters
            logger.info("Loading base model with adapters...")
            settings = get_settings()
            if settings.allow_remote_code:
                logger.warning(
                    "SECURITY: allow_remote_code is enabled. "
                    "Custom tokenizer code from the model repository will be executed."
                )
            model, tokenizer = load(
                model_path,
                tokenizer_config={"trust_remote_code": settings.allow_remote_code},
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
            raise NotImplementedError(
                "GGUF export is not yet supported. "
                "Export the fused model and convert manually using llama.cpp: "
                f"python convert.py {output_path} --outfile model.gguf"
            )
        
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
        
        loop = asyncio.get_running_loop()
        
        def _load():
            settings = get_settings()
            if settings.allow_remote_code:
                logger.warning(
                    "SECURITY: allow_remote_code is enabled. "
                    "Custom tokenizer code from the model repository will be executed."
                )
            model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": settings.allow_remote_code})
            if adapter_path and os.path.exists(adapter_path):
                logger.info(f"Loading adapters from {adapter_path}")
                load_adapters(model, adapter_path)
            return (model, tokenizer)
        
        result = await loop.run_in_executor(None, _load)
        
        logger.info("Model loaded successfully for inference")
        return result
        
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
    architecture: str = "qwen2",
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
        
        # Get stop strings based on architecture
        from ..core.model_architectures import get_stop_strings
        if hasattr(tokenizer, 'apply_chat_template'):
            stop_strings = get_stop_strings(architecture)
        else:
            stop_strings = ["\nUser:", "\nHuman:", "<|end|>"]
        
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
            # Fallback for models without chat template - use architecture-specific template
            from ..core.model_architectures import get_chat_template_fallback
            fallback_template = get_chat_template_fallback(architecture)
            if hasattr(tokenizer, 'apply_chat_template') and fallback_template:
                # Use Jinja-like template substitution for basic cases
                # Most architecture fallbacks use {% for %} loops which need a real Jinja engine.
                # As a safe fallback, use the generic format but log the missing template.
                logger.warning(
                    f"No chat_template found for {architecture}. "
                    f"Generation may be suboptimal. Consider using a model with a built-in template."
                )
                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            else:
                formatted_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # Generate
        logger.info(f"Generating response with max_tokens={max_tokens}, eos_token_id={eos_token_id}")
        
        loop = asyncio.get_running_loop()
        
        def _generate():
            return generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                verbose=False
            )
        
        response_text = await loop.run_in_executor(None, _generate)
        
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
