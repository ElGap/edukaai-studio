"""
Training Monitor Module

Monitors the training subprocess and streams updates to the UI.
"""

import subprocess
import threading
import queue
import re
import sys
import time
import os
import json
import shutil
from pathlib import Path

# Import debug logger
try:
    from .debug_logger import (
        log_debug, log_info, log_warning, log_error,
        log_subprocess_start, log_subprocess_output, 
        log_subprocess_exit, log_exception
    )
    DEBUG_LOGGING = True
except ImportError:
    DEBUG_LOGGING = False
    print("[MONITOR] Debug logging not available")

# Import resource monitor
try:
    from ui.resource_monitor import ResourceMonitor, ResourceStats
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError:
    RESOURCE_MONITORING_AVAILABLE = False
    print("[MONITOR] Resource monitoring not available")


# ============ SECURITY VALIDATION FUNCTIONS ============

def validate_model_id(model_id: str) -> str:
    """Validate HuggingFace model ID to prevent command injection.
    
    Args:
        model_id: The model identifier to validate (e.g., 'org/model-name')
        
    Returns:
        The validated model ID
        
    Raises:
        ValueError: If model ID contains invalid characters or patterns
        
    Examples:
        >>> validate_model_id("mlx-community/Phi-3-mini-4k-instruct-4bit")
        "mlx-community/Phi-3-mini-4k-instruct-4bit"
        
        >>> validate_model_id("model; rm -rf /")
        ValueError: Invalid model ID format
    """
    if not model_id or not isinstance(model_id, str):
        raise ValueError("Model ID must be a non-empty string")
    
    # Strip and check if empty after stripping
    stripped = model_id.strip()
    if not stripped:
        raise ValueError("Model ID must be a non-empty string")
    
    # Check for shell metacharacters that could enable command injection
    dangerous_chars = [';', '&', '|', '`', '$', '(', ')', '<', '>', '\\', '{', '}', '[', ']']
    for char in dangerous_chars:
        if char in model_id:
            raise ValueError(
                f"Invalid model ID: contains dangerous character '{char}'. "
                f"Model IDs should only contain alphanumeric characters, hyphens, "
                f"underscores, and forward slashes (for organization/model format). "
                f"Example: 'mlx-community/Phi-3-mini-4k-instruct-4bit'"
            )
    
    # Also check for command substitution patterns: $() and ${}
    if '$(' in model_id or '${' in model_id:
        raise ValueError(
            "Invalid model ID: contains command substitution pattern. "
            "Model IDs cannot contain shell command substitution like '$()' or '${}'. "
            "Example: 'mlx-community/Phi-3-mini-4k-instruct-4bit'"
        )
    
    # Validate format: organization/model-name or just model-name
    # Allow alphanumeric, hyphens, underscores, dots (for versions), and forward slashes
    import re
    pattern = r'^[\w\-.]+(/[\w\-.]+)*$'
    if not re.match(pattern, model_id):
        raise ValueError(
            f"Invalid model ID format: '{model_id}'. "
            f"Expected format: 'organization/model-name' or 'model-name'. "
            f"Allowed characters: alphanumeric, hyphens, underscores, dots."
        )
    
    return model_id


def validate_training_file(filepath: str, max_size_mb: int = 100) -> bool:
    """Validate uploaded training file for security and format.
    
    Checks:
    - File size limit (prevent DoS)
    - File extension (JSON/JSONL only)
    - JSON structure validity
    - Content type (JSON array or JSONL)
    
    Args:
        filepath: Path to the uploaded training file
        max_size_mb: Maximum allowed file size in MB (default: 100)
        
    Returns:
        True if file is valid
        
    Raises:
        ValueError: If file fails any validation check
        
    Examples:
        >>> validate_training_file("/path/to/train.jsonl")
        True
        
        >>> validate_training_file("/path/to/huge.zip")
        ValueError: File size exceeds limit
    """
    path = Path(filepath)
    
    # Check file exists
    if not path.exists():
        raise ValueError(f"File not found: {filepath}")
    
    # Check file size (prevent DoS via huge files)
    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = path.stat().st_size
    if file_size > max_size_bytes:
        raise ValueError(
            f"File too large: {file_size / (1024*1024):.1f}MB exceeds limit of {max_size_mb}MB. "
            f"Please reduce training data size or split into smaller files."
        )
    
    # Check file extension
    allowed_extensions = {'.json', '.jsonl', '.txt'}
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Invalid file type: {path.suffix}. "
            f"Allowed types: {', '.join(allowed_extensions)}. "
            f"Training data must be JSON array, JSONL, or plain text."
        )
    
    # Validate JSON structure
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            if not content:
                raise ValueError("File is empty")
            
            # Check if it's a JSON array
            if content.startswith('['):
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("JSON array expected, got other type")
                if len(data) == 0:
                    raise ValueError("JSON array is empty")
            elif content.startswith('{'):
                # Could be single JSON object or JSONL with first line starting with {
                # Try parsing first non-empty line to determine format
                first_line = None
                for line in content.split('\n'):
                    if line.strip():
                        first_line = line.strip()
                        break
                
                if first_line:
                    try:
                        # Try parsing as single object first
                        json.loads(content)
                        # If that works, it's a single JSON object
                    except json.JSONDecodeError:
                        # If single parse fails, check if it's JSONL
                        # by trying to parse first line only
                        try:
                            json.loads(first_line)
                            # It's JSONL - validate each line separately below
                            pass
                        except json.JSONDecodeError:
                            raise ValueError("Invalid JSON format")
                
                # If we get here and single parse failed, validate as JSONL
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    # Single parse failed, treat as JSONL
                    lines = content.split('\n')
                    for i, line in enumerate(lines[:10]):
                        if line.strip():
                            try:
                                json.loads(line)
                            except json.JSONDecodeError:
                                raise ValueError(f"Invalid JSON on line {i+1}: {line[:50]}...")
            else:
                # Assume JSONL - validate each line
                lines = content.split('\n')
                if len(lines) == 0:
                    raise ValueError("No data lines found")
                for i, line in enumerate(lines[:10]):  # Check first 10 lines
                    if line.strip():
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON on line {i+1}: {line[:50]}...")
    except UnicodeDecodeError:
        raise ValueError("File encoding error: must be valid UTF-8 text")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    
    return True


def validate_data_path(data_path: str, allowed_base: Path = None) -> Path:
    """Validate training data path to prevent directory traversal attacks.
    
    Args:
        data_path: The path to validate
        allowed_base: Base directory that must contain the path (default: any temp dir)
        
    Returns:
        The resolved, validated Path object
        
    Raises:
        ValueError: If path is outside allowed directory or contains traversal
    """
    if not data_path or not isinstance(data_path, str):
        raise ValueError("Data path must be a non-empty string")
    
    # Resolve to absolute path
    try:
        resolved = Path(data_path).resolve()
    except Exception as e:
        raise ValueError(f"Invalid path format: {data_path}. Error: {e}")
    
    # Check for directory traversal patterns
    path_str = str(resolved)
    
    # If allowed_base specified, validate path is within it
    if allowed_base is not None:
        allowed_resolved = allowed_base.resolve()
        if not path_str.startswith(str(allowed_resolved)):
            raise ValueError(
                f"Path traversal attempt detected: '{data_path}' resolves to '{resolved}' "
                f"which is outside allowed directory '{allowed_base}'. "
                f"Training data must be in the designated training directory."
            )
    
    # Additional safety: reject paths with suspicious patterns
    suspicious_patterns = ['..', '~', '$HOME', '$PWD', '${', '&&', '||', '`', ';']
    for pattern in suspicious_patterns:
        if pattern in data_path:
            raise ValueError(
                f"Invalid path: contains suspicious pattern '{pattern}'. "
                f"Training data paths cannot contain shell expansion or traversal patterns."
            )
    
    return resolved



class TrainingMonitor:
    """Monitors training process and streams updates to UI."""
    
    def __init__(self, output_queue, progress_queue):
        """
        Initialize the monitor.
        
        Args:
            output_queue: Queue for log lines
            progress_queue: Queue for progress updates (dicts)
        """
        self.output_queue = output_queue
        self.progress_queue = progress_queue
        self.process = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.training_complete = False
        self.was_stopped = False  # Track if training was stopped by user
        self.output_dir = None
        
        # Initialize resource monitor
        self.resource_monitor = None
        if RESOURCE_MONITORING_AVAILABLE:
            try:
                self.resource_monitor = ResourceMonitor()
                print("[MONITOR] Resource monitoring enabled")
            except Exception as e:
                print(f"[MONITOR] Resource monitor init failed: {e}")
        
    def _convert_data_format(self, source_file, dest_file, model_name=None):
        """
        Convert Alpaca format to mlx_lm compatible text format.
        """
        print(f"[DATA] Converting {source_file} -> {dest_file}")
        
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Check if it's a JSON array
            if content.startswith('['):
                data_array = json.loads(content)
            else:
                # JSONL - parse each line
                data_array = [json.loads(line) for line in content.split('\n') if line.strip()]
            
            # Convert to mlx_lm text format
            converted_count = 0
            with open(dest_file, 'w', encoding='utf-8') as outfile:
                for item in data_array:
                    try:
                        # Alpaca format -> text format
                        if 'instruction' in item and 'output' in item:
                            instruction = item.get('instruction', '')
                            input_text = item.get('input', '')
                            output = item.get('output', '')
                            
                            # Create text field in mlx_lm format
                            if input_text and input_text.strip():
                                text = f"Instruction: {instruction}\n\nInput: {input_text}\n\nOutput: {output}"
                            else:
                                text = f"Instruction: {instruction}\n\nOutput: {output}"
                            
                            outfile.write(json.dumps({"text": text}) + '\n')
                            converted_count += 1
                        
                        # Already in text format
                        elif 'text' in item:
                            outfile.write(json.dumps(item) + '\n')
                            converted_count += 1
                        
                        # Chat format
                        elif 'messages' in item:
                            messages = item['messages']
                            text_parts = []
                            for msg in messages:
                                role = msg.get('role', '')
                                content = msg.get('content', '')
                                if role == 'user':
                                    text_parts.append(f"User: {content}")
                                elif role == 'assistant':
                                    text_parts.append(f"Assistant: {content}")
                                elif role == 'system':
                                    text_parts.append(f"System: {content}")
                            
                            text = "\n\n".join(text_parts)
                            outfile.write(json.dumps({"text": text}) + '\n')
                            converted_count += 1
                            
                    except Exception as e:
                        print(f"[DATA] Error converting item: {e}")
                        continue
            
            print(f"[DATA] Converted {converted_count} samples to mlx_lm text format")
            return True
                
        except Exception as e:
            print(f"[DATA] Error converting file: {e}")
            return False
        
    def start_training(self, args_dict, data_file, validation_file=None, validation_strategy='auto_split', validation_split_percentage=10):
        """
        Start training process in background.
        
        Args:
            args_dict: Dictionary of training arguments
            data_file: Path to training data file
            validation_file: Optional path to validation data file (if strategy is 'upload_own')
            validation_strategy: 'upload_own', 'auto_split', or 'no_validation'
            validation_split_percentage: Percentage to split for validation (5-25, default 10)
            
        Returns:
            bool: True if training started successfully
        """
        try:
            print(f"\n[MONITOR] Starting training with args: {args_dict}")
            print(f"[MONITOR] Data file: {data_file}")
            print(f"[MONITOR] Validation strategy: {validation_strategy}")
            
            # DEBUG: Log detailed startup info
            if DEBUG_LOGGING:
                log_info("=" * 80)
                log_info("START TRAINING CALLED")
                log_info("=" * 80)
                log_info(f"args_dict: {json.dumps(args_dict, indent=2, default=str)}")
                log_info(f"data_file: {data_file}")
                log_info(f"validation_strategy: {validation_strategy}")
                log_info(f"validation_split_percentage: {validation_split_percentage}")
                log_info(f"Python executable: {sys.executable}")
                log_info(f"Working directory: {Path.cwd()}")
            
            if validation_file:
                print(f"[MONITOR] Validation file: {validation_file}")
            elif validation_strategy == 'auto_split':
                print(f"[MONITOR] Auto-split percentage: {validation_split_percentage}%")
            
            # SECURITY: Validate uploaded files before processing
            try:
                print("[SECURITY] Validating training data file...")
                validate_training_file(data_file, max_size_mb=100)
                print(f"[SECURITY] Training file validated: {data_file}")
                if DEBUG_LOGGING:
                    log_info(f"Training file validated: {data_file}")
                
                if validation_file:
                    print("[SECURITY] Validating validation data file...")
                    validate_training_file(validation_file, max_size_mb=50)
                    print(f"[SECURITY] Validation file validated: {validation_file}")
                    
            except ValueError as e:
                error_msg = f"[SECURITY ERROR] File validation failed: {e}"
                print(error_msg)
                if DEBUG_LOGGING:
                    log_error(error_msg)
                self.output_queue.put(error_msg)
                return False
            
            # ===== RESET MONITOR STATE FOR NEW TRAINING =====
            self.training_complete = False
            self.stop_event.clear()
            self.pause_event.clear()
            print(f"[MONITOR] Reset training state: complete=False, events cleared")
            if DEBUG_LOGGING:
                log_info("Reset training state: complete=False")
            
            # Create a temp data directory with proper file names
            import tempfile
            import shutil
            
            data_dir = Path(data_file).parent
            temp_data_dir = tempfile.mkdtemp(prefix="training_data_")
            
            # Copy uploaded file as train.jsonl (with format conversion)
            train_path = Path(temp_data_dir) / "train.jsonl"
            model_name = args_dict.get('model', '')
            conversion_success = self._convert_data_format(data_file, train_path, model_name)
            if not conversion_success:
                print(f"[MONITOR] Warning: Failed to convert data format, trying direct copy")
                shutil.copy(data_file, train_path)
            print(f"[MONITOR] Copied training data to: {train_path}")
            
            # Handle validation based on strategy
            val_path = Path(temp_data_dir) / "valid.jsonl"
            
            if validation_strategy == 'upload_own' and validation_file:
                # User provided separate validation file
                print("[MONITOR] Using user-provided validation file")
                conversion_success = self._convert_data_format(validation_file, val_path, model_name)
                if not conversion_success:
                    shutil.copy(validation_file, val_path)
                print(f"[MONITOR] Copied validation data to: {val_path}")
                
            elif validation_strategy == 'no_validation':
                # No validation - create minimal validation file with 1 sample from training
                # (mlx_lm.lora requires valid.jsonl to exist even if we don't use it)
                print("[MONITOR] No validation requested - creating minimal validation file")
                
                # Read first training sample and use as validation
                with open(train_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        with open(val_path, 'w', encoding='utf-8') as vf:
                            vf.write(first_line + '\n')
                        print(f"[MONITOR] Created minimal validation file (1 sample) at: {val_path}")
                    
            else:
                # Auto-split from training data (default behavior)
                print(f"[MONITOR] Auto-splitting {validation_split_percentage}% from training data")
                
                # Calculate split
                with open(data_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content.startswith('['):
                    # JSON array format
                    data_array = json.loads(content)
                    total = len(data_array)
                    val_count = max(1, int(total * validation_split_percentage / 100))
                    
                    # Take samples evenly distributed
                    step = max(1, total // val_count)
                    val_indices = list(range(0, total, step))[:val_count]
                    val_samples = [data_array[i] for i in val_indices]
                else:
                    # JSONL format
                    lines = [l for l in content.split('\n') if l.strip()]
                    total = len(lines)
                    val_count = max(1, int(total * validation_split_percentage / 100))
                    
                    # Take samples evenly distributed
                    step = max(1, total // val_count)
                    val_indices = list(range(0, total, step))[:val_count]
                    val_samples = [json.loads(lines[i]) for i in val_indices]
                
                # Convert and write validation samples
                converted_count = 0
                with open(val_path, 'w', encoding='utf-8') as f:
                    for item in val_samples:
                        try:
                            # Convert Alpaca format -> text format for mlx_lm
                            if 'instruction' in item and 'output' in item:
                                instruction = item.get('instruction', '')
                                input_text = item.get('input', '')
                                output = item.get('output', '')
                                
                                if input_text and input_text.strip():
                                    text = f"Instruction: {instruction}\n\nInput: {input_text}\n\nOutput: {output}"
                                else:
                                    text = f"Instruction: {instruction}\n\nOutput: {output}"
                                
                                f.write(json.dumps({"text": text}) + '\n')
                                converted_count += 1
                            elif 'text' in item:
                                f.write(json.dumps(item) + '\n')
                                converted_count += 1
                        except Exception as e:
                            print(f"[MONITOR] Error writing validation sample: {e}")
                            continue
                
                print(f"[MONITOR] Created validation data ({converted_count} samples, {validation_split_percentage}% split) at: {val_path}")
            
            # SECURITY: Validate inputs before subprocess execution
            try:
                model_id = args_dict.get('model', 'mlx-community/Phi-3-mini-4k-instruct-4bit')
                validated_model = validate_model_id(model_id)
                print(f"[SECURITY] Model ID validated: {validated_model}")
                
                # Validate data path is within temp directory
                temp_path = Path(temp_data_dir)
                validated_data_path = validate_data_path(temp_data_dir, allowed_base=temp_path.parent)
                print(f"[SECURITY] Data path validated: {validated_data_path}")
                
            except ValueError as e:
                error_msg = f"[SECURITY ERROR] Input validation failed: {e}"
                print(error_msg)
                self.output_queue.put(error_msg)
                return False
            
            # Prepare command - call lora-train.py from scripts directory
            # Go up from ui/ -> edukaai_studio/ -> src/ -> project root -> scripts/
            project_root = Path(__file__).parent.parent.parent.parent
            lora_train_script = project_root / "scripts" / "lora-train.py"
            
            if not lora_train_script.exists():
                # Fallback to old location for backward compatibility
                lora_train_script = project_root / "lora-train.py"
            
            cmd = [
                sys.executable,
                str(lora_train_script),
                "--model", validated_model,
                "--iters", str(args_dict.get('iters', 450)),
                "--learning-rate", str(args_dict.get('learning_rate', 1e-4)),
                "--batch-size", str(args_dict.get('batch_size', 1)),
                "--grad-accumulation-steps", str(args_dict.get('grad_accumulation', 4)),
                "--data", temp_data_dir,
                "--steps-per-eval", "50",
                "--save-every", "50",
                "--validation-strategy", validation_strategy,
            ]
            
            # Add LoRA parameters
            lora_rank = args_dict.get('lora_rank', 16)
            lora_alpha = args_dict.get('lora_alpha', 32)
            lora_dropout = args_dict.get('lora_dropout', 0.0)
            
            cmd.extend([
                "--lora-rank", str(lora_rank),
                "--lora-alpha", str(lora_alpha),
                "--lora-dropout", str(lora_dropout)
            ])
            
            print(f"[MONITOR] LoRA Config: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
            
            print(f"[MONITOR] Command: {' '.join(cmd)}")
            
            # Add optional args
            if args_dict.get('max_seq_length'):
                cmd.extend(["--max-seq-length", str(args_dict['max_seq_length'])])
            
            # Handle early stopping - try different key names
            early_stopping = args_dict.get('early_stopping') or args_dict.get('early_stopping_patience', 2)
            if early_stopping > 0:
                cmd.extend(["--early-stopping-patience", str(early_stopping)])
            
            if args_dict.get('grad_checkpoint'):
                cmd.append("--grad-checkpoint")
            
            # Check if model is already cached
            model_name = args_dict.get('model', 'unknown')
            cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
            model_cached = False
            
            # Check if model exists in cache (models--{org}--{model} format)
            if model_name and model_name != 'unknown':
                # Convert model_id to cache directory name format
                # e.g., "mlx-community/Phi-3-mini-4k-instruct-4bit" -> "models--mlx-community--Phi-3-mini-4k-instruct-4bit"
                cache_model_name = model_name.replace('/', '--')
                cache_model_dir = os.path.join(cache_dir, f"models--{cache_model_name}")
                
                if os.path.exists(cache_model_dir):
                    model_cached = True
                    print(f"[MONITOR] Model found in cache: {cache_model_dir}")
                else:
                    print(f"[MONITOR] Model not in cache, will download to: {cache_dir}")
            
            # Notify user about model loading
            if model_cached:
                load_msg = f"📦 Loading cached model: {model_name}"
                cache_msg = "   (Model is already downloaded - loading from cache)"
            else:
                load_msg = f"⏳ Loading model from HuggingFace: {model_name}"
                cache_msg = "   (This may take a while for first download - models are cached for future use)"
            
            start_msg = "   Starting mlx_lm.lora training process..."
            print(f"[MONITOR] {load_msg}")
            print(f"[MONITOR] {cache_msg}")
            print(f"[MONITOR] {start_msg}")
            self.output_queue.put(load_msg)
            self.output_queue.put(cache_msg)
            self.output_queue.put(start_msg)
            
            # Start subprocess with line buffering
            print("[MONITOR] Starting subprocess...")
            
            # Set unbuffered mode for Python script itself and HF token if available
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Set HF_TOKEN for gated model access during download
            hf_token = args_dict.get('hf_token')
            if hf_token:
                env['HF_TOKEN'] = hf_token
                print(f"[MONITOR] HF_TOKEN set for gated model access")
            
            # DEBUG: Log subprocess details (AFTER env is created)
            if DEBUG_LOGGING:
                log_info("=" * 80)
                log_info("SUBPROCESS STARTUP")
                log_info("=" * 80)
                log_info(f"Command: {' '.join(str(c) for c in cmd)}")
                log_info(f"Working directory: {Path.cwd()}")
                log_info(f"Python executable: {sys.executable}")
                log_info(f"Script exists: {Path(cmd[1]).exists()}")
                log_info(f"Script path: {cmd[1]}")
                log_info("Environment variables:")
                for key, value in env.items():
                    if 'token' in key.lower() or 'key' in key.lower() or 'secret' in key.lower():
                        log_info(f"  {key}: [MASKED]")
                    elif key.startswith('PYTHON') or key.startswith('HF_'):
                        log_info(f"  {key}: {value}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered (best for text mode)
                universal_newlines=True,
                env=env,
                # Security: Prevent shell injection
                shell=False
            )
            
            # DEBUG: Log process info
            if DEBUG_LOGGING:
                log_info(f"Process started with PID: {self.process.pid}")
                log_info(f"Process stdin: {self.process.stdin}")
                log_info(f"Process stdout: {self.process.stdout}")
                log_info(f"Process stderr: {self.process.stderr}")
            
            # Set up resource monitoring and limits
            max_training_time = 4 * 60 * 60  # 4 hours default timeout
            max_memory_gb = 16  # 16GB memory limit (configurable)
            
            print(f"[MONITOR] Subprocess started with PID: {self.process.pid}")
            
            # Quick check if process is still running after 100ms
            time.sleep(0.1)
            
            # DEBUG: Check process status
            poll_result = self.process.poll()
            if DEBUG_LOGGING:
                log_info(f"Process poll after 100ms: {poll_result}")
            
            if poll_result is not None:
                exit_code = poll_result
                print(f"[MONITOR] ERROR: Subprocess exited immediately with code {exit_code}!")
                
                # DEBUG: Try to capture any output
                if DEBUG_LOGGING:
                    log_error(f"Subprocess exited immediately with code {exit_code}")
                    try:
                        # Try to read any stdout that might have been produced
                        stdout_data = self.process.stdout.read() if self.process.stdout else ""
                        if stdout_data:
                            log_error(f"Stdout before exit: {stdout_data[:2000]}")
                        else:
                            log_error("No stdout data available")
                    except Exception as e:
                        log_error(f"Error reading stdout: {e}")
                
                # Try to get any error output
                try:
                    stderr_output = self.process.stderr.read() if self.process.stderr else ""
                    if stderr_output:
                        print(f"[MONITOR] stderr: {stderr_output[:500]}")
                except:
                    pass
                self.output_queue.put(f"[ERROR] Training process failed to start (exit code: {exit_code})")
                return False
            
            print(f"[SECURITY] Resource limits: max_time={max_training_time//3600}h, max_memory={max_memory_gb}GB")
            
            # DEBUG: Log thread startup
            if DEBUG_LOGGING:
                log_info("Starting monitoring threads...")
            
            # Start resource watchdog thread
            self._start_resource_watchdog(max_training_time, max_memory_gb)
            
            # Get total iterations from args for progress calculation
            total_iters = args_dict.get('iters', 450)
            print(f"[MONITOR] Expected total iterations: {total_iters}")
            
            # Start monitoring threads
            self.stop_event.clear()
            self.pause_event.clear()
            
            self._output_thread = threading.Thread(target=self._read_output, daemon=True)
            self._output_thread.start()
            self._progress_thread = threading.Thread(target=self._parse_progress, args=(total_iters,), daemon=True)
            self._progress_thread.start()
            
            print("[MONITOR] Monitoring threads started")
            
            # DEBUG: Log success
            if DEBUG_LOGGING:
                log_info("=" * 80)
                log_info("TRAINING STARTED SUCCESSFULLY")
                log_info("=" * 80)
            
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"[ERROR] Error starting training: {str(e)}\n{traceback.format_exc()}"
            print(f"[MONITOR] {error_msg}")
            self.output_queue.put(error_msg)
            return False
    
    def _read_output(self):
        """Read output and put in queue for UI."""
        try:
            print(f"[READ_OUTPUT] Starting to read from stdout, PID: {self.process.pid if self.process else 'None'}")
            
            # DEBUG: Log start of reading
            if DEBUG_LOGGING:
                log_info("=" * 80)
                log_info("STARTING OUTPUT READ LOOP")
                log_info("=" * 80)
                log_info(f"Process PID: {self.process.pid if self.process else 'None'}")
                log_info(f"Process stdout: {self.process.stdout}")
            
            line_count = 0
            last_download_update = 0
            first_lines = []  # Store first few lines for debugging
            
            for line in self.process.stdout:
                if self.stop_event.is_set():
                    if DEBUG_LOGGING:
                        log_info("Stop event set, breaking read loop")
                    break
                
                # Check if paused
                while self.pause_event.is_set() and not self.stop_event.is_set():
                    time.sleep(0.1)
                
                if line.strip():
                    line_count += 1
                    
                    # DEBUG: Log every single line (first 50 and then periodically)
                    if DEBUG_LOGGING:
                        if line_count <= 50:
                            log_debug(f"Line {line_count}: {line.strip()}")
                            first_lines.append(line.strip())
                        elif line_count % 100 == 0:
                            log_debug(f"Line {line_count}: {line.strip()[:200]}")
                    
                    # Check for download progress - these often use carriage returns
                    # Look for patterns like "Fetching X files" or percentage indicators
                    if 'Fetching' in line or ('%' in line and ('it/s' in line or 'MiB' in line or 'MB' in line)):
                        # Throttle download updates - only show every few seconds
                        current_time = time.time()
                        if current_time - last_download_update > 2:  # Update every 2 seconds
                            # Extract percentage if present
                            pct_match = re.search(r'(\d+)%', line)
                            if pct_match:
                                pct = pct_match.group(1)
                                download_msg = f"⬇️ Downloading model... {pct}% complete"
                                print(f"[READ_OUTPUT] {download_msg}")
                                self.output_queue.put(download_msg + "\n")
                            last_download_update = current_time
                    
                    if line_count <= 5 or line_count % 10 == 0:
                        print(f"[READ_OUTPUT] Line {line_count}: {line.strip()[:100]}")
                    
                    if 'Val loss' in line:
                        print(f"[READ_OUTPUT] PUTTING VAL LOSS IN QUEUE: {line.strip()[:80]}")
                    
                    self.output_queue.put(line)
            
            print(f"[READ_OUTPUT] Finished reading, total lines: {line_count}")
            
            # DEBUG: Log end of reading
            if DEBUG_LOGGING:
                log_info("=" * 80)
                log_info("OUTPUT READ LOOP FINISHED")
                log_info("=" * 80)
                log_info(f"Total lines read: {line_count}")
                log_info(f"First 20 lines:")
                for i, first_line in enumerate(first_lines[:20]):
                    log_info(f"  {i+1}: {first_line[:150]}")
                if line_count == 0:
                    log_warning("NO LINES WERE READ - Process may have exited immediately")
            
            # Wait for process to finish and get return code
            return_code = self.process.wait()
            
            # DEBUG: Log return code
            if DEBUG_LOGGING:
                log_info(f"Process wait() returned with code: {return_code}")
            
            print(f"[READ_OUTPUT] Process exited with return code: {return_code}")
            
            # Training complete or stopped
            self.training_complete = True
            
            if self.was_stopped:
                self.output_queue.put("\n[STATUS] Training stopped by user")
            elif return_code == 0:
                self.output_queue.put("\n[STATUS] Training completed successfully")
            else:
                error_msg = f"\n[ERROR] Training failed with code: {return_code}"
                print(f"[READ_OUTPUT] {error_msg}")
                self.output_queue.put(error_msg)
                
        except Exception as e:
            import traceback
            error_msg = f"\n[ERROR] Error reading output: {str(e)}\n{traceback.format_exc()}"
            print(f"[READ_OUTPUT] {error_msg}")
            self.output_queue.put(error_msg)
            
            # DEBUG: Log exception details
            if DEBUG_LOGGING:
                log_error(f"Exception in _read_output: {e}")
                log_error(traceback.format_exc())
    
    def _parse_progress(self, total_iters_expected):
        """Parse training output for progress metrics from mlx_lm.lora directly."""
        train_losses = {}
        val_losses = {}
        train_speeds = []  # Initialize training speeds list
        total_trained_tokens = 0  # Initialize token counter
        best_loss = float('inf')
        best_iter = 0
        current_iter = 0
        total_iters = total_iters_expected  # Use the value passed from start_training
        peak_memory_gb = 0.0
        
        print(f"[PARSE_PROGRESS] Starting with total_iters={total_iters}")
        
        while not self.training_complete and not self.stop_event.is_set():
            try:
                # Try to get line without blocking
                line = self.output_queue.get(timeout=0.1)
                
                # Log every line for debugging (first 100 chars)
                if current_iter % 50 == 0 or 'Iter' in line:
                    print(f"[PARSE] Line: {line[:100]}")
                
                # Debug: Track Val loss lines specifically
                if 'Val loss' in line:
                    print(f"[PARSE] GOT VAL LOSS FROM QUEUE: {line.strip()[:100]}")
                
                # Parse download progress from huggingface_hub (model downloading)
                # Format: "Fetching 11 files:   0%|          | 0/11 [00:00<?, ?it/s]"
                # Format: "Downloading pytorch_model.bin:  45%|████▌     | 1.23G/2.74G [00:15<00:18, 82.5MB/s]"
                download_match = re.search(r'(Fetching|Downloading)\s+(\d+)\s+files?:\s+(\d+)%', line)
                if download_match:
                    action = download_match.group(1)  # Fetching or Downloading
                    total_files = download_match.group(2)
                    percent = int(download_match.group(3))
                    
                    # Rate limit: Only log every 10% increment to avoid spam
                    if not hasattr(self, '_last_download_pct'):
                        self._last_download_pct = -1
                    
                    if percent != self._last_download_pct and percent % 10 == 0:
                        self._last_download_pct = percent
                        print(f"[DOWNLOAD] {action} model files: {percent}% complete ({total_files} files)")
                        # Put a clean status message in the queue for UI display
                        self.output_queue.put(f"⬇️ Downloading model: {percent}% complete...")
                
                # Also check for individual file download progress
                file_download_match = re.search(r'Downloading.*:\s+(\d+)%', line)
                if file_download_match and not download_match:
                    percent = int(file_download_match.group(1))
                    
                    # Rate limit: Only log every 10% increment and avoid duplicates
                    if not hasattr(self, '_last_file_download_pct'):
                        self._last_file_download_pct = -1
                        self._file_download_count = 0  # Track how many files we've seen
                    
                    # Only log if: different percent AND is 10% increment AND different from last logged
                    should_log = (
                        percent != self._last_file_download_pct and 
                        percent % 10 == 0 and
                        percent >= self._last_file_download_pct  # Only log increasing progress
                    )
                    
                    if should_log:
                        self._last_file_download_pct = percent
                        self._file_download_count += 1
                        print(f"[DOWNLOAD] File #{self._file_download_count}: {percent}%")
                        # Only show file download at 0% and 100% to reduce spam
                        if percent == 0 or percent == 100:
                            self.output_queue.put(f"⬇️ Downloading model file {self._file_download_count}: {percent}%...")
                
                # Parse mlx_lm.lora format directly
                # Format: "Iter 50: Train loss 2.345, Learning Rate 1.000e-04, It/sec 3.435, Tokens/sec 665.450, Trained Tokens 2193, Peak mem 4.197 GB"
                train_match = re.search(r'Iter\s+(\d+).*Train loss\s+([\d.]+|nan|inf)', line, re.IGNORECASE)
                if train_match:
                    current_iter = int(train_match.group(1))
                    loss_str = train_match.group(2)
                    if loss_str.lower() not in ['nan', 'inf']:
                        train_losses[current_iter] = float(loss_str)
                        
                        # Parse additional metrics if available
                        iter_speed_match = re.search(r'It/sec\s+([\d.]+)', line)
                        token_speed_match = re.search(r'Tokens/sec\s+([\d.]+)', line)
                        trained_tokens_match = re.search(r'Trained Tokens\s+(\d+)', line)
                        
                        if iter_speed_match:
                            iter_speed = float(iter_speed_match.group(1))
                            train_speeds.append((current_iter, iter_speed))
                        
                        if token_speed_match:
                            token_speed = float(token_speed_match.group(1))
                        
                        if trained_tokens_match:
                            total_trained_tokens = int(trained_tokens_match.group(1))
                        
                        print(f"[PARSE] Train iter {current_iter}: loss={loss_str}")
                
                # Format: "Iter 50: Val loss 1.987, Val took 5.436s" (mlx_lm format)
                # Also: "Iter 150: 2.0250" or "Iteration 300: 1.5860" (summary format)
                
                # Debug: Check what's in the line before regex
                if 'Val loss' in line:
                    print(f"[PARSE-DEBUG-VAL] Line contains 'Val loss': {repr(line[:100])}")
                
                val_match = re.search(r'Iter\s+(\d+).*Val loss\s+([\d.]+)', line)
                if not val_match:
                    # Try alternative format: "Iter 150: 2.0250" or "Iteration 300: 1.5860"
                    val_match = re.search(r'Iter(?:ation)?\s+(\d+):\s+([\d.]+)', line)
                
                if val_match:
                    current_iter = int(val_match.group(1))
                    val_loss = float(val_match.group(2))
                    val_losses[current_iter] = val_loss
                    
                    # Check if this is best
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_iter = current_iter
                        print(f"[PARSE] ✓ New best at iter {current_iter}: val_loss={val_loss:.4f}")
                    else:
                        print(f"[PARSE] Val iter {current_iter}: loss={val_loss:.4f} (best={best_loss:.4f})")
                    
                    # Debug: log validation loss being tracked
                    print(f"[PARSE DEBUG] Added val_loss at iter {current_iter}: {val_loss:.4f}, total val_losses: {len(val_losses)}")
                
                # Parse memory usage: "Peak mem X.XXX GB"
                mem_match = re.search(r'Peak mem\s+([\d.]+)\s*GB', line)
                if mem_match:
                    peak_memory_gb = float(mem_match.group(1))
                
                # Alternative format from lora-train logs: "Memory captured: X.XXX GB"
                debug_mem_match = re.search(r'Memory captured:\s+([\d.]+)\s*GB', line)
                if debug_mem_match:
                    peak_memory_gb = float(debug_mem_match.group(1))
                
                # Get resource stats every 10 iterations or on validation
                resource_stats = None
                if self.resource_monitor and (current_iter % 10 == 0 or val_match):
                    try:
                        self.resource_monitor.get_stats()
                        resource_stats = {
                            'cpu': self.resource_monitor.stats.cpu_percent,
                            'ram_used': self.resource_monitor.stats.ram_used_gb,
                            'ram_total': self.resource_monitor.stats.ram_total_gb,
                            'ram_percent': self.resource_monitor.stats.ram_percent,
                            'mlx_memory': self.resource_monitor.stats.mlx_gpu_used_gb
                        }
                    except Exception as e:
                        pass  # Silently skip if resource monitoring fails
                
                # Send progress update - but rate limit to avoid spam
                # Only send every 5th line or when we have training data
                should_send_update = (
                    train_match is not None or  # We parsed training data this iteration
                    val_match is not None or     # We parsed validation data
                    resource_stats is not None or  # We have resource stats to report
                    (hasattr(self, '_update_counter') and self._update_counter % 5 == 0)  # Every 5th update
                )
                
                if not hasattr(self, '_update_counter'):
                    self._update_counter = 0
                self._update_counter += 1
                
                if should_send_update:
                    progress_percent = int((current_iter / total_iters) * 100) if total_iters > 0 else 0
                    progress_data = {
                        'iteration': current_iter,
                        'total': total_iters,
                        'progress_percent': progress_percent,
                        'train_losses': train_losses.copy(),
                        'val_losses': val_losses.copy(),
                        'best_loss': best_loss,
                        'best_iter': best_iter,
                        'peak_memory_gb': peak_memory_gb,
                        'raw_line': line if train_match or val_match else '',
                        'resource_stats': resource_stats,
                        # Training speed metrics for live analytics
                        'train_speeds': train_speeds.copy() if train_speeds else [],
                        'total_trained_tokens': total_trained_tokens if 'total_trained_tokens' in locals() else 0,
                        'current_iter_speed': train_speeds[-1][1] if train_speeds else 0.0,
                    }
                    self.progress_queue.put(progress_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[MONITOR] Error parsing progress: {e}")
                import traceback
                print(f"[MONITOR] {traceback.format_exc()}")
                continue
        
        # Send final progress
        final_progress = {
            'iteration': current_iter,
            'total': total_iters,
            'progress_percent': 100 if self.training_complete else int((current_iter / total_iters) * 100),
            'train_losses': train_losses.copy(),
            'val_losses': val_losses.copy(),
            'best_loss': best_loss,
            'best_iter': best_iter,
            'peak_memory_gb': peak_memory_gb,
            'complete': True
        }
        self.progress_queue.put(final_progress)
        print(f"[PARSE_PROGRESS] Finished. Final iter={current_iter}/{total_iters}, best={best_loss:.4f} at iter {best_iter}")
    
    def _start_resource_watchdog(self, max_time_seconds: int, max_memory_gb: int):
        """Start a watchdog thread to monitor resource usage and enforce limits.
        
        Args:
            max_time_seconds: Maximum training time before forced termination
            max_memory_gb: Maximum memory usage in GB before warning
        """
        def watchdog():
            """Monitor resources and terminate if limits exceeded."""
            start_time = time.time()
            warned_memory = False
            
            while not self.stop_event.is_set() and not self.training_complete:
                elapsed = time.time() - start_time
                
                # Check time limit
                if elapsed > max_time_seconds:
                    error_msg = f"[SECURITY] Training terminated: exceeded time limit of {max_time_seconds//3600} hours"
                    print(error_msg)
                    self.output_queue.put(f"\n{error_msg}")
                    self.stop()
                    break
                
                # Check memory usage if resource monitoring available
                if RESOURCE_MONITORING_AVAILABLE and self.resource_monitor:
                    try:
                        stats = self.resource_monitor.get_stats()
                        if stats and hasattr(stats, 'mlx_gpu_used_gb'):
                            current_mem = stats.mlx_gpu_used_gb
                            if current_mem > max_memory_gb:
                                if not warned_memory:
                                    warn_msg = f"[SECURITY WARNING] Memory usage ({current_mem:.2f}GB) exceeds limit ({max_memory_gb}GB)"
                                    print(warn_msg)
                                    self.output_queue.put(f"\n{warn_msg}")
                                    warned_memory = True
                                # Note: We don't terminate on memory warning, just alert
                                # MLX manages memory, and temporary spikes are normal
                    except Exception as e:
                        # Silently ignore monitoring errors
                        pass
                
                # Sleep for 30 seconds between checks
                time.sleep(30)
        
        # Start watchdog in daemon thread
        watchdog_thread = threading.Thread(target=watchdog, daemon=True, name="ResourceWatchdog")
        watchdog_thread.start()
        print(f"[SECURITY] Resource watchdog started (checking every 30s)")
    
    def stop(self):
        """Stop training early."""
        self.was_stopped = True
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except:
                self.process.kill()
    
    def is_stopped(self):
        """Check if training was stopped by user."""
        return self.was_stopped
    
    def pause(self):
        """Pause training (not implemented - would require signal handling)."""
        # Note: Actual pause would require modifying MLX-LM to handle signals
        # For now, this is a placeholder
        pass
    
    def is_running(self):
        """Check if training is still running."""
        return self.process is not None and self.process.poll() is None
    
    def is_complete(self):
        """Check if training completed."""
        return self.training_complete
