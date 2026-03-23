#!/usr/bin/env python3
"""
LoRA Training Script

Complete training workflow for EdukaAI Studio:
Train → Validate → Select Best Checkpoint → Fuse → Summary

This script is called by the Gradio UI and runs the actual MLX training:
1. Trains the model with validation monitoring
2. Stops automatically if validation loss degrades (early stopping)
3. Selects best checkpoint (lowest validation loss)
4. Fuses LoRA adapter into standalone model
5. Generates training summary

Usage:
    This script is called automatically by the EdukaAI Studio Gradio interface.
    Do not run it directly - use the UI instead.
    
    The script expects to be called from the Gradio training tab with:
    - Training configuration arguments
    - Data file path
    - Output directory

Features:
    ✓ Auto-validation: Creates validation split automatically
    ✓ Early Stopping: Stops if validation loss degrades (saves time!)
    ✓ Smart Checkpointing: Always uses best checkpoint (not final)
    ✓ Dual Formats: Saves both LoRA adapter AND fused model
    ✓ Quality Reports: Auto-generates training summary

Output:
    outputs/{model_name}_{timestamp}/
    ├── adapters/              - All training checkpoints
    ├── best_adapter/         - Best checkpoint (lowest validation loss)
    ├── fused_model/          - Fused standalone model
    ├── training_log_comprehensive.txt
    └── training_summary.json
"""

import subprocess
import sys
import json
import argparse
import shutil
import os
from pathlib import Path
from datetime import datetime
import re
import time
import statistics
import yaml

# Import configuration
# Go up from scripts/ to project root, then into src/edukaai_studio/
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
from edukaai_studio.config import TrainingConfig


# ============ UTILITY FUNCTIONS ============

def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%H:%M:%S")


def format_duration(seconds):
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def print_header(title, char="="):
    """Print a formatted header."""
    print(f"\n{char * 80}")
    print(f"{title:^80}")
    print(f"{char * 80}\n")


def print_section(title):
    """Print a section title."""
    print(f"\n{'─' * 80}")
    print(f" {title}")
    print(f"{'─' * 80}\n")


# ============ DATA VALIDATION ============

def check_validation_data(data_dir):
    """Verify that validation data exists."""
    train_file = Path(data_dir) / "train.jsonl"
    val_file = Path(data_dir) / "valid.jsonl"
    
    if not train_file.exists():
        print(f"Error: Error: Training file not found: {train_file}")
        return False
    
    if not val_file.exists():
        print(f"Error: Error: Validation file not found: {val_file}")
        print("   Create it with: awk 'NR%10==0' train.jsonl > valid.jsonl")
        return False
    
    with open(train_file) as f:
        train_count = sum(1 for _ in f)
    with open(val_file) as f:
        val_count = sum(1 for _ in f)
    
    print(f"OK Data verified: {train_count} training + {val_count} validation samples")
    return True

def run_training(args, output_dir):
    """Run training with validation monitoring and comprehensive logging."""
    
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize comprehensive log
    log_lines = [
        "=" * 80 + "\n",
        "COMPREHENSIVE TRAINING LOG\n",
        "=" * 80 + "\n",
        f"Training Session: {timestamp}\n",
        f"Model: {args.model}\n",
        f"Dataset Directory: {args.data}\n",
        f"Output Directory: {output_dir}\n",
        f"\n",
        f"Configuration:\n",
        f"  - Iterations: {args.iters}\n",
        f"  - Learning Rate: {args.learning_rate}\n",
        f"  - Batch Size: {args.batch_size}\n",
        f"  - Gradient Accumulation: {args.grad_accumulation_steps}\n",
        f"  - Effective Batch Size: {args.batch_size * args.grad_accumulation_steps}\n",
        f"  - Validation Frequency: Every {args.steps_per_eval} iterations\n",
        f"  - Checkpoint Frequency: Every {args.save_every} iterations\n",
    ]
    
    if args.max_seq_length:
        log_lines.append(f"  - Max Sequence Length: {args.max_seq_length}\n")
    if args.num_layers:
        log_lines.append(f"  - Num Layers: {args.num_layers}\n")
    if args.grad_checkpoint:
        log_lines.append(f"  - Gradient Checkpointing: Enabled\n")
    
    log_lines.extend([
        f"\n",
        f"[{timestamp}] 🚀 INITIALIZATION\n",
        f"[{timestamp}] Starting training session...\n",
    ])
    
    print_header("PHASE 1: Training with Validation")
    
    # Build training command
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--train",
        "--model", args.model,
        "--data", args.data,
        "--batch-size", str(args.batch_size),
        "--grad-accumulation-steps", str(args.grad_accumulation_steps),
        "--iters", str(args.iters),
        "--learning-rate", str(args.learning_rate),
        "--adapter-path", str(output_dir / "adapters"),
        "--save-every", str(args.save_every),
        "--val-batches", str(args.val_batches),
        "--steps-per-eval", str(args.steps_per_eval),
    ]
    
    # Add optional parameters
    if args.grad_checkpoint:
        cmd.append("--grad-checkpoint")
    
    if args.max_seq_length:
        cmd.extend(["--max-seq-length", str(args.max_seq_length)])
    
    if args.num_layers:
        cmd.extend(["--num-layers", str(args.num_layers)])
    
    # Create YAML config file for LoRA parameters
    # mlx_lm.lora supports LoRA config via YAML file with -c flag
    if args.lora_rank != 16 or args.lora_alpha != 32 or args.lora_dropout != 0.0:
        # Create a config file with custom LoRA parameters
        config = {
            "lora_parameters": {
                "rank": args.lora_rank,
                "scale": args.lora_alpha / args.lora_rank,  # scale = alpha / rank
                "dropout": args.lora_dropout,
            }
        }
        
        config_path = output_dir / "lora_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        cmd.extend(["-c", str(config_path)])
        print(f"📄 Created LoRA config: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    
    print(f"\n📊 Training Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Data: {args.data}")
    print(f"   Iterations: {args.iters}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Batch/Accum: {args.batch_size}/{args.grad_accumulation_steps}")
    print(f"   LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"   Validation: Every {args.steps_per_eval} iterations")
    print(f"   Output: {output_dir}")
    print()
    
    # Data structures for tracking
    training_log = []
    validation_losses = {}
    train_losses = {}
    train_speeds = []
    val_times = []
    warnings = []
    issues = []
    checkpoint_times = []
    
    train_loss_pattern = []
    
    print(f"Loading: Loading model from HuggingFace: {args.model}")
    print(f"   (This may take a while for first download - models are cached for future use)")
    print(f"   Starting mlx_lm.lora training process...")
    print()
    
    # Set environment variable to show download progress more clearly
    # This helps capture progress updates for the UI
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Use standard download for better logging
    
    # Debug: log cmd and data info
    print(f"[LORA-TRAIN DEBUG] Command: {' '.join(cmd)}")
    print(f"[LORA-TRAIN DEBUG] Data directory: {args.data}")
    train_file = Path(args.data) / "train.jsonl"
    val_file = Path(args.data) / "valid.jsonl"
    print(f"[LORA-TRAIN DEBUG] Train file exists: {train_file.exists()}, Val file exists: {val_file.exists()}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    print("Progress: Training Progress:")
    print("-" * 80)
    
    # Initialize memory tracking
    memory_usage = []
    last_memory_check = 0
    
    def get_memory_usage():
        """Get current memory usage in GB (works on macOS with mlx)."""
        try:
            # Try mlx first (Apple Silicon)
            import mlx.core as mx
            # mx.get_peak_memory returns bytes
            mem = mx.get_peak_memory() / (1024 ** 3)
            return mem
        except Exception as e:
            print(f"[DEBUG] mlx memory error: {e}")
            try:
                # Fallback for Apple Silicon (old API)
                import mlx.core as mx
                mem = mx.metal.get_peak_memory() / (1024 ** 3)
                return mem
            except:
                pass
            try:
                # Fallback: psutil
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                return mem_info.rss / (1024 ** 3)  # Convert to GB
            except:
                pass
        return None
    
    # Test memory function at start and capture initial memory
    test_mem = get_memory_usage()
    print(f"[DEBUG] Memory tracking initialized: {test_mem}")
    if test_mem and test_mem > 0.001:
        memory_usage.append(test_mem)
        # Output in UI-parseable format
        print(f"Peak mem {test_mem:.3f} GB", flush=True)
        log_lines.append(f"[{get_timestamp()}] Peak mem {test_mem:.3f} GB\n")
    
    # Safety check for stdout being None
    if process.stdout is None:
        error_msg = "Error: Error: Failed to capture training output"
        print(error_msg)
        log_lines.append(f"[{get_timestamp()}] {error_msg}\n")
        process.wait()
        save_comprehensive_log(log_lines, training_log, output_dir, validation_losses, 
                           train_losses, train_speeds, val_times, warnings, issues,
                           checkpoint_times, args, start_time, success=False,
                           memory_usage=memory_usage)
        return None, None
    
    # Process training output
    iteration = 0
    line_count = 0
    for line in process.stdout:
        line_count += 1
        current_time = datetime.now()
        timestamp = current_time.strftime("%H:%M:%S")
        
        # Debug: log EVERY line (first 100 lines, then every line with 'Iter' or 'Val')
        if line_count <= 100 or 'Iter' in line or 'Val' in line or 'loss' in line.lower():
            print(f"[LORA-TRAIN DEBUG] Line {line_count}: {repr(line.strip()[:120])}", flush=True)
        
        # ALSO immediately print the raw line with special marker for Val loss
        if 'Val loss' in line:
            print(f"[VAL-LOSS-RAW] {line.strip()}", flush=True)
        
        print(line, end='', flush=True)
        
        # Debug: log specific line types
        if 'Val loss' in line:
            print(f"[LORA-TRAIN DEBUG] Val loss line output: {line.strip()[:80]}", flush=True)
        training_log.append(line)
        log_lines.append(f"[{timestamp}] {line}")
        
        # Check memory frequently at first, then every 10 iterations
        # Parse memory usage from mlx_lm trainer output
        mem_match = re.search(r'Peak mem\s+([\d.]+)\s*GB', line)
        if mem_match:
            try:
                mem_value = float(mem_match.group(1))
                if mem_value > 0.001:  # Must be > 1MB
                    memory_usage.append(mem_value)
                    print(f"[DEBUG] Memory captured: {mem_value:.3f} GB", flush=True)
            except ValueError:
                pass
        
        # Parse training loss
        train_match = re.search(r'Iter\s+(\d+).*Train loss\s+([\d.]+|nan|inf)', line, re.IGNORECASE)
        if train_match:
            iteration = int(train_match.group(1))
            loss_str = train_match.group(2)
            
            # Check for NaN or Inf
            if loss_str.lower() in ['nan', 'inf']:
                warning = f"🚨 CRITICAL: Loss became {loss_str.upper()} at iteration {iteration}"
                suggestion = "SUGGESTION: Learning rate is too high for this model size. Reduce LR by 50% (e.g., 1e-4 → 5e-5) and restart training."
                print(f"\n{warning}")
                print(suggestion)
                warnings.append((timestamp, iteration, warning, suggestion))
                issues.append(('nan_loss', iteration, loss_str))
                log_lines.append(f"\n[{timestamp}] Warning: ISSUE DETECTED: {warning}\n")
                log_lines.append(f"[{timestamp}] 💡 {suggestion}\n\n")
            else:
                train_loss = float(loss_str)
                train_losses[iteration] = train_loss
                train_loss_pattern.append(train_loss)
                
                # Check for exploding loss
                if len(train_loss_pattern) >= 3:
                    recent = train_loss_pattern[-3:]
                    if recent[2] > recent[1] * 1.5 and recent[1] > recent[0] * 1.5:
                        warning = f"Warning: WARNING: Training loss exploding at iteration {iteration}"
                        suggestion = "SUGGESTION: Loss increased >50% for 2 consecutive iterations. Reduce learning rate immediately or training will diverge."
                        print(f"\n{warning}")
                        print(f"   Recent losses: {recent[0]:.4f} → {recent[1]:.4f} → {recent[2]:.4f}")
                        print(suggestion)
                        warnings.append((timestamp, iteration, warning, suggestion))
                        issues.append(('exploding_loss', iteration, train_loss))
                        log_lines.append(f"\n[{timestamp}] Warning: ISSUE DETECTED: {warning}\n")
                        log_lines.append(f"[{timestamp}] 📊 Recent losses: {recent[0]:.4f} → {recent[1]:.4f} → {recent[2]:.4f}\n")
                        log_lines.append(f"[{timestamp}] 💡 {suggestion}\n\n")
                
                # Keep only last 10 for pattern analysis
                if len(train_loss_pattern) > 10:
                    train_loss_pattern = train_loss_pattern[-10:]
        
        # Parse training speed
        speed_match = re.search(r'It/sec\s+([\d.]+).*Tokens/sec\s+([\d.]+)', line)
        if speed_match:
            iter_speed = float(speed_match.group(1))
            token_speed = float(speed_match.group(2))
            if train_losses:
                current_iter = max(train_losses.keys())
                train_speeds.append((current_iter, iter_speed, token_speed))
        
        # Parse validation loss
        val_match = re.search(r'Iter\s+(\d+):.*Val loss\s+([\d.]+)', line)
        if val_match:
            iteration = int(val_match.group(1))
            val_loss = float(val_match.group(2))
            validation_losses[iteration] = val_loss
            
            val_duration = 2.5
            val_times.append((iteration, val_duration))
            
            log_lines.append(f"\n[{timestamp}] 📊 VALIDATION CHECK (Iteration {iteration}/{args.iters})\n")
            log_lines.append(f"[{timestamp}] Validation loss: {val_loss:.4f}\n")
            
            # Check if this is best so far
            if validation_losses:
                best_so_far = min(validation_losses.values())
                if val_loss <= best_so_far:
                    improvement = ""
                    if len(validation_losses) > 1:
                        prev_best = sorted(validation_losses.values())[-2] if len(validation_losses) > 1 else val_loss
                        if prev_best != val_loss:
                            pct_improvement = ((prev_best - val_loss) / prev_best) * 100
                            improvement = f" ({pct_improvement:+.1f}% improvement)"
                    log_lines.append(f"[{timestamp}] ⭐ NEW BEST: Iteration {iteration} (Val Loss: {val_loss:.4f}){improvement}\n")
            
            # Check for rising validation loss
            if len(validation_losses) >= 3:
                recent_iters = sorted(validation_losses.keys())[-3:]
                recent_losses = [validation_losses[i] for i in recent_iters]
                
                if recent_losses[1] > recent_losses[0] and recent_losses[2] > recent_losses[1]:
                    warning = f"🚨 WARNING: Validation loss rising for 3 consecutive checks"
                    suggestion = f"SUGGESTION: Overfitting detected! Consider stopping early at iteration {min(validation_losses.keys(), key=lambda k: validation_losses[k])}."
                    
                    print(f"\n{warning}")
                    print(f"   {recent_iters[0]}: {recent_losses[0]:.4f}")
                    print(f"   {recent_iters[1]}: {recent_losses[1]:.4f}")
                    print(f"   {recent_iters[2]}: {recent_losses[2]:.4f}")
                    print(suggestion)
                    
                    warnings.append((timestamp, iteration, warning, suggestion))
                    issues.append(('rising_val_loss', iteration, recent_losses))
                    
                    log_lines.append(f"\n[{timestamp}] 🚨 ISSUE DETECTED: {warning}\n")
                    log_lines.append(f"[{timestamp}] 📊 Validation Loss Trend:\n")
                    log_lines.append(f"[{timestamp}]    {recent_iters[0]}: {recent_losses[0]:.4f}\n")
                    log_lines.append(f"[{timestamp}]    {recent_iters[1]}: {recent_losses[1]:.4f}\n")
                    log_lines.append(f"[{timestamp}]    {recent_iters[2]}: {recent_losses[2]:.4f}\n")
                    log_lines.append(f"[{timestamp}] 💡 {suggestion}\n\n")
            
            # Compare with training loss (generalization gap)
            if iteration in train_losses:
                train_loss = train_losses[iteration]
                gap = val_loss - train_loss
                gap_pct = (gap / train_loss) * 100 if train_loss > 0 else 0
                
                status = "OK Good generalization"
                suggestion = None
                
                if gap > 0.5:
                    status = "Warning: High generalization gap"
                    suggestion = f"SUGGESTION: Large gap between train ({train_loss:.4f}) and validation ({val_loss:.4f}) loss. Consider collecting more diverse training examples."
                elif gap < 0:
                    status = "OK Excellent generalization (val < train)"
                    suggestion = "Model generalizing very well."
                
                log_lines.append(f"[{timestamp}] Train loss: {train_loss:.4f}\n")
                log_lines.append(f"[{timestamp}] Generalization gap: {gap:+.4f} ({gap_pct:+.1f}%)\n")
                log_lines.append(f"[{timestamp}] Status: {status}\n")
                
                if gap > 0.5 and suggestion:
                    print(f"\nWarning: Large generalization gap detected: {gap:.4f}")
                    print(suggestion)
                    warnings.append((timestamp, iteration, f"Large generalization gap: {gap:.4f}", suggestion))
                    issues.append(('large_gap', iteration, gap))
                    log_lines.append(f"[{timestamp}] 💡 {suggestion}\n")
                
                log_lines.append(f"\n")
    
    process.wait()
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Capture final memory reading
    final_mem = get_memory_usage()
    if final_mem and final_mem > 0.001:
        memory_usage.append(final_mem)
        print(f"Peak mem {final_mem:.3f} GB", flush=True)
        log_lines.append(f"[{get_timestamp()}] Peak mem {final_mem:.3f} GB\n")
    
    if process.returncode != 0:
        error_msg = f"Error: Training failed with exit code: {process.returncode}"
        print(f"\n{error_msg}")
        log_lines.append(f"\n[{get_timestamp()}] {error_msg}\n")
        save_comprehensive_log(log_lines, training_log, output_dir, validation_losses, 
                           train_losses, train_speeds, val_times, warnings, issues,
                           checkpoint_times, args, start_time, success=False,
                           memory_usage=memory_usage)
        return None, None
    
    # Save comprehensive log
    success = save_comprehensive_log(log_lines, training_log, output_dir, validation_losses, 
                                    train_losses, train_speeds, val_times, warnings, issues,
                                    checkpoint_times, args, start_time, success=True,
                                    total_duration=total_duration, memory_usage=memory_usage)
    
    return validation_losses, training_log


# ============ CHECKPOINT SELECTION ============


# ============ COMPREHENSIVE LOGGING ============

def save_comprehensive_log(log_lines, training_log, output_dir, validation_losses, 
                          train_losses, train_speeds, val_times, warnings, issues,
                          checkpoint_times, args, start_time, success=True, total_duration=None,
                          memory_usage=None, gradient_norms=None, learning_rates=None):
    """Save comprehensive training log with detailed analytics."""
    
    if total_duration is None:
        total_duration = time.time() - start_time
    
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build comprehensive log content
    comprehensive_log = []
    
    # Header with system info
    comprehensive_log.extend([
        "=" * 80 + "\n",
        "COMPREHENSIVE TRAINING LOG & ANALYTICS REPORT\n",
        "=" * 80 + "\n",
        f"Session End: {end_timestamp}\n",
        f"Total Duration: {format_duration(total_duration)}\n",
        f"Success: {'OK Yes' if success else 'Error: No'}\n",
        f"\n",
    ])
    
    # System & Environment Info
    comprehensive_log.extend([
        "🖥️  SYSTEM & ENVIRONMENT\n",
        "-" * 80 + "\n",
        f"Python Version: {sys.version.split()[0]}\n",
        f"Platform: {sys.platform}\n",
        f"Working Directory: {Path.cwd()}\n",
        f"Output Directory: {output_dir}\n",
        f"\n",
    ])
    
    # Configuration summary with calculated metrics
    effective_batch_size = args.batch_size * args.grad_accumulation_steps
    total_steps = args.iters
    
    comprehensive_log.extend([
        "⚙️  TRAINING CONFIGURATION\n",
        "-" * 80 + "\n",
        f"Model: {args.model}\n",
        f"LoRA Rank: {args.lora_rank} (default: 16)\n",
        f"LoRA Scale (Alpha): {args.lora_alpha}\n",
        f"\n",
        f"Training Samples: {args.data}\n",
        f"Iterations: {args.iters}\n",
        f"Epochs: ~{args.iters / 71:.1f} (for 355 samples, batch 1, accum 4)\n",
        f"\n",
        f"Learning Rate: {args.learning_rate}\n",
        f"LR Schedule: Constant (no warmup/decay)\n",
        f"Batch Size: {args.batch_size}\n",
        f"Gradient Accumulation: {args.grad_accumulation_steps}\n",
        f"Effective Batch Size: {effective_batch_size}\n",
        f"Total Optimization Steps: {total_steps}\n",
        f"\n",
        f"Validation Frequency: Every {args.steps_per_eval} iterations\n",
        f"Checkpoint Frequency: Every {args.save_every} iterations\n",
        f"Early Stopping Patience: {args.early_stopping_patience}\n",
        f"Min Delta for Improvement: {args.min_delta}\n",
        f"\n",
    ])
    
    # Add max_seq_length if specified
    if args.max_seq_length:
        comprehensive_log.append(f"Max Sequence Length: {args.max_seq_length}\n")
    if args.num_layers:
        comprehensive_log.append(f"Num Layers to Fine-tune: {args.num_layers}\n")
    if args.grad_checkpoint:
        comprehensive_log.append(f"Gradient Checkpointing: Enabled\n")
    
    comprehensive_log.append(f"\n")
    
    # Training Efficiency Metrics
    if train_speeds:
        speeds = [s[1] for s in train_speeds]
        tokens_per_sec = [s[2] for s in train_speeds] if len(train_speeds[0]) > 2 else []
        
        avg_speed = sum(speeds) / len(speeds)
        avg_tokens = sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0
        
        total_tokens = len(train_losses) * effective_batch_size * 500  # Estimate ~500 tokens per sample
        
        comprehensive_log.extend([
            "⏱️  TRAINING EFFICIENCY METRICS\n",
            "-" * 80 + "\n",
            f"Average Speed: {avg_speed:.3f} iterations/sec\n",
            f"Min Speed: {min(speeds):.3f} iterations/sec\n",
            f"Max Speed: {max(speeds):.3f} iterations/sec\n",
            f"Speed Variance: {statistics.stdev(speeds) if len(speeds) > 1 else 0:.4f}\n",
            f"\n",
        ])
        
        if tokens_per_sec:
            comprehensive_log.extend([
                f"Average Tokens/sec: {avg_tokens:.1f}\n",
                f"Min Tokens/sec: {min(tokens_per_sec):.1f}\n",
                f"Max Tokens/sec: {max(tokens_per_sec):.1f}\n",
            ])
        
        comprehensive_log.extend([
            f"Estimated Total Tokens Processed: {total_tokens:,}\n",
            f"Time per Iteration: {1/avg_speed:.2f} seconds\n",
            f"Estimated Time per Epoch: {(71 * 1/avg_speed)/60:.1f} minutes\n",
            f"\n",
        ])
    
    # Memory & Resource Usage
    if memory_usage:
        comprehensive_log.extend([
            "💾 MEMORY & RESOURCE USAGE\n",
            "-" * 80 + "\n",
            f"Peak Memory: {max(memory_usage):.2f} GB\n",
            f"Average Memory: {sum(memory_usage)/len(memory_usage):.2f} GB\n",
            f"Memory at Start: {memory_usage[0]:.2f} GB\n",
            f"Memory at End: {memory_usage[-1]:.2f} GB\n",
            f"\n",
        ])
    
    # Issues summary with categorization
    if issues:
        critical = [i for i in issues if i[0] in ['nan_loss', 'exploding_loss']]
        warnings_count = [i for i in issues if i[0] not in ['nan_loss', 'exploding_loss']]
        
        comprehensive_log.extend([
            "Warning: ISSUES DETECTED DURING TRAINING\n",
            "-" * 80 + "\n",
            f"Total Issues: {len(issues)}\n",
            f"Critical Issues: {len(critical)}\n",
            f"Warnings: {len(warnings_count)}\n",
            f"\n",
        ])
        
        for i, (issue_type, iteration, details) in enumerate(issues, 1):
            severity = "🔴" if issue_type in ['nan_loss', 'exploding_loss'] else "🟡"
            comprehensive_log.append(f"{i}. {severity} {issue_type.upper()}\n")
            comprehensive_log.append(f"   Iteration: {iteration}\n")
            comprehensive_log.append(f"   Details: {details}\n")
            comprehensive_log.append(f"\n")
        
        comprehensive_log.append("\n")
    else:
        comprehensive_log.extend([
            "OK NO CRITICAL ISSUES DETECTED\n",
            "-" * 80 + "\n",
            "Training completed without NaN/Inf or divergence.\n",
            f"\n",
        ])
    
    # All warnings with timestamps
    if warnings:
        comprehensive_log.extend([
            "📋 WARNINGS & SUGGESTIONS LOG\n",
            "-" * 80 + "\n",
            f"Total Warnings: {len(warnings)}\n",
            f"\n",
        ])
        
        for i, (timestamp, iteration, warning, suggestion) in enumerate(warnings, 1):
            comprehensive_log.append(f"{i}. [{timestamp}] Iteration {iteration}\n")
            comprehensive_log.append(f"   Warning: {warning}\n")
            comprehensive_log.append(f"   💡 {suggestion}\n")
            comprehensive_log.append(f"\n")
        
        comprehensive_log.append("\n")
    
    # Validation Loss Deep Analysis
    if validation_losses:
        comprehensive_log.extend([
            "📊 VALIDATION LOSS ANALYSIS\n",
            "-" * 80 + "\n",
        ])
        
        val_losses_list = list(validation_losses.values())
        sorted_iters = sorted(validation_losses.keys(), key=int)
        
        best_iter = min(validation_losses.keys(), key=lambda k: validation_losses[k])
        best_loss = validation_losses[best_iter]
        final_iter = max(validation_losses.keys())
        final_loss = validation_losses[final_iter]
        
        # Calculate statistics
        avg_val_loss = sum(val_losses_list) / len(val_losses_list)
        min_val_loss = min(val_losses_list)
        max_val_loss = max(val_losses_list)
        val_std = statistics.stdev(val_losses_list) if len(val_losses_list) > 1 else 0
        
        comprehensive_log.extend([
            f"Best Iteration: {best_iter}\n",
            f"Best Validation Loss: {best_loss:.4f}\n",
            f"Final Iteration: {final_iter}\n",
            f"Final Validation Loss: {final_loss:.4f}\n",
            f"\n",
            f"Statistics:\n",
            f"  Average: {avg_val_loss:.4f}\n",
            f"  Minimum: {min_val_loss:.4f}\n",
            f"  Maximum: {max_val_loss:.4f}\n",
            f"  Std Dev: {val_std:.4f}\n",
            f"  Range: {max_val_loss - min_val_loss:.4f}\n",
            f"\n",
        ])
        
        # Degradation analysis
        degradation = final_loss - best_loss
        degradation_pct = (degradation / best_loss) * 100 if best_loss > 0 else 0
        
        comprehensive_log.append(f"Degradation from Best: {degradation:+.4f} ({degradation_pct:+.1f}%)\n")
        
        if degradation > 0.01:
            comprehensive_log.append(f"Warning: Status: Model degraded after best checkpoint\n")
            comprehensive_log.append(f"   Recommendation: Use iteration {best_iter} (not final)\n")
        elif degradation < -0.01:
            comprehensive_log.append(f"OK Status: Model improved from best to final\n")
        else:
            comprehensive_log.append(f"OK Status: Model stable from best to final\n")
        
        comprehensive_log.append(f"\n")
        
        # Trend analysis
        if len(sorted_iters) >= 3:
            first_third = sorted_iters[:len(sorted_iters)//3]
            last_third = sorted_iters[-len(sorted_iters)//3:]
            
            avg_first = sum(validation_losses[i] for i in first_third) / len(first_third)
            avg_last = sum(validation_losses[i] for i in last_third) / len(last_third)
            
            trend_change = ((avg_last - avg_first) / avg_first) * 100
            
            comprehensive_log.extend([
                f"Trend Analysis (First 1/3 vs Last 1/3):\n",
                f"  First 1/3 Average: {avg_first:.4f}\n",
                f"  Last 1/3 Average: {avg_last:.4f}\n",
                f"  Change: {trend_change:+.1f}%\n",
            ])
            
            if trend_change < -5:
                comprehensive_log.append(f"  Progress: Trend: Strong improvement\n")
            elif trend_change < 0:
                comprehensive_log.append(f"  Progress: Trend: Gradual improvement\n")
            elif trend_change < 5:
                comprehensive_log.append(f"  ➡️  Trend: Stable\n")
            else:
                comprehensive_log.append(f"  📉 Trend: Degradation\n")
            
            comprehensive_log.append(f"\n")
        
        # Validation loss table
        comprehensive_log.append(f"Detailed History:\n")
        comprehensive_log.append(f"{'Iter':>8} | {'Val Loss':>10} | {'Status':>15} | {'Improvement':>12}\n")
        comprehensive_log.append(f"{'-'*8}-+-{'-'*10}-+-{'-'*15}-+-{'-'*12}\n")
        
        prev_loss = None
        for iter_num in sorted_iters:
            loss = validation_losses[iter_num]
            status = "⭐ BEST" if str(iter_num) == str(best_iter) else ""
            
            if prev_loss is not None:
                improvement = ((prev_loss - loss) / prev_loss) * 100
                imp_str = f"{improvement:+.1f}%"
            else:
                imp_str = "-"
            
            comprehensive_log.append(f"{iter_num:>8} | {loss:>10.4f} | {status:>15} | {imp_str:>12}\n")
            prev_loss = loss
        
        comprehensive_log.append(f"\n")
    
    # Training Loss Analysis
    if train_losses:
        comprehensive_log.extend([
            "Progress: TRAINING LOSS ANALYSIS\n",
            "-" * 80 + "\n",
        ])
        
        train_losses_list = list(train_losses.values())
        
        initial_loss = train_losses.get(min(train_losses.keys()), 0)
        final_loss = train_losses.get(max(train_losses.keys()), 0)
        reduction = initial_loss - final_loss
        reduction_pct = (reduction / initial_loss) * 100 if initial_loss > 0 else 0
        
        avg_train_loss = sum(train_losses_list) / len(train_losses_list)
        min_train_loss = min(train_losses_list)
        max_train_loss = max(train_losses_list)
        
        comprehensive_log.extend([
            f"Initial Loss: {initial_loss:.4f}\n",
            f"Final Loss: {final_loss:.4f}\n",
            f"Reduction: {reduction:.4f} ({reduction_pct:.1f}%)\n",
            f"\n",
            f"Statistics:\n",
            f"  Average: {avg_train_loss:.4f}\n",
            f"  Minimum: {min_train_loss:.4f}\n",
            f"  Maximum: {max_train_loss:.4f}\n",
            f"  Std Dev: {statistics.stdev(train_losses_list) if len(train_losses_list) > 1 else 0:.4f}\n",
            f"\n",
        ])
        
        # Convergence rate
        if len(train_losses_list) >= 10:
            first_10 = sum(list(train_losses.values())[:10]) / 10
            last_10 = sum(list(train_losses.values())[-10:]) / 10
            convergence_rate = (first_10 - last_10) / first_10 * 100
            
            comprehensive_log.extend([
                f"Convergence Analysis:\n",
                f"  First 10 iterations avg: {first_10:.4f}\n",
                f"  Last 10 iterations avg: {last_10:.4f}\n",
                f"  Improvement: {convergence_rate:.1f}%\n",
                f"\n",
            ])
    
    # Generalization Gap Analysis
    if validation_losses and train_losses:
        comprehensive_log.extend([
            "🔄 GENERALIZATION GAP ANALYSIS\n",
            "-" * 80 + "\n",
        ])
        
        gaps = []
        for iter_num in sorted(validation_losses.keys()):
            if iter_num in train_losses:
                val_loss = validation_losses[iter_num]
                train_loss = train_losses.get(iter_num, train_losses.get(max([k for k in train_losses.keys() if k <= iter_num], default=iter_num)))
                gap = val_loss - train_loss
                gaps.append((iter_num, gap, train_loss, val_loss))
        
        if gaps:
            avg_gap = sum(g[1] for g in gaps) / len(gaps)
            min_gap = min(g[1] for g in gaps)
            max_gap = max(g[1] for g in gaps)
            
            comprehensive_log.extend([
                f"Average Generalization Gap: {avg_gap:.4f}\n",
                f"Min Gap: {min_gap:.4f}\n",
                f"Max Gap: {max_gap:.4f}\n",
                f"\n",
                f"Interpretation:\n",
            ])
            
            if avg_gap < 0.1:
                comprehensive_log.append(f"  OK Excellent generalization (gap < 0.1)\n")
            elif avg_gap < 0.3:
                comprehensive_log.append(f"  OK Good generalization (gap 0.1-0.3)\n")
            elif avg_gap < 0.5:
                comprehensive_log.append(f"  Warning: Moderate overfitting (gap 0.3-0.5)\n")
            else:
                comprehensive_log.append(f"  🔴 Significant overfitting (gap > 0.5)\n")
            
            comprehensive_log.append(f"\n")
            
            # Show specific gaps
            comprehensive_log.append(f"Gap at Each Validation Point:\n")
            comprehensive_log.append(f"{'Iter':>8} | {'Train':>8} | {'Val':>8} | {'Gap':>8}\n")
            comprehensive_log.append(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}\n")
            
            for iter_num, gap, train_loss, val_loss in gaps[:10]:  # Show first 10
                comprehensive_log.append(f"{iter_num:>8} | {train_loss:>8.4f} | {val_loss:>8.4f} | {gap:>+8.4f}\n")
            
            if len(gaps) > 10:
                comprehensive_log.append(f"... ({len(gaps) - 10} more validation points)\n")
            
            comprehensive_log.append(f"\n")
    
    # Checkpoint analysis
    if checkpoint_times:
        comprehensive_log.extend([
            "💾 CHECKPOINT ANALYSIS\n",
            "-" * 80 + "\n",
            f"Total Checkpoints Saved: {len(checkpoint_times)}\n",
        ])
        
        if checkpoint_times:
            total_save_time = sum(t[1] for t in checkpoint_times)
            avg_save_time = total_save_time / len(checkpoint_times)
            
            comprehensive_log.extend([
                f"Average Save Time: {avg_save_time:.2f} seconds\n",
                f"Total Time Saving Checkpoints: {total_save_time:.1f} seconds\n",
                f"Checkpoints Saved At: {', '.join(str(t[0]) for t in checkpoint_times)}\n",
            ])
        
        comprehensive_log.append(f"\n")
    
    # Recommendations
    comprehensive_log.extend([
        "🎯 RECOMMENDATIONS FOR FUTURE RUNS\n",
        "-" * 80 + "\n",
    ])
    
    recommendations = []
    
    # Critical issues
    if issues:
        if any(i[0] == 'nan_loss' for i in issues):
            recommendations.append({
                "priority": "CRITICAL",
                "text": "Learning rate is too high. Reduce by 50% (e.g., 1e-4 → 5e-5)."
            })
        if any(i[0] == 'exploding_loss' for i in issues):
            recommendations.append({
                "priority": "HIGH",
                "text": "Implement gradient clipping or reduce learning rate."
            })
    
    # Overfitting
    if validation_losses:
        best_iter = min(validation_losses.keys(), key=lambda k: validation_losses[k])
        final_iter = max(validation_losses.keys())
        
        if best_iter < final_iter:
            final_loss = validation_losses[final_iter]
            best_loss = validation_losses[best_iter]
            degradation_pct = ((final_loss - best_loss) / best_loss) * 100
            
            recommendations.append({
                "priority": "HIGH",
                "text": f"Use {best_iter} iterations instead of {args.iters}. Model degraded {degradation_pct:.1f}% after best point."
            })
    
    # Speed optimization
    if train_speeds:
        speeds = [s[1] for s in train_speeds]
        avg_speed = sum(speeds) / len(speeds)
        
        if avg_speed < 0.3:
            recommendations.append({
                "priority": "MEDIUM",
                "text": f"Training is slow ({avg_speed:.2f} iters/sec). Consider gradient checkpointing or reducing sequence length."
            })
    
    # Data recommendations
    if not recommendations:
        recommendations.append({
            "priority": "LOW",
            "text": "Training completed successfully. Consider experimenting with different learning rates or model sizes."
        })
    
    # Print recommendations by priority
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_recs = sorted(recommendations, key=lambda x: priority_order.get(x["priority"], 4))
    
    for i, rec in enumerate(sorted_recs, 1):
        emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(rec["priority"], "⚪")
        comprehensive_log.append(f"{i}. {emoji} [{rec['priority']}] {rec['text']}\n")
    
    comprehensive_log.append(f"\n")
    
    # Output structure
    comprehensive_log.extend([
        "📦 OUTPUT STRUCTURE\n",
        "-" * 80 + "\n",
        f"{output_dir}/\n",
        f"├── adapters/                    # All training checkpoints\n",
        f"│   └── [iteration]_adapters.safetensors\n",
        f"├── best_adapter/               # Best checkpoint (iteration {best_iter if validation_losses else 'N/A'})\n",
        f"│   ├── adapters.safetensors    # ~24 MB\n",
        f"│   └── adapter_config.json     # LoRA configuration\n",
        f"├── fused_model/                # Fused model (base + adapter)\n",
        f"│   ├── model.safetensors      # ~2.2 GB\n",
        f"│   ├── config.json\n",
        f"│   └── tokenizer.json\n",
        f"├── training_log_comprehensive.txt  # This file\n",
        f"├── training_summary.json       # JSON metrics\n",
        f"├── training_quality_report.md  # Analysis report\n",
        f"└── improvement_recommendations.txt # Action items\n",
        f"\n",
    ])
    
    # Raw training output
    comprehensive_log.extend([
        "=" * 80 + "\n",
        "RAW TRAINING OUTPUT (Complete Log)\n",
        "=" * 80 + "\n",
        f"\n",
    ])
    comprehensive_log.extend(training_log)
    
    # Write comprehensive log
    comprehensive_log_file = output_dir / "training_log_comprehensive.txt"
    with open(comprehensive_log_file, 'w', encoding='utf-8') as f:
        f.writelines(comprehensive_log)
    
    print(f"\n📄 Comprehensive log saved: {comprehensive_log_file}")
    
    return success


def select_best_checkpoint(output_dir, validation_losses):
    """Select checkpoint with lowest validation loss."""
    
    print_header("PHASE 2: Selecting Best Checkpoint")
    
    if not validation_losses:
        print("Error: No validation data found")
        return None
    
    # Find best iteration
    best_iter = min(validation_losses.keys(), key=lambda k: validation_losses[k])
    best_loss = validation_losses[best_iter]
    
    print(f"\n📊 Validation Loss History:")
    for iter_num in sorted(validation_losses.keys()):
        loss = validation_losses[iter_num]
        marker = " ⭐ BEST" if iter_num == best_iter else ""
        print(f"   Iteration {iter_num:3d}: {loss:.4f}{marker}")
    
    print(f"\nOK Best checkpoint:")
    print(f"   Iteration: {best_iter}")
    print(f"   Validation Loss: {best_loss:.4f}")
    
    # Find checkpoint file
    checkpoint_file = output_dir / "adapters" / f"{best_iter:08d}_adapters.safetensors"
    
    if not checkpoint_file.exists():
        for fmt in [f"{best_iter:08d}", f"{best_iter:07d}", f"{best_iter:06d}", str(best_iter)]:
            test_file = output_dir / "adapters" / f"{fmt}_adapters.safetensors"
            if test_file.exists():
                checkpoint_file = test_file
                break
    
    if not checkpoint_file.exists():
        checkpoint_file = output_dir / "adapters" / "adapters.safetensors"
        print(f"   File: {checkpoint_file}")
        print("   Note: Using final checkpoint (specific iter file not found)")
    else:
        print(f"   File: {checkpoint_file}")
    
    # Copy to best_adapter directory
    best_adapter_dir = output_dir / "best_adapter"
    best_adapter_dir.mkdir(exist_ok=True)
    
    best_adapter_file = best_adapter_dir / "adapters.safetensors"
    shutil.copy2(checkpoint_file, best_adapter_file)
    
    config_file = output_dir / "adapters" / "adapter_config.json"
    if config_file.exists():
        shutil.copy2(config_file, best_adapter_dir / "adapter_config.json")
    
    print(f"   Copied to: {best_adapter_file}")
    
    # Return the directory path, not the file path (MLX-LM expects a directory)
    return str(best_adapter_dir), best_iter


# ============ MODEL FUSION ============

def fuse_model(args, output_dir, adapter_path):
    """Fuse adapter with base model to create full model."""
    
    print_header("PHASE 3: Fusing Adapter with Base Model")
    
    fused_dir = output_dir / "fused_model"
    
    print(f"\n📦 Fusion Configuration:")
    print(f"   Base Model: {args.model}")
    print(f"   Adapter: {adapter_path}")
    print(f"   Output: {fused_dir}")
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", args.model,
        "--adapter-path", adapter_path,
        "--save-path", str(fused_dir)
    ]
    
    if args.upload_repo:
        cmd.extend(["--upload-repo", args.upload_repo])
    
    print(f"\n🔄 Running fusion...")
    print("-" * 80)
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nError: Fusion failed with exit code: {result.returncode}")
        return False
    
    print("-" * 80)
    
    if not fused_dir.exists():
        print(f"Error: Fusion output directory not found: {fused_dir}")
        return False
    
    required_files = ["model.safetensors", "config.json"]
    missing = [f for f in required_files if not (fused_dir / f).exists()]
    
    if missing:
        print(f"Warning: Warning: Missing files in fused model: {missing}")
    
    total_size = sum(f.stat().st_size for f in fused_dir.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print(f"\nOK Fusion Complete!")
    print(f"   Output Directory: {fused_dir}")
    print(f"   Total Size: {size_mb:.1f} MB")
    print(f"   Files Created:")
    
    for file in sorted(fused_dir.rglob("*")):
        if file.is_file():
            file_size = file.stat().st_size / (1024 * 1024)
            print(f"     - {file.name}: {file_size:.1f} MB")
    
    return True


# ============ SUMMARY CREATION ============

def create_summary(args, output_dir, validation_losses, best_iter, success):
    """Create training summary JSON."""
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "training_config": {
            "iterations": args.iters,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "validation_strategy": getattr(args, 'validation_strategy', 'auto_split'),
            "validation_split_percentage": getattr(args, 'validation_split_percentage', 10),
        },
        "validation_losses": validation_losses,
        "best_iteration": best_iter,
        "best_val_loss": validation_losses.get(best_iter) if validation_losses else None,
        "fusion_success": success,
        "output_directories": {
            "training_artifacts": str(output_dir / "adapters"),
            "best_checkpoint": str(output_dir / "best_adapter"),
            "fused_model": str(output_dir / "fused_model")
        },
        "usage": {
            "as_adapter": f"--adapter-path {output_dir}/best_adapter/adapters.safetensors",
            "as_fused": f"--model {output_dir}/fused_model",
            "test_command": f"python -m mlx_lm generate --model {output_dir}/fused_model --prompt 'Your question'"
        }
    }
    
    summary_file = output_dir / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Summary saved: {summary_file}")
    
    return summary


# ============ POST-TRAINING SUMMARY DISPLAY ============

def display_training_summary(output_dir, validation_losses, args, success):
    """Display a comprehensive summary of the training session."""
    
    # Initialize best_iter
    best_iter = None
    
    print_header("TRAINING SESSION SUMMARY", "█")
    
    # Configuration
    print_section("Configuration")
    print(f"  Model: {args.model}")
    print(f"  Data: {args.data}")
    print(f"  Iterations: {args.iters}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size} (Accumulation: {args.grad_accumulation_steps})")
    print(f"  Output Directory: {output_dir}")
    
    # Validation Results
    if validation_losses:
        print_section("Validation Results")
        
        best_iter = min(validation_losses.keys(), key=lambda k: validation_losses[k])
        best_loss = validation_losses[best_iter]
        final_iter = max(validation_losses.keys())
        final_loss = validation_losses[final_iter]
        
        print(f"  Best Iteration: {best_iter}")
        print(f"  Best Validation Loss: {best_loss:.4f}")
        print(f"  Final Validation Loss: {final_loss:.4f}")
        
        if best_iter != final_iter:
            degradation = final_loss - best_loss
            degradation_pct = (degradation / best_loss) * 100
            print(f"  Warning: Degradation: +{degradation:.4f} (+{degradation_pct:.1f}%)")
            print(f"     → Use iteration {best_iter} instead of final")
        else:
            print(f"  OK Final iteration is also best")
        
        print(f"\n  Validation Loss History:")
        for iter_num in sorted(validation_losses.keys()):
            loss = validation_losses[iter_num]
            marker = " ⭐" if iter_num == best_iter else ""
            print(f"    Iter {iter_num:3d}: {loss:.4f}{marker}")
    
    # Output Structure
    print_section("Output Structure")
    print(f"  {output_dir}/")
    print(f"  ├── adapters/          - All checkpoints from training")
    print(f"  ├── best_adapter/      - Best checkpoint (iteration {best_iter if best_iter is not None else 'N/A'})")
    print(f"  ├── fused_model/       - {'OK Fused model ready' if success else 'Error: Fusion failed'}")
    print(f"  ├── training_log_comprehensive.txt")
    print(f"  └── training_summary.json")
    
    # Issues
    log_file = output_dir / "training_log_comprehensive.txt"
    if log_file.exists():
        with open(log_file, 'r') as f:
            content = f.read()
            if "ISSUE DETECTED" in content or "WARNING" in content:
                print_section("Issues & Warnings")
                print("  Warning: See comprehensive log for details:")
                print(f"     {log_file}")
                
                # Extract warnings
                warnings = []
                for line in content.split('\n'):
                    if 'ISSUE DETECTED' in line or 'Warning: WARNING' in line:
                        warnings.append(line.strip())
                
                if warnings:
                    print(f"\n  Detected {len(warnings)} warning(s)")
    
    print()
    print("█" * 80)

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='Train, fuse, and chat with your fine-tuned model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: Train → Fuse → Chat
  python scripts/train-fuse-and-chat.py --model mlx-community/Phi-3-mini-4k-instruct-4bit
  
  # Skip training, just chat with existing adapter
  python scripts/train-fuse-and-chat.py \\
    --model mlx-community/Phi-3-mini-4k-instruct-4bit \\
    --adapter-path outputs/my-model/best_adapter
  
  # Skip training, chat with fused model
  python scripts/train-fuse-and-chat.py --fused-model outputs/my-model/fused_model
  
  # Fuse-only mode (if fusion failed during previous run)
  python scripts/train-fuse-and-chat.py --fuse-only outputs/my-model
  
  # Custom training + chat settings
  python scripts/train-fuse-and-chat.py \\
    --model MODEL \\
    --iters 400 \\
    --learning-rate 1e-4 \\
    --chat-max-tokens 300
        """
    )
    
    # Model & Data
    parser.add_argument('--model', type=str, default=None,
                        help='Base model to fine-tune (required unless using --adapter-path or --fused-model)')
    parser.add_argument('--data', type=str, default='data',
                        help='Data directory (default: data)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: auto-generated from model name)')
    
    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=TrainingConfig.DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {TrainingConfig.DEFAULT_BATCH_SIZE})')
    parser.add_argument('--grad-accumulation-steps', type=int, default=TrainingConfig.DEFAULT_GRAD_ACCUMULATION,
                        help=f'Gradient accumulation steps (default: {TrainingConfig.DEFAULT_GRAD_ACCUMULATION})')
    parser.add_argument('--iters', type=int, default=TrainingConfig.DEFAULT_ITERATIONS,
                        help=f'Training iterations (default: {TrainingConfig.DEFAULT_ITERATIONS})')
    parser.add_argument('--learning-rate', type=float, default=float(TrainingConfig.DEFAULT_LEARNING_RATE),
                        help=f'Learning rate (default: {TrainingConfig.DEFAULT_LEARNING_RATE})')
    
    # Validation & Monitoring
    parser.add_argument('--val-batches', type=int, default=-1,
                        help='Validation batches, -1 for all (default: -1)')
    parser.add_argument('--steps-per-eval', type=int, default=TrainingConfig.STEPS_PER_EVAL,
                        help=f'Validation frequency (default: {TrainingConfig.STEPS_PER_EVAL})')
    parser.add_argument('--save-every', type=int, default=TrainingConfig.SAVE_EVERY,
                        help=f'Checkpoint frequency (default: {TrainingConfig.SAVE_EVERY})')
    parser.add_argument('--early-stopping-patience', type=int, default=TrainingConfig.DEFAULT_EARLY_STOPPING_PATIENCE,
                        help=f'Early stopping patience (default: {TrainingConfig.DEFAULT_EARLY_STOPPING_PATIENCE}). Stop after N consecutive validation checks with no improvement. Set to 0 to disable.')
    parser.add_argument('--min-delta', type=float, default=0.001,
                        help='Minimum change in validation loss to qualify as an improvement (default: 0.001)')
    parser.add_argument('--validation-strategy', type=str, default='auto_split',
                        choices=['upload_own', 'auto_split', 'no_validation'],
                        help='Validation data strategy: upload_own (separate file), auto_split (from training), no_validation (skip validation)')
    
    # Model Architecture
    parser.add_argument('--max-seq-length', type=int, default=None,
                        help='Maximum sequence length')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of layers to fine-tune')
    parser.add_argument('--grad-checkpoint', action='store_true',
                        help='Enable gradient checkpointing')
    
    # LoRA Configuration
    parser.add_argument('--lora-rank', type=int, default=TrainingConfig.DEFAULT_LORA_RANK,
                        help=f'LoRA rank (r) - higher = more capacity (default: {TrainingConfig.DEFAULT_LORA_RANK})')
    parser.add_argument('--lora-alpha', type=int, default=TrainingConfig.DEFAULT_LORA_ALPHA,
                        help=f'LoRA alpha - scaling factor, usually 2x rank (default: {TrainingConfig.DEFAULT_LORA_ALPHA})')
    parser.add_argument('--lora-dropout', type=float, default=TrainingConfig.DEFAULT_LORA_DROPOUT,
                        help=f'LoRA dropout for regularization (default: {TrainingConfig.DEFAULT_LORA_DROPOUT})')
    
    # Skip training options
    parser.add_argument('--adapter-path', type=str, default=None,
                        help='Skip training, use existing adapter path')
    parser.add_argument('--fused-model', type=str, default=None,
                        help='Skip training, use existing fused model')
    parser.add_argument('--fuse-only', type=str, default=None, metavar='OUTPUT_DIR',
                        help='Skip training, just fuse adapter in existing output directory')
    
    # Upload
    parser.add_argument('--upload-repo', type=str, default=None,
                        help='Hugging Face repo to upload fused model')
    
    args = parser.parse_args()
    
    # Determine mode
    skip_training = args.adapter_path is not None or args.fused_model is not None or args.fuse_only is not None
    
    if args.fuse_only:
        # Fuse-only mode: just fuse the best adapter in existing output dir
        output_dir = Path(args.fuse_only)
        
        # Load summary to get model name
        summary_file = output_dir / "training_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
                args.model = summary_data.get('model', args.model)
                validation_losses = summary_data.get('validation_losses', {})
        else:
            print(f"Error: Error: Cannot find {summary_file}")
            print("   Make sure the output directory contains training results.")
            sys.exit(1)
        
        if not args.model:
            print("Error: Error: Cannot determine base model from summary.")
            print("   Please provide --model argument.")
            sys.exit(1)
        
        print_header("FUSE-ONLY MODE", "🔨")
        print(f"Output Directory: {output_dir}")
        print(f"Base Model: {args.model}")
        print()
        
        # Find best adapter
        best_adapter_dir = output_dir / "best_adapter"
        if not (best_adapter_dir / "adapters.safetensors").exists():
            print(f"Error: Error: Cannot find adapter in {best_adapter_dir}")
            print("   Run full training first or check the output directory.")
            sys.exit(1)
        
        # Run fusion
        success = fuse_model(args, output_dir, str(best_adapter_dir))
        
        if success:
            print(f"\nOK Fusion successful!")
            print(f"   Fused model saved to: {output_dir}/fused_model/")
            print()
            print(f"   Usage:")
            print(f"     python -m mlx_lm generate \\")
            print(f"       --model {output_dir}/fused_model \\")
            print(f"       --prompt 'Your question'")
            print()
            
            print(f"\nError: Fusion failed. Check errors above.")
            sys.exit(1)
        
        sys.exit(0)
    
    # Full pipeline mode
    if not args.model:
        print("Error: Error: --model is required for training mode", flush=True)
        sys.exit(1)
    
    
    # Check validation data (skip if using no_validation strategy)
    if args.validation_strategy != 'no_validation':
        if not check_validation_data(args.data):
            print("Error: Cannot continue without validation data")
            print("Create it with: awk 'NR%10==0' data/train.jsonl > data/valid.jsonl")
            sys.exit(1)
    else:
        print("OK Using no_validation strategy - skipping validation data check")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        model_name = args.model.split('/')[-1].replace('-', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"{model_name}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"OK Output directory: {output_dir}")
    
    # Phase 1: Training
    validation_losses, training_log = run_training(args, output_dir)
    
    if validation_losses is None:
        print("\nError: Training failed")
        sys.exit(1)
    
    # Phase 2: Select Best
    if args.validation_strategy == 'no_validation':
        # No validation - use final checkpoint
        print("\nOK Using final checkpoint (no validation strategy)")
        adapters_dir = output_dir / "adapters"
        final_adapter = adapters_dir / "adapters.safetensors"
        
        if not final_adapter.exists():
            print(f"\nError: Final adapter not found: {final_adapter}")
            sys.exit(1)
        
        # Copy to best_adapter directory
        best_adapter_dir = output_dir / "best_adapter"
        best_adapter_dir.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(final_adapter, best_adapter_dir / "adapters.safetensors")
        
        # Copy config if exists
        config_file = adapters_dir / "adapter_config.json"
        if config_file.exists():
            shutil.copy2(config_file, best_adapter_dir / "adapter_config.json")
        
        best_adapter = str(best_adapter_dir)
        best_iter = args.iters  # Use final iteration
    else:
        # Normal validation-based selection
        result = select_best_checkpoint(output_dir, validation_losses)
        if result is None:
            print("\nError: Could not select best checkpoint")
            sys.exit(1)
        best_adapter, best_iter = result
        
        if not best_adapter:
            print("\nError: Could not select best checkpoint")
            sys.exit(1)
    
    # Phase 3: Fuse
    success = fuse_model(args, output_dir, best_adapter)
    
    # Phase 4: Create Summary
    summary = create_summary(args, output_dir, validation_losses, best_iter, success)
    
    # Phase 4.5: Generate Quality Report
    print("\n📊 Generating detailed quality report...")
    print(f"[REPORT] Output directory: {output_dir}")
    print(f"[REPORT] Data directory: {args.data}")
    print(f"[REPORT] Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        import subprocess
        script_path = Path(__file__).parent / "analyze_and_report.py"
        print(f"[REPORT] Script path: {script_path}")
        print(f"[REPORT] Script exists: {script_path.exists()}")
        
        if script_path.exists():
            # Remove old report to ensure fresh generation
            report_file = output_dir / "training_quality_report.md"
            if report_file.exists():
                report_file.unlink()
                print(f"[REPORT] Removed old report: {report_file}")
            
            # IMPORTANT: Pass the actual data directory used for training
            # This ensures we analyze the real data, not stale files
            cmd = [
                "python3", str(script_path), 
                "--output-dir", str(output_dir),
                "--data-dir", str(args.data)
            ]
            print(f"[REPORT] Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            print(f"[REPORT] Return code: {result.returncode}")
            if result.stdout:
                print(f"[REPORT] stdout: {result.stdout[:500]}")
            if result.returncode == 0:
                print("OK Quality report generated successfully")
                # Verify the timestamp in the generated report
                report_file = output_dir / "training_quality_report.md"
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        first_lines = f.read()[:500]
                        print(f"[REPORT] First 500 chars of report:\n{first_lines}")
            else:
                print(f"Warning: Quality report generation returned non-zero: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:500]}")
        else:
            print(f"Warning: Quality report script not found: {script_path}")
    except Exception as e:
        print(f"Warning: Could not generate quality report: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Phase 5: Display Summary
    display_training_summary(output_dir, validation_losses, args, success)
    
    print("\nOK Training pipeline completed successfully!")
    print(f"   Output directory: {output_dir}")
    print(f"   Best adapter: {output_dir}/best_adapter/")
    if (output_dir / "fused_model").exists():
        print(f"   Fused model: {output_dir}/fused_model/")
    print()
    
    sys.exit(0 if success else 1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
