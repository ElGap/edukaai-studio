"""Train Tab for EdukaAI Studio.

Handles real training execution and monitoring with full MLX integration.
"""

import json
import queue
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Generator, Tuple

import gradio as gr

from edukaai_studio.ui.training_monitor import TrainingMonitor

# Import debug logger
try:
    from edukaai_studio.ui.debug_logger import (
        log_debug, log_info, log_warning, log_error, log_exception
    )
    DEBUG_LOGGING = True
except ImportError:
    DEBUG_LOGGING = False


def create_loss_plot(train_losses: Dict[int, float], val_losses: Dict[int, float], total_iters: int = None):
    """Create loss curve plot from training data.
    
    Args:
        train_losses: Dictionary of iteration -> training loss
        val_losses: Dictionary of iteration -> validation loss
        total_iters: Total number of iterations for x-axis limit
        
    Returns:
        Matplotlib figure or None
    """
    if not train_losses and not val_losses:
        return None
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Clear any existing figures to prevent memory leak
        plt.close('all')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training loss
        if train_losses:
            train_iters = sorted(train_losses.keys())
            train_vals = [train_losses[i] for i in train_iters]
            ax.plot(train_iters, train_vals, 'b-', label='Training Loss', linewidth=2)
        
        # Plot validation loss
        if val_losses:
            val_iters = sorted(val_losses.keys())
            val_vals = [val_losses[i] for i in val_iters]
            ax.plot(val_iters, val_vals, 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        
        # Set x-axis to show full training range
        if total_iters and total_iters > 0:
            ax.set_xlim(0, total_iters)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    except ImportError:
        return None
    except Exception as e:
        print(f"[PLOT] Error creating plot: {e}")
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except Exception:
            pass
        return None


def start_training_real(current_state: Dict[str, Any]) -> Generator:
    """Start REAL training using TrainingMonitor.
    
    Args:
        current_state: Current application state
        
    Yields:
        List of outputs for Gradio components
    """
    import time as _time_module  # Local import to avoid scoping issues
    
    # FIRST: Clear any previous training state to ensure fresh start
    # This prevents "stale state" issues where old completion data interferes
    print(f"[TRAIN] Clearing previous training state for fresh start...")
    from edukaai_studio.core.state import clear_state_file, save_state_to_disk
    clear_state_file()
    
    # Reset training state in current_state to ensure we don't use stale data
    current_state = {
        **current_state,
        'training_complete': False,
        'training_active': False,
        'output_dir': None,
        'completion_time': None,
        'train_losses': {},
        'val_losses': {},
        'best_loss': float('inf'),
        'best_iter': 0,
    }
    
    # Save cleared state to disk immediately
    save_state_to_disk(current_state)
    print(f"[TRAIN] State cleared and saved. Ready to start new training.")
    
    # Check prerequisites
    if not current_state.get('uploaded_file'):
        yield [
            0, "0 / 0", "-", "-", "-", "-", "-", "-",
            "Error: No training data uploaded. Upload in Step 1.",
            None,
            "Error: Upload required",
            current_state
        ]
        return
    
    if not current_state.get('training_config'):
        yield [
            0, "0 / 0", "-", "-", "-", "-", "-", "-",
            "Error: Training not configured. Configure in Step 2.",
            None,
            "Error: Configuration required",
            current_state
        ]
        return
    
    config = current_state['training_config']
    total_iters = config.get('iterations', 600)
    data_file = current_state['uploaded_file']
    
    # Debug: log what config was received
    print(f"[TRAIN DEBUG] Received training_config: {config}")
    print(f"[TRAIN DEBUG] Iterations from config: {config.get('iterations')}")
    print(f"[TRAIN DEBUG] Using total_iters: {total_iters}")
    
    # Initialize training state (immutable pattern)
    current_state = {
        **current_state,
        'training_active': True,
        'training_complete': False,
        'train_losses': {},
        'val_losses': {},
        'best_loss': float('inf'),
        'best_iter': 0,
    }
    
    log_lines = [
        f"[{datetime.now().strftime('%H:%M:%S')}] Initializing real training with MLX...",
        f"Model: {config.get('model_name', 'Unknown')}",
        f"Base model: {config.get('model_id', 'unknown')}",
        f"Iterations: {total_iters}",
        f"Learning rate: {config.get('learning_rate', '1e-4')}",
        f"LoRA rank: {config.get('lora_rank', 16)}",
        f"Data: {data_file}",
        ""
    ]
    last_logged_iter = 0
    
    yield [
        0, f"0 / {total_iters}", "-", "-", "-", "-", "-", "-",
        "\n".join(log_lines),
        None,
        "Initializing training...",
        current_state
    ]
    
    # Check if TrainingMonitor is available
    if TrainingMonitor:
        try:
            # Create queues for communication
            progress_queue = queue.Queue()
            output_queue = queue.Queue()
            
            # Initialize monitor
            monitor = TrainingMonitor(output_queue, progress_queue)
            current_state = {**current_state, 'monitor': monitor}
            
            # Build training arguments
            from edukaai_studio.ui.user_config import get_hf_token
            hf_token = get_hf_token()
            if hf_token:
                log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Using HF token for gated model access")
            
            training_args = {
                'model': config.get('model_id'),
                'iters': config.get('iterations', 600),
                'learning_rate': float(config.get('learning_rate', '1e-4')),
                'batch_size': 1,
                'grad_accumulation': config.get('grad_accumulation', 32),
                'max_seq_length': config.get('max_seq_length', 2048),
                'lora_rank': config.get('lora_rank', 16),
                'lora_alpha': config.get('lora_alpha', 32),
                'lora_dropout': 0.0,
                'lora_modules': 'all',
                'early_stopping': config.get('early_stopping', 2),
                'grad_checkpoint': False,
                'hf_token': hf_token,
            }
            
            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training configuration:")
            log_lines.append(f"  Model: {config.get('model_name', 'Unknown')}")
            log_lines.append(f"  Base model ID: {config.get('model_id', 'unknown')}")
            log_lines.append(f"  Iterations: {total_iters}")
            log_lines.append(f"  Learning rate: {config.get('learning_rate', '1e-4')}")
            log_lines.append(f"  LoRA rank: {config.get('lora_rank', 16)}")
            log_lines.append(f"  LoRA alpha: {config.get('lora_alpha', 32)}")
            log_lines.append(f"  Gradient accumulation: {config.get('grad_accumulation', 32)}")
            log_lines.append(f"  Early stopping: {config.get('early_stopping', 2)}")
            log_lines.append(f"  Validation split: {config.get('validation_split', 10)}%")
            log_lines.append("")
            
            yield [
                0, f"0 / {total_iters}", "-", "-", "-", "-", "-", "-",
                "\n".join(log_lines),
                None,
                "Starting training...",
                current_state
            ]
            
            # Start training
            success = monitor.start_training(
                training_args,
                data_file,
                validation_file=None,
                validation_strategy='auto_split',
                validation_split_percentage=config.get('validation_split', 10)
            )
            
            if not success:
                log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Failed to start training")
                yield [
                    0, f"0 / {total_iters}", "-", "-", "-", "-", "-", "-",
                    "\n".join(log_lines),
                    None,
                    "Error: Failed to start training",
                    current_state
                ]
                return
            
            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training started successfully")
            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring training progress...")
            log_lines.append("")
            
            # Register model in My Models as "running"
            try:
                from edukaai_studio.core.trained_models_registry import get_registry
                registry = get_registry()
                
                # Determine output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = config.get('model_name', 'model').replace(' ', '_')
                output_dir = f"outputs/{model_name}_{timestamp}"
                
                model_data = {
                    'output_dir': output_dir,
                    'base_model_id': config.get('model_id', 'unknown'),
                    'base_model_name': config.get('model_name', 'Unknown'),
                    'iterations': total_iters,
                    'learning_rate': str(config.get('learning_rate', '1e-4')),
                    'lora_rank': config.get('lora_rank', 16),
                    'lora_alpha': config.get('lora_alpha', 32),
                    'lora_dropout': 0.0,
                    'batch_size': 1,
                    'grad_accumulation': config.get('grad_accumulation', 32),
                    'dataset_path': data_file,
                    'dataset_size': config.get('dataset_size', 0),
                    'best_loss': float('inf'),
                    'final_loss': float('inf'),
                    'best_iteration': 0,
                    'train_losses': {},
                    'val_losses': {},
                    'training_duration_minutes': 0.0,
                    'exports': {
                        'adapter': None,
                        'fused': None,
                        'gguf': None
                    },
                    'tags': [],
                    'notes': f"Training started at {timestamp}",
                    'status': 'running'
                }
                
                model_id = registry.register_model(model_data)
                print(f"[TRAIN] Registered running model: {model_id}")
                log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Model registered in My Models (ID: {model_id})")
                current_state['registered_model_id'] = model_id
                
            except Exception as e:
                print(f"[TRAIN] Could not register model: {e}")
            
            # Monitor training progress
            current_iter = 0
            train_losses = {}
            val_losses = {}
            best_loss = float('inf')
            best_iter = 0
            peak_memory_gb = 0.0
            
            # Wait for process to start and produce some output
            print(f"[TRAIN] Waiting for training process to initialize...")
            startup_wait = 0
            while monitor.is_running() and startup_wait < 50:  # Wait up to 5 seconds
                _time_module.sleep(0.1)
                startup_wait += 1
                # Collect any startup output
                try:
                    while True:
                        line = output_queue.get_nowait()
                        if line.strip():
                            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line.strip()}")
                except queue.Empty:
                    pass
            
            # Check if process exited immediately
            if not monitor.is_running():
                print(f"[TRAIN ERROR] Training process exited immediately!")
                print(f"[TRAIN ERROR] Log lines collected: {len(log_lines)}")
                
                # Try to get any remaining output
                try:
                    while True:
                        line = output_queue.get_nowait()
                        if line.strip():
                            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line.strip()}")
                except queue.Empty:
                    pass
                
                if len(log_lines) == 0:
                    log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: Process exited with no output")
                    log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] This usually means the training script crashed immediately")
                    log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Check: python scripts/lora-train.py --help")
                
                yield [
                    0, "0 / 0", "-", "-", "-", "-", "-", "-",
                    "\n".join(log_lines),
                    None,
                    "Error: Training failed to start. Check logs above.",
                    current_state
                ]
                return
            
            print(f"[TRAIN] Training process is running, entering monitoring loop...")
            
            while monitor.is_running():
                # Get progress updates
                try:
                    while True:
                        progress_data = progress_queue.get_nowait()
                        if progress_data:
                            current_iter = progress_data.get('iteration', 0)
                            progress_percent = progress_data.get('progress_percent', 0)
                            
                            # Update losses from progress_data (monitor sends 'train_losses' dict)
                            if 'train_losses' in progress_data:
                                # Monitor sends full dict of all train losses
                                train_losses.update(progress_data['train_losses'])
                            if 'val_losses' in progress_data:
                                # Monitor sends full dict of all validation losses
                                val_losses.update(progress_data['val_losses'])
                            
                            # Also check for singular keys (backward compatibility)
                            if 'train_loss' in progress_data and current_iter > 0:
                                train_losses[current_iter] = progress_data['train_loss']
                            if 'val_loss' in progress_data and current_iter > 0:
                                val_losses[current_iter] = progress_data['val_loss']
                            
                            # Track best loss
                            if train_losses and current_iter in train_losses:
                                current_train_loss = train_losses[current_iter]
                                if current_train_loss < best_loss:
                                    best_loss = current_train_loss
                                    best_iter = current_iter
                            
                            # Update resource stats
                            peak_memory_gb = progress_data.get('peak_memory_gb', peak_memory_gb)
                            
                            # Log every 10 iterations
                            if current_iter % 10 == 0 and current_iter != last_logged_iter:
                                train_loss = train_losses.get(current_iter, 0.0)
                                val_loss = val_losses.get(current_iter, 0.0)
                                memory_gb = progress_data.get('memory_gb', 0.0)
                                log_lines.append(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] Step {current_iter}/{total_iters} | "
                                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                                    f"Best: {best_loss:.4f} | Memory: {memory_gb:.2f}GB"
                                )
                                last_logged_iter = current_iter
                                
                                # Create plot
                                plot = create_loss_plot(train_losses, val_losses, total_iters)
                                
                                # Yield progress update
                                yield [
                                    progress_percent,
                                    f"{current_iter} / {total_iters}",
                                    f"{train_losses.get(current_iter, 0.0):.4f}",
                                    f"{val_losses.get(current_iter, 0.0):.4f}",
                                    f"{best_loss:.4f}",
                                    f"{peak_memory_gb:.2f}",
                                    "-",
                                    "-",
                                    "\n".join(log_lines),
                                    plot,
                                    f"Training... Step {current_iter}/{total_iters}",
                                    current_state
                                ]
                except queue.Empty:
                    pass
                
                # Get log output
                try:
                    while True:
                        log_line = output_queue.get_nowait()
                        if log_line.strip():
                            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_line.strip()}")
                except queue.Empty:
                    pass
                
                _time_module.sleep(0.1)
            
            # Process final progress data
            final_progress_data = None
            try:
                while True:
                    progress_data = progress_queue.get_nowait()
                    if progress_data:
                        final_progress_data = progress_data
                        train_losses = progress_data.get('train_losses', {})
                        val_losses = progress_data.get('val_losses', {})
                        current_state = {
                            **current_state,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                        }
            except queue.Empty:
                pass
            
            # Wait for monitoring loop to finish (process exited)
            print(f"[TRAIN] Waiting for training process to finish...")
            while monitor.is_running():
                _time_module.sleep(0.1)
            
            print(f"[TRAIN] Process finished, waiting for output reader to complete...")
            # Give the output reader thread time to finish processing remaining output
            # and set training_complete = True
            _time_module.sleep(1.0)
            
            # Handle completion
            was_stopped = hasattr(monitor, 'was_stopped') and monitor.was_stopped
            
            # Try to wait for output reader thread if we have the reference
            if hasattr(monitor, '_output_thread') and monitor._output_thread:
                print(f"[TRAIN] Joining output reader thread...")
                monitor._output_thread.join(timeout=2.0)
                if monitor._output_thread.is_alive():
                    print(f"[TRAIN WARNING] Output thread still alive after timeout")
            
            # Check completion: either monitor says complete OR process exited successfully
            is_complete = monitor.is_complete() or (
                monitor.process and 
                monitor.process.poll() is not None and 
                monitor.process.returncode == 0 and
                not was_stopped
            )
            
            print(f"[TRAIN DEBUG] Completion check:")
            print(f"  monitor.is_complete(): {monitor.is_complete()}")
            print(f"  process.returncode: {monitor.process.returncode if monitor.process else 'N/A'}")
            print(f"  was_stopped: {was_stopped}")
            print(f"  is_complete: {is_complete}")
            
            # Yield final progress update if available
            if final_progress_data and not was_stopped:
                final_iter = final_progress_data.get('iteration', 0)
                final_total = final_progress_data.get('total', total_iters)
                final_percent = final_progress_data.get('progress_percent',
                    int((final_iter / final_total) * 100) if final_total > 0 else 0)
                
                t_loss = final_progress_data.get('train_loss', 0.0)
                v_loss = final_progress_data.get('val_loss', 0.0)
                b_loss = final_progress_data.get('best_loss', best_loss)
                mem_gb = final_progress_data.get('peak_memory_gb', peak_memory_gb)
                
                final_plot = create_loss_plot(train_losses, val_losses, total_iters)
                
                yield [
                    final_percent,
                    f"{final_iter} / {final_total}",
                    f"{t_loss:.4f}",
                    f"{v_loss:.4f}",
                    f"{b_loss:.4f}",
                    f"{mem_gb:.2f}",
                    "-",
                    "-",
                    "\n".join(log_lines),
                    final_plot,
                    f"Final update: Step {final_iter}/{final_total}",
                    current_state
                ]
            
            if was_stopped:
                completion_time = datetime.now().strftime('%H:%M:%S')
                log_lines.append(f"[{completion_time}] Training stopped by user")
                
                train_losses = current_state.get('train_losses', {})
                val_losses = current_state.get('val_losses', {})
                final_train_loss = list(train_losses.values())[-1] if train_losses else 0.0
                final_val_loss = list(val_losses.values())[-1] if val_losses else 0.0
                best_loss = current_state.get('best_loss', float('inf'))
                peak_memory_gb = current_state.get('peak_memory_gb', 0.0)
                
                final_plot = create_loss_plot(train_losses, val_losses, total_iters)
                final_memory_str = f"{peak_memory_gb:.2f}" if peak_memory_gb > 0 else "-"
                current_iter_val = list(train_losses.keys())[-1] if train_losses else 0
                
                yield [
                    int((current_iter_val / total_iters) * 100) if total_iters > 0 else 0,
                    f"{current_iter_val} / {total_iters}",
                    f"{final_train_loss:.4f}",
                    f"{final_val_loss:.4f}",
                    f"{best_loss:.4f}",
                    final_memory_str,
                    "-",
                    "-",
                    "\n".join(log_lines),
                    final_plot,
                    "Error: Training stopped by user",
                    current_state
                ]
                return
            
            elif is_complete:
                completion_time = datetime.now().strftime('%H:%M:%S')
                log_lines.append(f"[{completion_time}] Training complete!")
                
                # Add resource summary
                try:
                    from edukaai_studio.ui.resource_monitor import ResourceMonitor
                    resource_monitor = ResourceMonitor()
                    resource_summary = resource_monitor.get_summary()
                    log_lines.append(f"[{completion_time}] Resource Usage Summary:")
                    for line in resource_summary.strip().split('\n'):
                        if line.strip():
                            log_lines.append(f"[{completion_time}]   {line}")
                except Exception as e:
                    print(f"[RESOURCE] Could not get resource summary: {e}")
                
                # Parse output directory from logs
                actual_output_dir = None
                print(f"[TRAIN DEBUG] Searching for output_dir in {len(log_lines)} log lines...")
                for i, line in enumerate(reversed(log_lines)):
                    if i < 5:
                        print(f"[TRAIN DEBUG] Checking line: {line[:100]}")
                    # Match patterns like "Output: outputs/..." or "Output directory: outputs/..."
                    match = re.search(r'Output\s*:?\s*(outputs/[^\s]+)', line, re.IGNORECASE)
                    if match:
                        full_path = match.group(1)
                        parts = full_path.split('/')
                        if len(parts) >= 2:
                            actual_output_dir = '/'.join(parts[:2])
                        else:
                            actual_output_dir = full_path
                        print(f"[TRAIN DEBUG] Found output_dir: {actual_output_dir}")
                        break
                    # Also try matching "OK Output directory: outputs/..."
                    match2 = re.search(r'OK Output directory:\s*(outputs/[^\s]+)', line, re.IGNORECASE)
                    if match2:
                        full_path = match2.group(1)
                        parts = full_path.split('/')
                        if len(parts) >= 2:
                            actual_output_dir = '/'.join(parts[:2])
                        else:
                            actual_output_dir = full_path
                        print(f"[TRAIN DEBUG] Found output_dir (alt pattern): {actual_output_dir}")
                        break
                
                if not actual_output_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = config.get('model_name', 'model').replace(' ', '_')
                    actual_output_dir = f"outputs/{model_name}_{timestamp}"
                    print(f"[TRAIN DEBUG] Using fallback output_dir: {actual_output_dir}")
                
                # Update registered model status
                try:
                    from edukaai_studio.core.trained_models_registry import get_registry
                    registry = get_registry()
                    
                    model_id = current_state.get('registered_model_id')
                    if model_id:
                        registry.update_model(
                            model_id,
                            status='completed',
                            best_loss=best_loss,
                            final_loss=list(train_losses.values())[-1] if train_losses else best_loss,
                            train_losses=train_losses,
                            val_losses=val_losses,
                            training_duration_minutes=0.0,
                            exports={
                                'adapter': f"{actual_output_dir}/adapters/adapters.safetensors",
                                'fused': f"{actual_output_dir}/fused_model" if False else None,
                                'gguf': None
                            }
                        )
                        print(f"[TRAIN] Updated model status to completed: {model_id}")
                        log_lines.append(f"[{completion_time}] Model status updated to completed")
                    else:
                        # Auto-register on completion if not registered
                        model_data = {
                            'output_dir': actual_output_dir,
                            'base_model_id': config.get('model_id', 'unknown'),
                            'base_model_name': config.get('model_name', 'Unknown'),
                            'iterations': total_iters,
                            'learning_rate': str(config.get('learning_rate', '1e-4')),
                            'lora_rank': config.get('lora_rank', 16),
                            'lora_alpha': config.get('lora_alpha', 32),
                            'lora_dropout': 0.0,
                            'batch_size': 1,
                            'grad_accumulation': config.get('grad_accumulation', 32),
                            'dataset_path': data_file,
                            'dataset_size': config.get('dataset_size', 0),
                            'best_loss': best_loss,
                            'final_loss': list(train_losses.values())[-1] if train_losses else best_loss,
                            'best_iteration': best_iter,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'training_duration_minutes': 0.0,
                            'exports': {
                                'adapter': f"{actual_output_dir}/adapters/adapters.safetensors",
                                'fused': None,
                                'gguf': None
                            },
                            'tags': [],
                            'notes': f"Training completed on {completion_time}",
                            'status': 'completed'
                        }
                        
                        model_id = registry.register_model(model_data)
                        print(f"[TRAIN] Auto-registered model in My Models: {model_id}")
                        log_lines.append(f"[{completion_time}] Model registered in My Models tab")
                    
                except Exception as e:
                    print(f"[TRAIN] Could not auto-register model: {e}")
                    log_lines.append(f"[{completion_time}] Note: Model not registered (will be available after scan)")
                
                current_state = {
                    **current_state,
                    'training_active': False,
                    'training_complete': True,
                    'completion_time': completion_time,
                    'output_dir': actual_output_dir,
                }
                
                # Ensure state is fully updated before saving
                _time_module.sleep(0.1)  # Small delay to ensure state consistency
                
                # Save state
                print(f"[TRAIN DEBUG] Saving COMPLETED state:")
                print(f"  training_active: {current_state['training_active']}")
                print(f"  training_complete: {current_state['training_complete']}")
                print(f"  output_dir: {actual_output_dir}")
                from edukaai_studio.core.state import save_state_to_disk, load_state_from_disk
                save_result = save_state_to_disk(current_state)
                print(f"[TRAIN DEBUG] State saved: {save_result}")
                
                # Verify the save by reading it back immediately
                if save_result:
                    verify_state = load_state_from_disk()
                    if verify_state:
                        print(f"[TRAIN DEBUG] Verification - loaded from disk:")
                        print(f"  training_complete: {verify_state.get('training_complete')}")
                        print(f"  training_active: {verify_state.get('training_active')}")
                        if verify_state.get('training_complete') != True:
                            print(f"[TRAIN ERROR] State verification FAILED! Disk shows training_complete=False")
                    else:
                        print(f"[TRAIN ERROR] Could not load state for verification!")
                
                train_losses = current_state.get('train_losses', {})
                val_losses = current_state.get('val_losses', {})
                final_train_loss = list(train_losses.values())[-1] if train_losses else 0.0
                final_val_loss = list(val_losses.values())[-1] if val_losses else 0.0
                best_loss = current_state.get('best_loss', float('inf'))
                peak_memory_gb = current_state.get('peak_memory_gb', 0.0)
                
                final_plot = create_loss_plot(train_losses, val_losses, total_iters)
                final_memory_str = f"{peak_memory_gb:.2f}" if peak_memory_gb > 0 else "-"
                
                yield [
                    100,
                    f"{total_iters} / {total_iters}",
                    f"{final_train_loss:.4f}",
                    f"{final_val_loss:.4f}",
                    f"{best_loss:.4f}",
                    final_memory_str,
                    "-",
                    "-",
                    "\n".join(log_lines),
                    final_plot,
                    "✅ Training Complete!",
                    current_state
                ]
                return
            else:
                yield [
                    0, "0 / 0", "-", "-", "-", "-", "-", "-",
                    "\n".join(log_lines),
                    None,
                    "Training stopped",
                    current_state
                ]
                return
                
        except Exception as e:
            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {str(e)}")
            import traceback
            log_lines.append(traceback.format_exc())
            yield [
                0, "0 / 0", "-", "-", "-", "-", "-", "-",
                "\n".join(log_lines),
                None,
                f"Error: {str(e)}",
                current_state
            ]
    else:
        log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: TrainingMonitor not available")
        log_lines.append("Please ensure ui.training_monitor is properly installed")
        yield [
            0, "0 / 0", "-", "-", "-", "-", "-", "-",
            "\n".join(log_lines),
            None,
            "Error: Training infrastructure not available",
            current_state
        ]


def stop_training_handler(current_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Stop the training process.
    
    Args:
        current_state: Current application state
        
    Returns:
        Tuple of (status_message, new_state)
    """
    monitor = current_state.get('monitor')
    if monitor:
        try:
            monitor.stop()
            print("[STOP] Training monitor stopped")
        except Exception as e:
            print(f"[STOP] Error stopping monitor: {e}")
    
    new_state = {**current_state, 'training_active': False}
    return "Training stop requested", new_state


def create_train_tab(state: gr.State, tabs: gr.Tabs) -> Dict[str, Any]:
    """Create the Train tab.
    
    Args:
        state: Gradio state object
        tabs: Gradio tabs container
        
    Returns:
        Dictionary of component references
    """
    components = {}
    
    with gr.TabItem("3. Train"):
        gr.Markdown("## Step 3: Training")
        gr.Markdown("**Real training with MLX will take time. Do not close the browser.**")
        
        # Two-column layout
        with gr.Row():
            with gr.Column(scale=1):
                components['progress_slider'] = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="Progress", interactive=False
                )
                components['step_display'] = gr.Textbox(
                    label="Step", value="0 / 0", interactive=False
                )
                
                with gr.Row():
                    components['train_loss'] = gr.Textbox(
                        label="Training Loss", value="-", interactive=False
                    )
                    components['val_loss'] = gr.Textbox(
                        label="Validation Loss", value="-", interactive=False
                    )
                    components['best_loss'] = gr.Textbox(
                        label="Best Loss", value="-", interactive=False
                    )
                    components['memory_display'] = gr.Textbox(
                        label="Memory (GB)", value="-", interactive=False
                    )
                
                with gr.Row():
                    components['cpu_display'] = gr.Textbox(
                        label="CPU %", value="-", interactive=False
                    )
                    components['ram_display'] = gr.Textbox(
                        label="RAM %", value="-", interactive=False
                    )
                
                with gr.Row():
                    components['start_btn'] = gr.Button(
                        "Start Training", variant="primary", size="sm"
                    )
                    components['stop_btn'] = gr.Button(
                        "Stop Training", variant="stop", size="sm"
                    )
            
            with gr.Column(scale=1):
                components['loss_plot'] = gr.Plot(label="Loss Curve", value=None)
        
        gr.Markdown("### Training Log")
        
        # Enhanced scrollable log display with auto-scroll and complete history
        components['log_display'] = gr.Textbox(
            label="Training Log Output (Complete History)",
            lines=25,
            interactive=False,
            value="Ready to start training...",
            elem_classes="training-log-scrollable"
        )
        
        # Add custom CSS for scrollable log area
        gr.HTML("""
        <style>
        .training-log-scrollable textarea {
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace !important;
            font-size: 11px !important;
            line-height: 1.4 !important;
            background-color: #1e1e1e !important;
            color: #d4d4d4 !important;
            overflow-y: auto !important;
            max-height: 600px !important;
            min-height: 400px !important;
            border: 1px solid #333 !important;
        }
        .training-log-scrollable textarea::-webkit-scrollbar {
            width: 10px;
        }
        .training-log-scrollable textarea::-webkit-scrollbar-track {
            background: #2d2d2d;
        }
        .training-log-scrollable textarea::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 5px;
        }
        .training-log-scrollable textarea::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
        </style>
        """)
        components['train_status'] = gr.Textbox(
            label="Status", value="Ready", interactive=False
        )
        
        # Wire up events
        components['start_btn'].click(
            fn=start_training_real,
            inputs=[state],
            outputs=[
                components['progress_slider'], components['step_display'],
                components['train_loss'], components['val_loss'],
                components['best_loss'], components['memory_display'],
                components['cpu_display'], components['ram_display'],
                components['log_display'], components['loss_plot'],
                components['train_status'], state
            ]
        )
        
        components['stop_btn'].click(
            fn=stop_training_handler,
            inputs=[state],
            outputs=[components['train_status'], state]
        )
    
    return components
