#!/usr/bin/env python3
"""
EdukaAI Studio - Fully Functional Fine-Tuning UI with Real Training

Real training, real chat, working downloads.
"""

import gradio as gr
import sys
import time
import subprocess
import os
import json
import queue
import threading
import re
import random
from pathlib import Path
from datetime import datetime

# Import configuration
from edukaai_studio.config import STUDIO_MODELS, SERVER, TRAINING

# Import UI components
try:
    from edukaai_studio.ui.training_monitor import TrainingMonitor
    from edukaai_studio.ui.chat_wrapper import ChatWrapper
except ImportError as e:
    print(f"Import warning: {e}")
    TrainingMonitor = None
    ChatWrapper = None


# File-based state persistence for critical training state
# Use absolute path to ensure file is created in the right location
STATE_FILE = Path(__file__).parent.parent.parent / ".studio_state.json"
print(f"[STATE] State file location: {STATE_FILE}")

def save_state_to_disk(state):
    """Save critical training state to disk."""
    try:
        print(f"[STATE SAVE] Attempting to save to: {STATE_FILE}")
        print(f"[STATE SAVE] State file parent exists: {STATE_FILE.parent.exists()}")
        print(f"[STATE SAVE] Current state: training_complete={state.get('training_complete')}, output_dir={state.get('output_dir')}")
        
        data = {
            'training_complete': state.get('training_complete', False),
            'training_active': state.get('training_active', False),
            'output_dir': state.get('output_dir', None),
            'completion_time': state.get('completion_time', None),
            'model_name': state.get('training_config', {}).get('model_name', None),
        }
        
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Verify file was written
        if STATE_FILE.exists():
            file_size = STATE_FILE.stat().st_size
            print(f"[STATE SAVED] SUCCESS! File: {STATE_FILE} ({file_size} bytes)")
            # Read back to verify
            with open(STATE_FILE, 'r') as f:
                saved = json.load(f)
            print(f"[STATE SAVED] Verified content: {saved}")
        else:
            print(f"[STATE SAVE ERROR] FAILED! File not created at {STATE_FILE}")
    except Exception as e:
        print(f"[STATE SAVE ERROR] {e}")
        import traceback
        print(f"[STATE SAVE ERROR] {traceback.format_exc()}")

def load_state_from_disk():
    """Load training state from disk."""
    try:
        print(f"[STATE LOAD] Checking: {STATE_FILE}")
        print(f"[STATE LOAD] File exists: {STATE_FILE.exists()}")
        
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
            print(f"[STATE LOADED] Data: {data}")
            return data
        else:
            print(f"[STATE LOAD] No state file found at {STATE_FILE}")
    except Exception as e:
        print(f"[STATE LOAD ERROR] {e}")
        import traceback
        print(f"[STATE LOAD ERROR] {traceback.format_exc()}")
    return {}


def auto_discover_latest_training():
    """Find the most recent completed training in outputs directory."""
    outputs_base = Path("outputs")
    if not outputs_base.exists():
        return None, None
    
    latest_training = None
    latest_time = 0
    latest_model_name = None
    
    for subdir in outputs_base.iterdir():
        if subdir.is_dir():
            # Check if this looks like a completed training (has fused_model or adapters)
            has_fused = (subdir / "fused_model").exists()
            has_adapters = (subdir / "adapters" / "adapters.safetensors").exists() or \
                          (subdir / "best_adapter" / "adapters.safetensors").exists()
            
            if has_fused or has_adapters:
                mtime = subdir.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_training = str(subdir)
                    # Try to extract model name from directory name
                    dir_name = subdir.name
                    # Remove timestamp suffix if present
                    if '_20' in dir_name:
                        latest_model_name = dir_name.rsplit('_20', 1)[0].replace('_', ' ')
                    else:
                        latest_model_name = dir_name.replace('_', ' ')
    
    return latest_training, latest_model_name


def get_initial_state():
    """Initialize application state, checking for existing training on disk."""
    base_state = {
        'uploaded_file': None,
        'selected_model': STUDIO_MODELS.DEFAULT_STUDIO_MODEL,
        'selected_preset': 'balanced',
        'training_config': {},
        'training_active': False,
        'training_complete': False,
        'output_dir': None,
        'train_losses': {},
        'val_losses': {},
        'best_loss': float('inf'),
        'best_iter': 0,
        'log_lines': [],
        'monitor': None,
        'completion_time': None,
        'model_name': None,
    }
    
    # Check for existing training on disk
    print("[INIT] Checking for existing training sessions...")
    
    def read_actual_model_from_adapter(output_dir):
        """Read the actual base model from adapter_config.json."""
        adapter_config_paths = [
            Path(output_dir) / "adapters" / "adapter_config.json",
            Path(output_dir) / "best_adapter" / "adapter_config.json",
        ]
        
        for config_path in adapter_config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    actual_model = config.get('model')
                    if actual_model:
                        print(f"[INIT] OK: Read actual model from adapter config: {actual_model}")
                        return actual_model
                except Exception as e:
                    print(f"[INIT] Warning: Error reading adapter config: {e}")
        
        return None
    
    def update_state_with_actual_model(state, output_dir):
        """Update state with actual model from adapter config."""
        actual_model = read_actual_model_from_adapter(output_dir)
        if actual_model:
            state['training_config'] = {
                'model_id': actual_model,
                'model_name': actual_model.split('/')[-1].replace('-', ' ').replace('_', ' '),
            }
            # Try to find matching model in STUDIO_MODELS
            for model_id, model in STUDIO_MODELS.DEFAULT_MODELS.items():
                if model.get('model_id') == actual_model:
                    state['selected_model'] = model_id
                    state['training_config']['model_name'] = model.get('name')
                    print(f"[INIT] OK: Matched to known model: {model.get('name')}")
                    break
            else:
                # Model not in our list, but we have the ID
                print(f"[INIT] Warning: Model not in predefined list, using ID: {actual_model}")
        return state
    
    # First try to load from state file
    disk_state = load_state_from_disk()
    if disk_state and disk_state.get("training_complete") and disk_state.get("output_dir"):
        output_dir = disk_state.get('output_dir')
        if Path(output_dir).exists():
            # Validate that training actually succeeded by checking for output files
            has_adapter = (Path(output_dir) / "best_adapter" / "adapters.safetensors").exists() or \
                         (Path(output_dir) / "adapters" / "adapters.safetensors").exists()
            has_fused = (Path(output_dir) / "fused_model").exists()
            has_summary = (Path(output_dir) / "training_summary.json").exists()
            
            if has_adapter or has_fused or has_summary:
                print(f"[INIT] OK: Found completed training in state file: {output_dir}")
                base_state.update({
                    'training_complete': True,
                    'training_active': False,
                    'output_dir': output_dir,
                    'completion_time': disk_state.get('completion_time'),
                    'model_name': disk_state.get('model_name'),
                })
                # Read actual model from adapter
                base_state = update_state_with_actual_model(base_state, output_dir)
                return base_state
            else:
                # Directory exists but no training files - stale state
                print(f"[INIT] Warning: Stale training state detected: {output_dir}")
                print(f"[INIT]   Directory exists but no training files found")
                print(f"[INIT]   Clearing stale state and starting fresh")
                # Clear the stale state
                try:
                    STATE_FILE.unlink(missing_ok=True)
                    print(f"[INIT] OK: Cleared stale state file")
                except Exception as e:
                    print(f"[INIT] Warning: Could not clear state file: {e}")
        else:
            # Directory doesn't exist - stale state
            print(f"[INIT] Warning: Stale training state detected: {output_dir}")
            print(f"[INIT]   Output directory not found")
            print(f"[INIT]   Clearing stale state and starting fresh")
            try:
                STATE_FILE.unlink(missing_ok=True)
                print(f"[INIT] OK: Cleared stale state file")
            except Exception as e:
                print(f"[INIT] Warning: Could not clear state file: {e}")
    
    # If no state file or invalid, try auto-discovery
    discovered_dir, discovered_model = auto_discover_latest_training()
    if discovered_dir:
        print(f"[INIT] OK: Auto-discovered training: {discovered_dir}")
        base_state.update({
            'training_complete': True,
            'training_active': False,
            'output_dir': discovered_dir,
            'model_name': discovered_model,
        })
        # Read actual model from adapter
        base_state = update_state_with_actual_model(base_state, discovered_dir)
        # Save to disk for future reference
        save_state_to_disk(base_state)
    else:
        print("[INIT] No existing training found. Starting fresh.")
    
    return base_state


def create_ui():
    """Create the fully functional EdukaAI Studio UI."""
    
    with gr.Blocks(title="EdukaAI Studio") as app:
        state = gr.State(get_initial_state())
        
        # Header with training selector
        with gr.Row():
            gr.Markdown("# 🎓 EdukaAI Studio")
            with gr.Column(scale=1):
                new_training_btn = gr.Button("🆕 New Training", variant="stop", size="sm")
        
        # Training selector - single row with delete button
        with gr.Row():
            with gr.Column(scale=4):
                training_selector = gr.Dropdown(
                    label="Previous",
                    choices=[],
                    value=None,
                    interactive=True,
                    allow_custom_value=False
                )
            with gr.Column(scale=1):
                delete_training_btn = gr.Button("🗑️", size="sm", visible=False)
        
        # Tabs
        tabs = gr.Tabs()
        with tabs:
            # Tab 1: Upload
            with gr.TabItem("1. Upload"):
                gr.Markdown("## Step 1: Upload Training Data")
                
                file_upload = gr.File(
                    label="Training Data (JSONL or Alpaca JSON)",
                    file_types=[".jsonl", ".json"]
                )
                
                file_status = gr.Textbox(
                    label="Status",
                    value="Please upload a JSONL or Alpaca JSON file",
                    interactive=False
                )
                
                # Data preview component
                data_preview = gr.Dataframe(
                    label="Data Preview (5 samples)",
                    headers=["Instruction", "Input", "Output"],
                    visible=False,
                    interactive=False
                )
                
                preview_type = gr.Radio(
                    label="Preview Mode",
                    choices=["First 5", "Random 5"],
                    value="First 5",
                    visible=False
                )
                
                gr.Markdown("### Supported Formats")
                gr.Markdown("**Format 1: JSONL (one JSON object per line)**")
                gr.Code('{"instruction": "What is offside?", "output": "A player is offside when..."}', language="json")
                gr.Markdown("**Format 2: Alpaca JSON (array of objects)**")
                gr.Code('[{"instruction": "What is offside?", "input": "", "output": "A player is offside when..."}]', language="json")
                
                gr.Markdown("---")
                gr.Markdown("### Next Step")
                gr.Markdown("Once your file is uploaded and validated, click the button below to proceed to configuration.")
                
                def process_file(file_path, preview_mode, current_state):
                    if not file_path:
                        return "No file selected", None, gr.Dataframe(visible=False), gr.Radio(visible=False), current_state, gr.Button(visible=False)
                    
                    try:
                        # Detect file type by extension
                        is_jsonl = file_path.endswith('.jsonl')
                        
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        examples = []
                        
                        if is_jsonl:
                            # Parse JSONL (line by line)
                            lines = [l for l in content.split('\n') if l.strip()]
                            for line in lines:
                                example = json.loads(line)
                                examples.append(example)
                        else:
                            # Parse JSON (array or single object)
                            data = json.loads(content)
                            if isinstance(data, list):
                                examples = data
                            elif isinstance(data, dict):
                                # Single example wrapped in dict
                                examples = [data]
                        
                        count = len(examples)
                        
                        if count == 0:
                            return "Error: No valid examples found", None, gr.Dataframe(visible=False), gr.Radio(visible=False), current_state, gr.Button(visible=False)
                        
                        # Validate format (Alpaca or simple)
                        first = examples[0]
                        has_instruction = 'instruction' in first
                        has_output = 'output' in first
                        has_input = 'input' in first  # Optional Alpaca field
                        
                        if not has_instruction and 'prompt' in first:
                            has_instruction = True  # Alternative field name
                        
                        if not has_output and 'response' in first:
                            has_output = True  # Alternative field name
                        
                        if not (has_instruction and has_output):
                            return "Error: File must have 'instruction' and 'output' fields (or 'prompt'/'response')", None, gr.Dataframe(visible=False), gr.Radio(visible=False), current_state, gr.Button(visible=False)
                        
                        # Extract sample data for preview
                        if preview_mode == "Random 5" and len(examples) > 5:
                            samples = random.sample(examples, 5)
                        else:
                            samples = examples[:5]
                        
                        # Format samples for DataFrame
                        preview_data = []
                        for sample in samples:
                            instruction = sample.get('instruction', sample.get('prompt', ''))
                            input_text = sample.get('input', '')
                            output_text = sample.get('output', sample.get('response', ''))
                            # Truncate long text for display
                            instruction = instruction[:100] + "..." if len(instruction) > 100 else instruction
                            input_text = input_text[:100] + "..." if len(input_text) > 100 else input_text
                            output_text = output_text[:150] + "..." if len(output_text) > 150 else output_text
                            preview_data.append([instruction, input_text, output_text])
                        
                        new_state = {**current_state, 'uploaded_file': file_path}
                        format_type = "Alpaca" if has_input else "standard"
                        success_msg = f"OK: {count} {format_type} examples validated and ready"
                        
                        return success_msg, preview_data, gr.Dataframe(visible=True), gr.Radio(visible=True), new_state, gr.Button(visible=True)
                        
                    except json.JSONDecodeError as e:
                        return f"Error: Invalid JSON format - {str(e)}", None, gr.Dataframe(visible=False), gr.Radio(visible=False), current_state, gr.Button(visible=False)
                    except Exception as e:
                        return f"Error: {str(e)}", None, gr.Dataframe(visible=False), gr.Radio(visible=False), current_state, gr.Button(visible=False)
                
                # Navigation button to Configure tab
                go_to_configure_btn = gr.Button(
                    "→ Go to Configure →",
                    variant="primary",
                    size="lg",
                    visible=False
                )
                
                file_upload.change(
                    fn=process_file,
                    inputs=[file_upload, preview_type, state],
                    outputs=[file_status, data_preview, data_preview, preview_type, state, go_to_configure_btn]
                )
                
                # Handle preview type change to refresh preview
                def refresh_preview(preview_mode, current_state):
                    file_path = current_state.get('uploaded_file')
                    if not file_path:
                        return None, gr.Dataframe(visible=False)
                    
                    try:
                        is_jsonl = file_path.endswith('.jsonl')
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        examples = []
                        if is_jsonl:
                            lines = [l for l in content.split('\n') if l.strip()]
                            for line in lines:
                                examples.append(json.loads(line))
                        else:
                            data = json.loads(content)
                            examples = data if isinstance(data, list) else [data]
                        
                        if preview_mode == "Random 5" and len(examples) > 5:
                            samples = random.sample(examples, 5)
                        else:
                            samples = examples[:5]
                        
                        preview_data = []
                        for sample in samples:
                            instruction = sample.get('instruction', sample.get('prompt', ''))
                            input_text = sample.get('input', '')
                            output_text = sample.get('output', sample.get('response', ''))
                            instruction = instruction[:100] + "..." if len(instruction) > 100 else instruction
                            input_text = input_text[:100] + "..." if len(input_text) > 100 else input_text
                            output_text = output_text[:150] + "..." if len(output_text) > 150 else output_text
                            preview_data.append([instruction, input_text, output_text])
                        
                        return preview_data, gr.Dataframe(visible=True)
                    except:
                        return None, gr.Dataframe(visible=False)
                
                preview_type.change(
                    fn=refresh_preview,
                    inputs=[preview_type, state],
                    outputs=[data_preview, data_preview]
                )
                
                # Handle button click to switch tabs
                def switch_to_configure_tab():
                    """Return the index of the Configure tab (index 1)."""
                    return gr.Tabs(selected=1)
                
                go_to_configure_btn.click(
                    fn=switch_to_configure_tab,
                    outputs=[tabs]
                )
            
            # Tab 2: Configure
            with gr.TabItem("2. Configure") as configure_tab:
                gr.Markdown("## Step 2: Configure Training")
                
                all_models = STUDIO_MODELS.get_all_models()
                model_choices = [(m['name'], m['id']) for m in all_models]
                
                # Load user models and add to choices
                try:
                    from edukaai_studio.ui.hf_model_validator import UserModelsManager
                    user_models_mgr = UserModelsManager()
                    user_models = user_models_mgr.get_all_models()
                    print(f"[CONFIGURE] Loaded {len(user_models)} user models: {[m['model_id'] for m in user_models]}")
                    for um in user_models:
                        model_choices.append((f"{um['name']} (Custom)", um['model_id']))
                except Exception as e:
                    print(f"[CONFIGURE] Error loading user models: {e}")
                    import traceback
                    print(f"[CONFIGURE] {traceback.format_exc()}")
                
                # Model selection with training preset on same line
                with gr.Row():
                    with gr.Column(scale=2):
                        model_dropdown = gr.Dropdown(
                            label="Base Model",
                            choices=model_choices,
                            value=STUDIO_MODELS.DEFAULT_STUDIO_MODEL
                        )
                    
                    with gr.Column(scale=2):
                        preset_dropdown = gr.Dropdown(
                            label="Training Preset",
                            choices=[
                                ("Quick", "quick"),
                                ("Balanced", "balanced"),
                                ("Maximum", "maximum")
                            ],
                            value="balanced"
                        )
                
                gr.Markdown("---")
                gr.Markdown("### Training Parameters")
                gr.Markdown("*Adjust these to fine-tune your training. Presets set good defaults.*")
                
                with gr.Row():
                    with gr.Column():
                        iterations_slider = gr.Slider(
                            minimum=TRAINING.MIN_ITERATIONS, 
                            maximum=TRAINING.MAX_ITERATIONS, 
                            step=TRAINING.ITERATION_STEP, 
                            value=TRAINING.DEFAULT_ITERATIONS,
                            label="Training Steps (Iterations)"
                        )
                    with gr.Column():
                        learning_rate_dropdown = gr.Dropdown(
                            choices=TRAINING.LEARNING_RATE_OPTIONS,
                            value=TRAINING.DEFAULT_LEARNING_RATE,
                            label="Learning Rate"
                        )
                
                with gr.Row():
                    with gr.Column():
                        lora_rank_slider = gr.Slider(
                            minimum=8, 
                            maximum=64, 
                            step=8, 
                            value=TRAINING.DEFAULT_LORA_RANK,
                            label="LoRA Rank"
                        )
                    with gr.Column():
                        lora_alpha_slider = gr.Slider(
                            minimum=16, 
                            maximum=128, 
                            step=16, 
                            value=TRAINING.DEFAULT_LORA_ALPHA,
                            label="LoRA Alpha"
                        )
                
                with gr.Row():
                    with gr.Column():
                        max_seq_length_slider = gr.Slider(
                            minimum=512, maximum=4096, step=256, value=2048,
                            label="Max Sequence Length"
                        )
                    with gr.Column():
                        early_stopping_slider = gr.Slider(
                            minimum=1, maximum=5, step=1, 
                            value=TRAINING.DEFAULT_EARLY_STOPPING_PATIENCE,
                            label="Early Stopping Patience"
                        )
                
                with gr.Row():
                    with gr.Column():
                        validation_split_slider = gr.Slider(
                            minimum=TRAINING.MIN_VALIDATION_SPLIT_PCT, 
                            maximum=TRAINING.MAX_VALIDATION_SPLIT_PCT, 
                            step=5, 
                            value=TRAINING.DEFAULT_VALIDATION_SPLIT_PCT,
                            label="Validation Split (%)"
                        )
                    with gr.Column():
                        grad_accum_slider = gr.Slider(
                            minimum=4, maximum=128, step=4, 
                            value=TRAINING.DEFAULT_GRAD_ACCUMULATION,
                            label="Gradient Accumulation Steps"
                        )
                
                status_text = gr.Textbox(
                    label="Configuration Status",
                    value="",
                    interactive=False
                )
                
                gr.Markdown("---")
                gr.Markdown("### Next Step")
                gr.Markdown("Click 'Save Configuration' to confirm your settings, then proceed to training.")
                
                # Preset selection updates all parameter sliders
                def update_params_from_preset(preset_name):
                    """Update all parameter sliders when preset changes."""
                    from edukaai_studio.config import TrainingPresets
                    
                    preset_map = {
                        "quick": TrainingPresets.QUICK,
                        "balanced": TrainingPresets.BALANCED,
                        "maximum": TrainingPresets.MAXIMUM
                    }
                    
                    preset = preset_map.get(preset_name, TrainingPresets.BALANCED)
                    
                    return [
                        preset.get("iterations", 600),
                        preset.get("learning_rate", "1e-4"),
                        preset.get("lora_rank", 16),
                        preset.get("lora_alpha", 32),
                        preset.get("grad_accumulation", 32),
                        preset.get("early_stopping", 2),
                    ]
                
                preset_dropdown.change(
                    fn=update_params_from_preset,
                    inputs=[preset_dropdown],
                    outputs=[iterations_slider, learning_rate_dropdown, lora_rank_slider,
                            lora_alpha_slider, grad_accum_slider, early_stopping_slider]
                )
                
                def configure_training(model_id, preset_name, iterations, learning_rate, lora_rank, 
                                      lora_alpha, max_seq_length, early_stopping, validation_split,
                                      grad_accumulation, current_state):
                    if not current_state.get('uploaded_file'):
                        return "Warning: Upload training data first", current_state, gr.Button(visible=False)
                    
                    # Try to get model from predefined list first
                    model = STUDIO_MODELS.get_model(model_id)
                    
                    # If not found, check user models (custom models)
                    if not model:
                        try:
                            from edukaai_studio.ui.hf_model_validator import UserModelsManager
                            mgr = UserModelsManager()
                            user_models = mgr.get_all_models()
                            for um in user_models:
                                if um['model_id'] == model_id:
                                    # Convert user model format to match STUDIO_MODELS format
                                    model = {
                                        'id': um['model_id'],
                                        'name': um['name'],
                                        'model_id': um['model_id'],
                                        'size_gb': um.get('hf_data', {}).get('size_gb', 0),
                                        'mlx_compatible': um.get('hf_data', {}).get('mlx_compatible', True),
                                    }
                                    print(f"[CONFIGURE] Found custom model: {um['name']}")
                                    break
                        except Exception as e:
                            print(f"[CONFIGURE] Error checking user models: {e}")
                    
                    if not model:
                        return f"Error: Invalid model selected ({model_id})", current_state
                    
                    # Build training config with user-selected parameters
                    new_state = {
                        **current_state,
                        'training_config': {
                            'model_id': model.get('model_id'),
                            'model_name': model.get('name'),
                            'iterations': iterations,
                            'learning_rate': learning_rate,
                            'lora_rank': lora_rank,
                            'lora_alpha': lora_alpha,
                            'grad_accumulation': grad_accumulation,
                            'max_seq_length': max_seq_length,
                            'batch_size': 1,
                            'early_stopping': early_stopping,
                            'validation_split': validation_split,
                        }
                    }
                    
                    msg = f"OK: Configured: {model.get('name')} ({iterations} steps, rank={lora_rank}, lr={learning_rate})"
                    return msg, new_state
                
                save_config_btn = gr.Button("Save Configuration", variant="primary")
                
                save_config_btn.click(
                    fn=configure_training,
                    inputs=[model_dropdown, preset_dropdown, iterations_slider, learning_rate_dropdown,
                           lora_rank_slider, lora_alpha_slider, max_seq_length_slider,
                           early_stopping_slider, validation_split_slider, grad_accum_slider, state],
                           outputs=[status_text, state]
                )
                
                # Refresh model list when Configure tab is selected
                def refresh_configure_models():
                    """Reload predefined and user models for the dropdown."""
                    try:
                        all_models = STUDIO_MODELS.get_all_models()
                        model_choices = [(m['name'], m['id']) for m in all_models]
                        
                        # Load user models and add to choices
                        from edukaai_studio.ui.hf_model_validator import UserModelsManager
                        user_models_mgr = UserModelsManager()
                        user_models = user_models_mgr.get_all_models()
                        print(f"[CONFIGURE REFRESH] Found {len(user_models)} user models")
                        for um in user_models:
                            model_choices.append((f"{um['name']} (Custom)", um['model_id']))
                        
                        return gr.update(choices=model_choices)
                    except Exception as e:
                        print(f"[CONFIGURE REFRESH] Error: {e}")
                        return gr.update()
                
                configure_tab.select(
                    fn=refresh_configure_models,
                    outputs=[model_dropdown]
                )
            
            # Tab 3: Train
            with gr.TabItem("3. Train"):
                gr.Markdown("## Step 3: Training")
                gr.Markdown("**Real training with MLX will take time. Do not close the browser.**")
                
                # Two-column layout: Progress/metrics on left, plot on right
                with gr.Row():
                    # Left column: Progress and metrics
                    with gr.Column(scale=1):
                        progress_bar = gr.Slider(
                            minimum=0, maximum=100, value=0,
                            label="Progress", interactive=False
                        )
                        step_display = gr.Textbox(
                            label="Step", value="0 / 0", interactive=False
                        )
                        
                        # Metrics row
                        with gr.Row():
                            train_loss = gr.Textbox(
                                label="Training Loss", value="-", interactive=False
                            )
                            val_loss = gr.Textbox(
                                label="Validation Loss", value="-", interactive=False
                            )
                            best_loss = gr.Textbox(
                                label="Best Loss", value="-", interactive=False
                            )
                            memory_display = gr.Textbox(
                                label="Memory (GB)", value="-", interactive=False
                            )
                        
                        # Resource metrics row (CPU/RAM from logs)
                        with gr.Row():
                            cpu_display = gr.Textbox(
                                label="CPU %", value="-", interactive=False
                            )
                            ram_display = gr.Textbox(
                                label="RAM %", value="-", interactive=False
                            )
                        
                        # Small buttons row below metrics
                        with gr.Row():
                            start_btn = gr.Button(
                                "Start Training", 
                                variant="primary", 
                                size="sm",
                                min_width=100
                            )
                            stop_btn = gr.Button(
                                "Stop Training", 
                                variant="stop",
                                size="sm",
                                min_width=100
                            )
                    
                    # Right column: Loss curve plot
                    with gr.Column(scale=1):
                        loss_plot = gr.Plot(
                            label="Loss Curve",
                            value=None
                        )
                
                # Training log - full width below
                gr.Markdown("### Training Log")
                log_display = gr.Textbox(
                    label="Log",
                    lines=15,
                    max_lines=100,
                    interactive=False,
                    value="Ready to start training..."
                )
                
                train_status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )
                
                def create_loss_plot(train_losses, val_losses, total_iters=None):
                    """Create loss curve plot from training data."""
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
                        
                        # Return the figure object directly (Gradio expects this)
                        # Note: Gradio will handle closing the figure after displaying it
                        return fig
                    except ImportError:
                        # matplotlib not available
                        return None
                    except Exception as e:
                        print(f"[PLOT] Error creating plot: {e}")
                        # Clean up on error
                        try:
                            import matplotlib.pyplot as plt
                            plt.close('all')
                        except:
                            pass
                        return None
                
                def start_training_real(current_state):
                    """Start REAL training using TrainingMonitor."""
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
                    last_logged_iter = 0  # Track last logged iteration to prevent duplicates
                    
                    yield [
                        0, f"0 / {total_iters}", "-", "-", "-", "-", "-", "-",
                        "\n".join(log_lines[-50:]),
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
                                'hf_token': hf_token,  # Pass HF token for gated models
                            }
                            
                            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training configuration:")
                            log_lines.append(f"  Model: {config.get('model_name', 'Unknown')}")
                            log_lines.append(f"  Base model ID: {config.get('model_id', 'unknown')}")
                            log_lines.append(f"  Iterations: {total_iters}")
                            log_lines.append(f"  Learning rate: {config.get('learning_rate', '1e-4')}")
                            log_lines.append(f"  LoRA rank: {config.get('lora_rank', 16)}")
                            log_lines.append(f"  LoRA alpha: {config.get('lora_alpha', 32)}")
                            log_lines.append(f"  Max seq length: {config.get('max_seq_length', 2048)}")
                            log_lines.append(f"  Early stopping: {config.get('early_stopping', 2)} checks")
                            log_lines.append(f"  Validation split: {config.get('validation_split', 10)}%")
                            log_lines.append(f"  Gradient accumulation: {config.get('grad_accumulation', 32)} steps")
                            log_lines.append(f"  Data file: {data_file}")
                            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting training subprocess...")
                            
                            yield [
                                0, f"0 / {total_iters}", "-", "-", "-", "-", "-", "-",
                                "\n".join(log_lines[-50:]),
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
                                    "\n".join(log_lines[-50:]),
                                    None,
                                    "Error: Failed to start training",
                                    current_state
                                ]
                                return
                            
                            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training started successfully")
                            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring training progress...")
                            log_lines.append("")  # Empty line for readability
                            
                            # Monitor training progress
                            last_log_time = time.time()
                            progress_data = None
                            while not monitor.is_complete():
                                if not current_state.get('training_active', False):
                                    log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Stopping training...")
                                    monitor.stop()
                                    break
                                
                                # Get progress updates from queue
                                try:
                                    progress_data = progress_queue.get(timeout=0.5)
                                except queue.Empty:
                                    progress_data = None
                                
                                if progress_data:
                                    current_iter = progress_data.get('iteration', 0)
                                    total = progress_data.get('total', total_iters)
                                    train_losses = progress_data.get('train_losses', {})
                                    val_losses = progress_data.get('val_losses', {})
                                    best_l = progress_data.get('best_loss', float('inf'))
                                    best_iter = progress_data.get('best_iter', 0)
                                    peak_memory_gb = progress_data.get('peak_memory_gb', 0.0)
                                    
                                    # Use pre-calculated progress percent if available
                                    progress_percent = progress_data.get('progress_percent', 
                                        int((current_iter / total) * 100) if total > 0 else 0)
                                    
                                    # Get latest losses
                                    train_l = train_losses.get(current_iter, 0.0) if train_losses else 0.0
                                    val_l = val_losses.get(current_iter, 0.0) if val_losses else 0.0
                                    
                                    # Format validation loss separately
                                    val_loss_str = f"{val_l:.4f}" if val_l > 0 else "N/A"
                                    
                                    # Get resource stats if available
                                    resource_stats = progress_data.get('resource_stats')
                                    resource_str = ""
                                    if resource_stats:
                                        cpu = resource_stats.get('cpu', 0)
                                        ram_pct = resource_stats.get('ram_percent', 0)
                                        resource_str = f" | CPU: {cpu:.1f}% RAM: {ram_pct:.1f}%"
                                    
                                    # Only log when iteration changes (avoid duplicates)
                                    if current_iter > 0 and current_iter != last_logged_iter:
                                        last_logged_iter = current_iter
                                        
                                        # Log every iteration with detailed info
                                        log_lines.append(
                                            f"[{datetime.now().strftime('%H:%M:%S')}] "
                                            f"Iter {current_iter}/{total} | "
                                            f"Progress: {progress_percent}% | "
                                            f"Train Loss: {train_l:.4f} | "
                                            f"Val Loss: {val_loss_str} | "
                                            f"Best: {best_l:.4f} (iter {best_iter}) | "
                                            f"Memory: {peak_memory_gb:.2f}GB"
                                            f"{resource_str}"
                                        )
                                    
                                    # Update state (immutable pattern)
                                    current_state = {
                                        **current_state,
                                        'train_losses': train_losses,
                                        'val_losses': val_losses,
                                        'best_loss': best_l,
                                        'best_iter': best_iter,
                                        'peak_memory_gb': peak_memory_gb,
                                    }
                                    
                                    # Yield updates every 10 iterations
                                    if current_iter % 10 == 0:
                                        # Create loss plot with full iteration range
                                        plot_buf = create_loss_plot(train_losses, val_losses, total)
                                        
                                        # Format memory display
                                        memory_str = f"{peak_memory_gb:.2f}" if peak_memory_gb > 0 else "-"
                                        
                                        # Get resource stats for display
                                        cpu_str = "-"
                                        ram_str = "-"
                                        if resource_stats:
                                            cpu_str = f"{resource_stats.get('cpu', 0):.1f}"
                                            ram_str = f"{resource_stats.get('ram_percent', 0):.1f}"
                                        
                                        yield [
                                            progress_percent,
                                            f"{current_iter} / {total}",
                                            f"{train_l:.4f}",
                                            f"{val_l:.4f}",
                                            f"{best_l:.4f}",
                                            memory_str,
                                            cpu_str,
                                            ram_str,
                                            "\n".join(log_lines[-50:]),
                                            plot_buf,
                                            f"Training... {progress_percent}%",
                                            current_state
                                        ]
                                
                                # Also get any log output
                                try:
                                    while True:
                                        log_line = output_queue.get_nowait()
                                        if log_line.strip():
                                            log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_line.strip()}")
                                except queue.Empty:
                                    pass
                                
                                # Small delay to avoid overwhelming UI
                                time.sleep(0.1)
                            
                            # Debug: Log state when loop exits
                            print(f"[DEBUG] Loop exited - monitor.is_complete()={monitor.is_complete()}, training_active={current_state.get('training_active')}")
                            print(f"[DEBUG] Loop iteration count: completed monitoring loop")
                            print(f"[DEBUG] About to check completion condition...")
                            
                            # BACKUP: Force save completion state if monitor reports complete
                            # This handles cases where Gradio state doesn't persist properly
                            # BUT only if training wasn't stopped by user
                            if monitor.is_complete() and not monitor.is_stopped():
                                print(f"[DEBUG] Monitor reports complete (not stopped) - forcing state save as backup")
                                backup_state = {
                                    'training_complete': True,
                                    'training_active': False,
                                    'output_dir': current_state.get('output_dir'),
                                    'completion_time': datetime.now().strftime('%H:%M:%S'),
                                }
                                try:
                                    with open(STATE_FILE, 'w') as f:
                                        json.dump(backup_state, f, indent=2)
                                    print(f"[DEBUG] Backup state saved: {backup_state}")
                                except Exception as e:
                                    print(f"[DEBUG] Backup save failed: {e}")
                            elif monitor.is_stopped():
                                print(f"[DEBUG] Training was stopped - not saving backup completion state")
                            
                            # Training complete or stopped
                            # Check if training was stopped by user (takes precedence)
                            was_stopped = hasattr(monitor, 'is_stopped') and monitor.is_stopped()
                            completion_check = monitor.is_complete() and current_state.get('training_active', False) and not was_stopped
                            print(f"[DEBUG] Completion check: monitor.is_complete()={monitor.is_complete()} AND training_active={current_state.get('training_active')} AND not_stopped={not was_stopped} = {completion_check}")
                            
                            if was_stopped:
                                # Training was stopped by user - show stopped message
                                completion_time = datetime.now().strftime('%H:%M:%S')
                                log_lines.append(f"[{completion_time}] Training stopped by user")
                                print(f"[TRAINING STOPPED] User stopped training at {completion_time}")
                                
                                # Get final values from state
                                train_losses = current_state.get('train_losses', {})
                                val_losses = current_state.get('val_losses', {})
                                final_train_loss = list(train_losses.values())[-1] if train_losses else 0.0
                                final_val_loss = list(val_losses.values())[-1] if val_losses else 0.0
                                best_loss = current_state.get('best_loss', float('inf'))
                                peak_memory_gb = current_state.get('peak_memory_gb', 0.0)
                                
                                # Create final loss plot
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
                                    "\n".join(log_lines[-50:]),
                                    final_plot,
                                    "Error: Training stopped by user",
                                    current_state
                                ]
                                return
                            elif completion_check:
                                completion_time = datetime.now().strftime('%H:%M:%S')
                                log_lines.append(f"[{completion_time}] Training complete!")
                                
                                # Add resource summary if available
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
                                
                                print(f"[TRAINING COMPLETE] Entering completion block at {completion_time}")
                                print(f"[TRAINING COMPLETE] Before: training_active={current_state.get('training_active')}, training_complete={current_state.get('training_complete')}")
                                
                                # Try to parse actual output directory from training logs
                                # Look for pattern: "Output: outputs/..." or "Output Directory: outputs/..."
                                actual_output_dir = None
                                
                                print(f"[TRAINING COMPLETE] Searching through {len(log_lines)} log lines for output directory...")
                                
                                for i, line in enumerate(reversed(log_lines)):
                                    # Debug: print first few lines being checked
                                    if i < 10:
                                        print(f"[TRAINING COMPLETE] Checking line: {line[:100]}")
                                    
                                    # Match "Output: outputs/model_name/..." format - extract base dir
                                    # Pattern: "Output: outputs/anything_without_spaces_or_slashes"
                                    match = re.search(r'Output:\s*(outputs/[^\s]+)', line)
                                    if match:
                                        full_path = match.group(1)
                                        # Extract just the directory part (before any /)
                                        parts = full_path.split('/')
                                        if len(parts) >= 2:
                                            actual_output_dir = '/'.join(parts[:2])  # outputs/directory_name
                                        else:
                                            actual_output_dir = full_path
                                        print(f"[TRAINING COMPLETE] OK: FOUND! Parsed output dir: {actual_output_dir} (from: {match.group(0)})")
                                        break
                                    
                                    # Also try "Output Directory:" format
                                    match2 = re.search(r'Output Directory:\s*(outputs/[^\s]+)', line)
                                    if match2:
                                        full_path = match2.group(1)
                                        parts = full_path.split('/')
                                        if len(parts) >= 2:
                                            actual_output_dir = '/'.join(parts[:2])
                                        else:
                                            actual_output_dir = full_path
                                        print(f"[TRAINING COMPLETE] OK: FOUND! Parsed output dir from 'Output Directory:': {actual_output_dir}")
                                        break
                                
                                # Fallback if not found
                                if not actual_output_dir:
                                    print(f"[TRAINING COMPLETE] Warning: Could not parse output dir from logs, using fallback")
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    model_name = config.get('model_name', 'model').replace(' ', '_')
                                    actual_output_dir = f"outputs/{model_name}_{timestamp}_fused"
                                    print(f"[TRAINING COMPLETE] Fallback output dir: {actual_output_dir}")
                                
                                # Create final state (immutable pattern)
                                current_state = {
                                    **current_state,
                                    'training_active': False,
                                    'training_complete': True,
                                    'completion_time': completion_time,
                                    'output_dir': actual_output_dir,
                                }
                                
                                print(f"[TRAINING COMPLETE] About to save state to disk...")
                                print(f"[TRAINING COMPLETE] State before save: {current_state}")
                                
                                # Save to disk for persistence across tabs
                                save_state_to_disk(current_state)
                                
                                print(f"[TRAINING COMPLETE] State saved (or attempted)")
                                print(f"[TRAINING COMPLETE] output_dir set to: {actual_output_dir}")
                                print(f"[TRAINING COMPLETE] State: training_active={current_state.get('training_active')}, training_complete={current_state.get('training_complete')}")
                                
                                # Get final loss values from state
                                train_losses = current_state.get('train_losses', {})
                                val_losses = current_state.get('val_losses', {})
                                final_train_loss = list(train_losses.values())[-1] if train_losses else 0.0
                                final_val_loss = list(val_losses.values())[-1] if val_losses else 0.0
                                best_loss = current_state.get('best_loss', float('inf'))
                                peak_memory_gb = current_state.get('peak_memory_gb', 0.0)
                                
                                # Create final loss plot with full iteration range
                                final_plot = create_loss_plot(train_losses, val_losses, total_iters)
                                
                                # Format final memory
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
                                    "\n".join(log_lines[-50:]),
                                    final_plot,
                                    "OK: Training complete! Go to Results tab.",
                                    current_state
                                ]
                                # Explicitly return final state to ensure it's saved
                                return
                            else:
                                print(f"[DEBUG] ELSE BLOCK - monitor.is_complete()={monitor.is_complete()}, training_active={current_state.get('training_active')}")
                                yield [
                                    0, "0 / 0", "-", "-", "-", "-", "-", "-",
                                    "\n".join(log_lines[-50:]),
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
                                "\n".join(log_lines[-50:]),
                                None,
                                f"Error: {str(e)}",
                                current_state
                            ]
                    else:
                        # TrainingMonitor not available - show error
                        log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: TrainingMonitor not available")
                        log_lines.append("Please ensure ui.training_monitor is properly installed")
                        yield [
                            0, "0 / 0", "-", "-", "-", "-", "-", "-",
                            "\n".join(log_lines[-50:]),
                            None,
                            "Error: Training infrastructure not available",
                            current_state
                        ]
                
                def stop_training_handler(current_state):
                    """Stop the training process."""
                    # Get the monitor from state and stop it
                    monitor = current_state.get('monitor')
                    if monitor:
                        try:
                            monitor.stop()
                            print("[STOP] Training monitor stopped")
                        except Exception as e:
                            print(f"[STOP] Error stopping monitor: {e}")
                    
                    new_state = {**current_state, 'training_active': False}
                    return "Training stop requested", new_state
                
                start_btn.click(
                    fn=start_training_real,
                    inputs=[state],
                    outputs=[
                        progress_bar, step_display, train_loss, val_loss, best_loss, memory_display,
                        cpu_display, ram_display, log_display, loss_plot, train_status, state
                    ]
                )
                
                stop_btn.click(
                    fn=stop_training_handler,
                    inputs=[state],
                    outputs=[train_status, state]
                )
            
            # Tab 4: Results
            with gr.TabItem("4. Results") as results_tab:
                gr.Markdown("## Step 4: Training Results")
                
                results_status = gr.Textbox(
                    label="Status",
                    value="Complete training to see download options",
                    interactive=False
                )
                
                with gr.Row():
                    download_status = gr.Textbox(
                        label="Download Status",
                        value="",
                        interactive=False
                    )
                
                # File download components (initially hidden/empty)
                adapter_file = gr.File(
                    label="LoRA Adapter Download",
                    visible=False,
                    interactive=False
                )
                fused_file = gr.File(
                    label="Fused Model Download",
                    visible=False,
                    interactive=False
                )
                gguf_file = gr.File(
                    label="GGUF Model Download",
                    visible=False,
                    interactive=False
                )
                
                with gr.Row():
                    adapter_btn = gr.Button("📥 Download LoRA Adapter", variant="secondary")
                    fused_btn = gr.Button("📥 Download Fused Model", variant="secondary")
                
                with gr.Row():
                    gguf_btn = gr.Button("📥 Download GGUF Model", variant="secondary")
                
                refresh_results_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
                
                def refresh_results_status(current_state):
                    """Check current training status and update display."""
                    # First check disk-based state (fallback if Gradio state lost)
                    disk_state = load_state_from_disk()
                    
                    # Merge disk state with current state (disk takes precedence for completion)
                    if disk_state.get('training_complete'):
                        current_state = {
                            **current_state,
                            'training_complete': True,
                            'training_active': disk_state.get('training_active', False),
                            'output_dir': disk_state.get('output_dir', current_state.get('output_dir')),
                            'completion_time': disk_state.get('completion_time'),
                        }
                        print(f"[REFRESH] Loaded from disk: training_complete=True, output_dir={disk_state.get('output_dir')}")
                    
                    # Debug logging
                    print(f"[REFRESH] Final state: training_complete={current_state.get('training_complete')}, training_active={current_state.get('training_active')}, output_dir={current_state.get('output_dir', 'None')}")
                    
                    if current_state.get('training_complete'):
                        output_dir = current_state.get('output_dir', '')
                        
                        # If output_dir doesn't exist or is empty, try to find the actual directory
                        if not output_dir or not Path(output_dir).exists():
                            print(f"[REFRESH] Output dir from state not found: {output_dir}")
                            print(f"[REFRESH] Searching for actual output directory in outputs/...")
                            
                            # Look for directories with adapters and fused_model
                            outputs_base = Path("outputs")
                            if outputs_base.exists():
                                for subdir in outputs_base.iterdir():
                                    if subdir.is_dir():
                                        # Check if this directory has the expected structure
                                        has_adapters = (subdir / "adapters" / "adapters.safetensors").exists() or \
                                                      (subdir / "best_adapter" / "adapters.safetensors").exists()
                                        has_fused = (subdir / "fused_model").exists()
                                        
                                        if has_adapters and has_fused:
                                            output_dir = str(subdir)
                                            print(f"[REFRESH] OK: Found valid output directory: {output_dir}")
                                            # Update the state with the correct path
                                            current_state['output_dir'] = output_dir
                                            save_state_to_disk(current_state)
                                            break
                        
                        if output_dir and Path(output_dir).exists():
                            # Check if files actually exist
                            adapter_exists = (Path(output_dir) / "adapters" / "adapters.safetensors").exists() or \
                                           (Path(output_dir) / "best_adapter" / "adapters.safetensors").exists()
                            fused_exists = (Path(output_dir) / "fused_model").exists()
                            gguf_exists = len(list(Path(output_dir).rglob("*.gguf"))) > 0
                            
                            status_parts = []
                            if adapter_exists:
                                status_parts.append("Adapter ✓")
                            if fused_exists:
                                status_parts.append("Fused Model ✓")
                            if gguf_exists:
                                status_parts.append("GGUF ✓")
                            
                            if status_parts:
                                files_status = ", ".join(status_parts)
                                return f"OK: Training complete! {files_status} | Output: {output_dir}"
                            else:
                                return f"Warning: Training marked complete but files not found at: {output_dir}"
                        else:
                            return "OK: Training complete! (Output directory not found - try refreshing)"
                    elif current_state.get('training_active'):
                        return "Loading: Training in progress... Check Train tab for details."
                    else:
                        return "Warning: No training completed. Go to Train tab to start training."
                
                refresh_results_btn.click(
                    fn=refresh_results_status,
                    inputs=[state],
                    outputs=[results_status]
                )
                
                def download_adapter(current_state):
                    """Download LoRA adapter file."""
                    # Check disk state for training completion
                    disk_state = load_state_from_disk()
                    is_complete = current_state.get("training_complete") or (disk_state.get("training_complete") if disk_state else False)
                    output_dir = current_state.get('output_dir') or disk_state.get('output_dir')
                    
                    if not is_complete:
                        return None, "Error: Training not complete", gr.File(visible=False)
                    
                    if not output_dir:
                        return None, "Error: Output directory not found", gr.File(visible=False)
                    
                    # Check multiple possible locations for the adapter file
                    possible_paths = [
                        Path(output_dir) / "adapters" / "adapters.safetensors",
                        Path(output_dir) / "best_adapter" / "adapters.safetensors",
                        Path(output_dir) / "adapters.safetensors",  # Legacy location
                    ]
                    
                    for adapter_path in possible_paths:
                        if adapter_path.exists():
                            print(f"[DOWNLOAD] Found adapter at: {adapter_path}")
                            return str(adapter_path), f"OK: Click the file above to download adapter ({adapter_path.stat().st_size / 1024 / 1024:.1f} MB)", gr.File(visible=True, label=f"Package: Adapter: {adapter_path.name}")
                    
                    # If not found, try to auto-discover
                    print(f"[DOWNLOAD] Adapter not found at expected paths, searching...")
                    outputs_base = Path("outputs")
                    if outputs_base.exists():
                        for subdir in outputs_base.iterdir():
                            if subdir.is_dir():
                                for possible_path in [
                                    subdir / "adapters" / "adapters.safetensors",
                                    subdir / "best_adapter" / "adapters.safetensors",
                                ]:
                                    if possible_path.exists():
                                        print(f"[DOWNLOAD] OK: Auto-discovered adapter at: {possible_path}")
                                        # Update state with correct path
                                        current_state['output_dir'] = str(subdir)
                                        save_state_to_disk(current_state)
                                        return str(possible_path), f"OK: Click the file above to download adapter ({possible_path.stat().st_size / 1024 / 1024:.1f} MB)", gr.File(visible=True, label=f"Package: Adapter: {possible_path.name}")
                    
                    return None, f"Error: Adapter not found. Searched: {[str(p) for p in possible_paths]}", gr.File(visible=False)
                
                def download_fused(current_state):
                    """Download fused model directory as zip."""
                    # Check disk state for training completion
                    disk_state = load_state_from_disk()
                    is_complete = current_state.get("training_complete") or (disk_state.get("training_complete") if disk_state else False)
                    output_dir = current_state.get('output_dir') or disk_state.get('output_dir')
                    
                    if not is_complete:
                        return None, "Error: Training not complete", gr.File(visible=False)
                    
                    if not output_dir:
                        return None, "Error: Output directory not found", gr.File(visible=False)
                    
                    fused_path = Path(output_dir) / "fused_model"
                    if fused_path.exists():
                        # Create a zip file of the fused model
                        import shutil
                        zip_path = Path(output_dir) / "fused_model.zip"
                        try:
                            if zip_path.exists():
                                zip_path.unlink()
                            shutil.make_archive(str(zip_path.with_suffix('')), 'zip', fused_path)
                            print(f"[DOWNLOAD] Created zip archive: {zip_path}")
                            file_size_mb = zip_path.stat().st_size / 1024 / 1024
                            return str(zip_path), f"OK: Click the file above to download fused model ({file_size_mb:.1f} MB)", gr.File(visible=True, label=f"Package: Fused Model: {zip_path.name} ({file_size_mb:.1f} MB)")
                        except Exception as e:
                            print(f"[DOWNLOAD] Error creating zip: {e}")
                            return None, f"Error creating zip archive: {str(e)}", gr.File(visible=False)
                    
                    # If not found, try to auto-discover
                    print(f"[DOWNLOAD] Fused model not found at {fused_path}, searching...")
                    outputs_base = Path("outputs")
                    if outputs_base.exists():
                        for subdir in outputs_base.iterdir():
                            if subdir.is_dir():
                                possible_path = subdir / "fused_model"
                                if possible_path.exists():
                                    print(f"[DOWNLOAD] OK: Auto-discovered fused model at: {possible_path}")
                                    # Update state with correct path
                                    current_state['output_dir'] = str(subdir)
                                    save_state_to_disk(current_state)
                                    # Create zip
                                    import shutil
                                    zip_path = subdir / "fused_model.zip"
                                    try:
                                        if zip_path.exists():
                                            zip_path.unlink()
                                        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', possible_path)
                                        file_size_mb = zip_path.stat().st_size / 1024 / 1024
                                        return str(zip_path), f"OK: Click the file above to download fused model ({file_size_mb:.1f} MB)", gr.File(visible=True, label=f"Package: Fused Model: {zip_path.name} ({file_size_mb:.1f} MB)")
                                    except Exception as e:
                                        return None, f"Error creating zip: {str(e)}", gr.File(visible=False)
                    
                    return None, f"Error: Fused model not found at: {fused_path}", gr.File(visible=False)
                
                def download_gguf(current_state):
                    """Download or create GGUF model file on-demand for llama.cpp compatibility."""
                    # Check disk state for training completion
                    disk_state = load_state_from_disk()
                    is_complete = current_state.get("training_complete") or (disk_state.get("training_complete") if disk_state else False)
                    output_dir = current_state.get('output_dir') or disk_state.get('output_dir')
                    
                    if not is_complete:
                        return None, "Error: Training not complete", gr.File(visible=False)
                    
                    if not output_dir:
                        return None, "Error: Output directory not found", gr.File(visible=False)
                    
                    output_path = Path(output_dir)
                    
                    # First, check if GGUF already exists
                    print(f"[GGUF] Checking for existing GGUF files in {output_path}...")
                    
                    # Search for existing GGUF files
                    if output_path.exists():
                        for gguf_file in output_path.rglob("*.gguf"):
                            print(f"[GGUF] OK: Found existing GGUF file: {gguf_file}")
                            file_size_mb = gguf_file.stat().st_size / 1024 / 1024
                            return str(gguf_file), f"OK: GGUF model ready ({file_size_mb:.1f} MB) - click above to download", gr.File(visible=True, label=f"Package: GGUF Model: {gguf_file.name} ({file_size_mb:.1f} MB)")
                    
                    # Check if fused_model exists to convert
                    fused_path = output_path / "fused_model"
                    if not fused_path.exists():
                        return None, "Error: Fused model not found. Cannot convert to GGUF.", gr.File(visible=False)
                    
                    # Need to create GGUF on-demand
                    print(f"[GGUF] No existing GGUF found. Converting fused model on-demand...")
                    print(f"[GGUF] Fused model location: {fused_path}")
                    
                    # Check for llama.cpp conversion tools
                    conversion_tools = [
                        "convert_hf_to_gguf.py",
                        "llama.cpp/convert_hf_to_gguf.py",
                        "/Users/developer/finetunning/football-lora/llama.cpp/convert_hf_to_gguf.py",
                        "convert.py",
                    ]
                    
                    convert_script = None
                    for tool in conversion_tools:
                        if Path(tool).exists():
                            convert_script = tool
                            break
                        # Also check in PATH
                        try:
                            result = subprocess.run(["which", tool.replace('.py', '')], capture_output=True, text=True)
                            if result.returncode == 0:
                                convert_script = result.stdout.strip()
                                break
                        except:
                            pass
                    
                    if not convert_script:
                        # Try to find any convert script
                        try:
                            for root, dirs, files in os.walk("/Users/developer/finetunning/football-lora"):
                                for file in files:
                                    if "convert" in file.lower() and file.endswith(".py"):
                                        convert_script = os.path.join(root, file)
                                        print(f"[GGUF] Found potential converter: {convert_script}")
                                        break
                                if convert_script:
                                    break
                        except Exception as e:
                            print(f"[GGUF] Error searching for converter: {e}")
                    
                    if not convert_script:
                        print(f"[GGUF] No conversion tool found")
                        return None, """Error: GGUF conversion tool not found. 

To enable GGUF conversion, install llama.cpp:
  git clone https://github.com/ggerganov/llama.cpp.git
  cd llama.cpp && make

Then run the converter manually:
  python llama.cpp/convert_hf_to_gguf.py --model outputs/your_model/fused_model

Or download the fused model as a zip and convert it locally.""", gr.File(visible=False)
                    
                    # Perform conversion
                    gguf_output = output_path / "model.gguf"
                    
                    try:
                        print(f"[GGUF] Starting conversion using: {convert_script}")
                        print(f"[GGUF] Output will be: {gguf_output}")
                        
                        # Build command
                        if "convert_hf_to_gguf" in convert_script:
                            cmd = [
                                sys.executable, convert_script,
                                "--model", str(fused_path),
                                "--outfile", str(gguf_output),
                                "--outtype", "q4_k_m",  # 4-bit quantization, good balance
                            ]
                        else:
                            # Generic conversion
                            cmd = [
                                sys.executable, convert_script,
                                str(fused_path),
                                str(gguf_output),
                            ]
                        
                        print(f"[GGUF] Running command: {' '.join(cmd)}")
                        
                        # Run conversion with progress tracking
                        import subprocess
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            cwd="/Users/developer/finetunning/football-lora"
                        )
                        
                        stdout, stderr = process.communicate(timeout=600)  # 10 minute timeout
                        
                        if process.returncode != 0:
                            print(f"[GGUF] Conversion failed with return code: {process.returncode}")
                            print(f"[GGUF] stderr: {stderr}")
                            return None, f"Error: GGUF conversion failed:\n{stderr[:500]}", gr.File(visible=False)
                        
                        # Check if output was created
                        if gguf_output.exists():
                            file_size_mb = gguf_output.stat().st_size / 1024 / 1024
                            print(f"[GGUF] OK: Conversion successful! File size: {file_size_mb:.1f} MB")
                            return str(gguf_output), f"OK: GGUF model created on-demand ({file_size_mb:.1f} MB) - click above to download", gr.File(visible=True, label=f"Package: GGUF Model: {gguf_output.name} ({file_size_mb:.1f} MB)")
                        else:
                            # Try to find the created file
                            for gguf_file in output_path.rglob("*.gguf"):
                                file_size_mb = gguf_file.stat().st_size / 1024 / 1024
                                print(f"[GGUF] OK: Found created GGUF: {gguf_file}")
                                return str(gguf_file), f"OK: GGUF model created on-demand ({file_size_mb:.1f} MB)", gr.File(visible=True, label=f"Package: GGUF Model: {gguf_file.name} ({file_size_mb:.1f} MB)")
                            
                            return None, "Error: Conversion appeared to succeed but no GGUF file found", gr.File(visible=False)
                            
                    except subprocess.TimeoutExpired:
                        print(f"[GGUF] Conversion timed out after 10 minutes")
                        return None, "Error: GGUF conversion timed out (10 minutes). The model may be too large or the conversion is stuck.", gr.File(visible=False)
                    except Exception as e:
                        print(f"[GGUF] Error during conversion: {e}")
                        import traceback
                        print(f"[GGUF] Traceback: {traceback.format_exc()}")
                        return None, f"Error: Error during GGUF conversion: {str(e)}", gr.File(visible=False)
                
                adapter_btn.click(fn=download_adapter, inputs=[state], outputs=[adapter_file, download_status, adapter_file])
                fused_btn.click(fn=download_fused, inputs=[state], outputs=[fused_file, download_status, fused_file])
                gguf_btn.click(fn=download_gguf, inputs=[state], outputs=[gguf_file, download_status, gguf_file])
                
                # Auto-refresh status when Results tab is selected
                results_tab.select(
                    fn=refresh_results_status,
                    inputs=[state],
                    outputs=[results_status]
                )
            
            # Tab 5: Chat
            with gr.TabItem("5. Chat") as chat_tab:
                gr.Markdown("## Step 5: Chat & Compare")
                gr.Markdown("Test your fine-tuned model against the base model with advanced controls")
                
                chat_status = gr.Textbox(
                    label="Status",
                    value="Training must be complete to use chat",
                    interactive=False
                )
                
                # Advanced Parameters Section
                with gr.Accordion("⚙️ Advanced Parameters", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            system_prompt = gr.Textbox(
                                label="System Prompt",
                                value="You are a helpful assistant. Answer the user's question accurately and concisely.",
                                lines=3,
                                placeholder="Define the model's behavior and personality..."
                            )
                        with gr.Column(scale=1):
                            with gr.Row():
                                temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=0.7,
                                    step=0.1,
                                    info="Lower = more deterministic, Higher = more creative"
                                )
                            with gr.Row():
                                top_p = gr.Slider(
                                    label="Top-p (Nucleus Sampling)",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.05,
                                    info="Probability threshold for token selection"
                                )
                            with gr.Row():
                                max_tokens = gr.Slider(
                                    label="Max Tokens",
                                    minimum=10,
                                    maximum=2048,
                                    value=512,
                                    step=64,
                                    info="Maximum response length"
                                )
                    with gr.Row():
                        streaming_toggle = gr.Checkbox(
                            label="Enable Streaming",
                            value=False,
                            info="Show tokens as they generate (slower but more interactive)"
                        )
                        use_fused = gr.Checkbox(
                            label="Use Fused Model",
                            value=False,
                            info="Load fused model instead of base+adapter (faster, more memory)"
                        )
                
                # Token Counter Display
                with gr.Row():
                    token_info = gr.Textbox(
                        label="Token Information",
                        value="Tokens: Input 0 | Output 0 | Total 0",
                        interactive=False
                    )
                
                # Conversation History (Text format for now)
                with gr.Accordion("💬 Conversation History", open=False):
                    conversation_display = gr.Textbox(
                        label="Conversation Log",
                        lines=10,
                        max_lines=50,
                        value="Conversation history will appear here...",
                        interactive=False
                    )
                
                # Input Area
                with gr.Row():
                    question = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your question or message here...",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                # Model Response Display
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Base Model (Original)")
                        base_response = gr.Markdown(
                            value="*Base model response will appear here*",
                            container=True,
                            height=200
                        )
                        base_stats = gr.Textbox(
                            label="Stats",
                            value="Time: - | Tokens: - | Speed: -",
                            interactive=False,
                            visible=True
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Fine-Tuned Model (Trained)")
                        tuned_response = gr.Markdown(
                            value="*Fine-tuned model response will appear here*",
                            container=True,
                            height=200
                        )
                        tuned_stats = gr.Textbox(
                            label="Stats",
                            value="Time: - | Tokens: - | Speed: -",
                            interactive=False,
                            visible=True
                        )
                
                with gr.Row():
                    refresh_chat_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
                    clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary", size="sm")
                    compare_btn = gr.Button("📊 Compare Models", variant="secondary", size="sm")
                
                def refresh_chat_status(current_state):
                    """Check current training status for chat tab."""
                    # First check disk-based state (fallback if Gradio state lost)
                    disk_state = load_state_from_disk()
                    
                    # Merge disk state with current state
                    if disk_state.get('training_complete'):
                        current_state = {
                            **current_state,
                            'training_complete': True,
                            'training_active': disk_state.get('training_active', False),
                            'output_dir': disk_state.get('output_dir', current_state.get('output_dir')),
                        }
                    
                    if current_state.get('training_complete'):
                        output_dir = current_state.get('output_dir', '')
                        
                        # If output_dir doesn't exist or is empty, try to find the actual directory
                        if not output_dir or not Path(output_dir).exists():
                            print(f"[CHAT] Output dir from state not found: {output_dir}")
                            print(f"[CHAT] Searching for actual output directory in outputs/...")
                            
                            # Look for directories with adapters
                            outputs_base = Path("outputs")
                            if outputs_base.exists():
                                for subdir in outputs_base.iterdir():
                                    if subdir.is_dir():
                                        # Check if this directory has adapters
                                        has_adapters = (subdir / "adapters" / "adapters.safetensors").exists() or \
                                                      (subdir / "best_adapter" / "adapters.safetensors").exists()
                                        
                                        if has_adapters:
                                            output_dir = str(subdir)
                                            print(f"[CHAT] OK: Found valid output directory: {output_dir}")
                                            # Update the state with the correct path
                                            current_state['output_dir'] = output_dir
                                            save_state_to_disk(current_state)
                                            break
                        
                        if output_dir and Path(output_dir).exists():
                            # Check if adapter exists (check correct path)
                            adapter_exists = (Path(output_dir) / "adapters" / "adapters.safetensors").exists() or \
                                           (Path(output_dir) / "best_adapter" / "adapters.safetensors").exists()
                            if adapter_exists:
                                return "OK: Ready! Fine-tuned model available."
                            else:
                                return "Warning: Training complete but adapter not found. Only base model available."
                        else:
                            return "OK: Training complete! (Output directory not found - try refreshing)"
                    elif current_state.get('training_active'):
                        return "Loading: Training in progress... Check Train tab for details."
                    else:
                        return "Warning: No training completed. Go to Train tab to start training."
                
                refresh_chat_btn.click(
                    fn=refresh_chat_status,
                    inputs=[state],
                    outputs=[chat_status]
                )
                
                # Global conversation history storage
                conversation_history = []
                
                def ask_models(question, system_prompt, temperature, top_p, max_tokens, streaming_toggle, use_fused, current_state):
                    """Enhanced chat function with parameter controls and both model comparison."""
                    if not question or not question.strip():
                        return (
                            "No messages yet.",
                            "*No question provided*",
                            "*No question provided*",
                            "Stats: -",
                            "Stats: -",
                            "Tokens: Input 0 | Output 0 | Total 0",
                            "Please enter a message to start the conversation."
                        )
                    
                    # Check disk state for training completion
                    disk_state = load_state_from_disk()
                    is_complete = current_state.get("training_complete") or (disk_state.get("training_complete") if disk_state else False)
                    output_dir = current_state.get('output_dir') or disk_state.get('output_dir')
                    
                    # If output_dir doesn't exist, try to auto-discover
                    if not output_dir or not Path(output_dir).exists():
                        print(f"[CHAT] Output dir not found: {output_dir}, searching...")
                        outputs_base = Path("outputs")
                        if outputs_base.exists():
                            for subdir in outputs_base.iterdir():
                                if subdir.is_dir():
                                    if (subdir / "fused_model").exists():
                                        output_dir = str(subdir)
                                        print(f"[CHAT] OK: Auto-discovered: {output_dir}")
                                        current_state['output_dir'] = output_dir
                                        save_state_to_disk(current_state)
                                        break
                    
                    if not is_complete:
                        return (
                            "No messages yet.",
                            "Training not complete. Please complete training first.",
                            "Training not complete. Please complete training first.",
                            "Stats: -",
                            "Stats: -",
                            "Tokens: Input 0 | Output 0 | Total 0",
                            "Warning: Training must be completed before using chat."
                        )
                    
                    config = current_state.get('training_config', {})
                    # Get model_id from config, or use actual model from adapter, or fall back to reading adapter config
                    model_id = config.get('model_id')
                    if not model_id and output_dir:
                        # Try to read actual model from adapter_config.json
                        adapter_config_paths = [
                            Path(output_dir) / "adapters" / "adapter_config.json",
                            Path(output_dir) / "best_adapter" / "adapter_config.json",
                        ]
                        for config_path in adapter_config_paths:
                            if config_path.exists():
                                try:
                                    with open(config_path, 'r') as f:
                                        adapter_cfg = json.load(f)
                                    actual_model = adapter_cfg.get('model')
                                    if actual_model:
                                        model_id = actual_model
                                        print(f"[CHAT] Read model from adapter config: {model_id}")
                                        break
                                except:
                                    pass
                    
                    # Final fallback to default from STUDIO_MODELS
                    if not model_id:
                        default_model = STUDIO_MODELS.get_model(STUDIO_MODELS.DEFAULT_STUDIO_MODEL)
                        model_id = default_model.get('model_id', 'mlx-community/Phi-3-mini-4k_instruct-4bit')
                        print(f"[CHAT] Using default model: {model_id}")
                    
                    print(f"[CHAT] ===== New Message =====")
                    print(f"[CHAT] Question: {question[:100]}...")
                    print(f"[CHAT] System: {system_prompt[:80]}...")
                    print(f"[CHAT] Parameters: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
                    print(f"[CHAT] Model: {model_id}")
                    print(f"[CHAT] Output dir: {output_dir}")
                    print(f"[CHAT] Use fused: {use_fused}")
                    
                    base_resp = ""
                    tuned_resp = ""
                    base_stats = "Stats: Error | Tokens: - | Speed: -"
                    tuned_stats = "Stats: Error | Tokens: - | Speed: -"
                    token_info = "Tokens: Input 0 | Output 0 | Total 0"
                    
                    try:
                        import time
                        
                        # Initialize base model
                        print(f"[CHAT] Loading base model...")
                        base_start = time.time()
                        base_wrapper = ChatWrapper(model_id)
                        base_load_time = time.time() - base_start
                        print(f"[CHAT] OK: Base model loaded in {base_load_time:.2f}s")
                        
                        # Get base model response
                        print(f"[CHAT] Generating base model response...")
                        base_gen_start = time.time()
                        try:
                            base_resp = base_wrapper.chat(
                                question.strip(),
                                system_prompt=system_prompt,
                                temperature=temperature,
                                top_p=top_p,
                                max_tokens=int(max_tokens)
                            )
                            base_gen_time = time.time() - base_gen_start
                            # Get accurate metrics from ChatWrapper
                            base_metrics = base_wrapper.get_last_metrics()
                            base_token_count = base_metrics.get('generation_tokens', len(base_resp.split()))
                            base_tps = base_metrics.get('generation_tps', base_token_count / max(base_gen_time, 0.1))
                            base_stats = f"Time: {base_gen_time:.2f}s | Tokens: {base_token_count} | Speed: {base_tps:.1f} tok/s"
                            print(f"[CHAT] OK: Base model: {len(base_resp)} chars, {base_token_count} tokens, {base_gen_time:.2f}s")
                        except Exception as e:
                            print(f"[CHAT] Error: Base model generation error: {e}")
                            base_resp = f"[Error generating response: {str(e)}]"
                            base_stats = f"Stats: Error | Time: {time.time() - base_gen_start:.2f}s"
                        
                        # Find and load fine-tuned model
                        tuned_wrapper = None
                        tuned_load_time = 0
                        tuned_base_model = None  # Will store the actual base model from adapter config
                        
                        # First, try to read adapter_config.json to get the correct base model
                        adapter_config_path = None
                        if output_dir:
                            possible_config_paths = [
                                Path(output_dir) / "adapters" / "adapter_config.json",
                                Path(output_dir) / "best_adapter" / "adapter_config.json",
                            ]
                            for config_path in possible_config_paths:
                                if config_path.exists():
                                    adapter_config_path = config_path
                                    break
                        
                        if adapter_config_path:
                            try:
                                with open(adapter_config_path, 'r') as f:
                                    adapter_config = json.load(f)
                                tuned_base_model = adapter_config.get('model')
                                print(f"[CHAT] Adapter trained with base model: {tuned_base_model}")
                            except Exception as e:
                                print(f"[CHAT] Could not read adapter config: {e}")
                        
                        # Use the correct base model (from adapter config) or fall back to current model
                        if tuned_base_model:
                            actual_base_model = tuned_base_model
                            print(f"[CHAT] Using adapter's base model: {actual_base_model}")
                        else:
                            actual_base_model = model_id
                            print(f"[CHAT] Using current model as base: {actual_base_model}")
                        
                        if use_fused and output_dir:
                            # Try to use fused model directly
                            fused_path = Path(output_dir) / "fused_model"
                            if fused_path.exists():
                                print(f"[CHAT] Loading fused model from {fused_path}...")
                                fused_start = time.time()
                                try:
                                    tuned_wrapper = ChatWrapper(str(fused_path))
                                    tuned_load_time = time.time() - fused_start
                                    print(f"[CHAT] OK: Fused model loaded in {tuned_load_time:.2f}s")
                                except Exception as e:
                                    print(f"[CHAT] Warning: Failed to load fused model: {e}")
                                    tuned_wrapper = None
                        
                        if not tuned_wrapper and output_dir:
                            # Try to load with adapter using the CORRECT base model
                            adapter_paths = [
                                Path(output_dir) / "adapters" / "adapters.safetensors",
                                Path(output_dir) / "best_adapter" / "adapters.safetensors",
                                Path(output_dir) / "adapters.safetensors",
                            ]
                            
                            adapter_path = None
                            for p in adapter_paths:
                                if p.exists():
                                    adapter_path = str(p)
                                    break
                            
                            if adapter_path:
                                print(f"[CHAT] Loading fine-tuned model with adapter: {adapter_path}")
                                print(f"[CHAT] IMPORTANT: Using adapter's base model: {actual_base_model}")
                                tuned_start = time.time()
                                try:
                                    # Use the correct base model, not the default!
                                    tuned_wrapper = ChatWrapper(actual_base_model, adapter_path=adapter_path)
                                    tuned_load_time = time.time() - tuned_start
                                    print(f"[CHAT] OK: Fine-tuned model loaded in {tuned_load_time:.2f}s")
                                except Exception as e:
                                    print(f"[CHAT] Warning: Failed to load fine-tuned model: {e}")
                                    tuned_wrapper = None
                            else:
                                print(f"[CHAT] Warning: No adapter found in {output_dir}")
                        
                        # Get fine-tuned model response
                        if tuned_wrapper:
                            print(f"[CHAT] Generating fine-tuned model response...")
                            tuned_gen_start = time.time()
                            try:
                                tuned_resp = tuned_wrapper.chat(
                                    question.strip(),
                                    system_prompt=system_prompt,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_tokens=int(max_tokens)
                                )
                                tuned_gen_time = time.time() - tuned_gen_start
                                # Get accurate metrics from ChatWrapper
                                tuned_metrics = tuned_wrapper.get_last_metrics()
                                tuned_token_count = tuned_metrics.get('generation_tokens', len(tuned_resp.split()))
                                tuned_tps = tuned_metrics.get('generation_tps', tuned_token_count / max(tuned_gen_time, 0.1))
                                tuned_stats = f"Time: {tuned_gen_time:.2f}s | Tokens: {tuned_token_count} | Speed: {tuned_tps:.1f} tok/s"
                                print(f"[CHAT] OK: Fine-tuned model: {len(tuned_resp)} chars, {tuned_token_count} tokens, {tuned_gen_time:.2f}s")
                            except Exception as e:
                                print(f"[CHAT] Error: Fine-tuned model generation error: {e}")
                                tuned_resp = f"[Error generating response: {str(e)}]"
                                tuned_stats = f"Stats: Error | Time: {time.time() - tuned_gen_start:.2f}s"
                        else:
                            tuned_resp = "[Fine-tuned model not available. The adapter may be missing or failed to load.]"
                            tuned_stats = "Stats: Not Available"
                            print(f"[CHAT] Warning: Fine-tuned model unavailable")
                        
                        # Update conversation history
                        history_entry = f"Q: {question}\nBase: {base_resp[:300]}...\nTuned: {tuned_resp[:300]}...\n{'='*50}\n"
                        conversation_history.append(history_entry)
                        
                        # Format for display (join all history)
                        history_text = "\n".join(conversation_history) if conversation_history else "No messages yet."
                        
                        # Calculate token info
                        input_tokens = len(question.split())
                        output_tokens = len(base_resp.split()) + len(tuned_resp.split())
                        total_tokens = input_tokens + output_tokens
                        token_info = f"Tokens: Input {input_tokens} | Output {output_tokens} | Total {total_tokens}"
                        
                        status_msg = "OK: Responses generated successfully"
                        
                    except Exception as e:
                        print(f"[CHAT] Error: Critical error: {e}")
                        import traceback
                        print(f"[CHAT] Traceback: {traceback.format_exc()}")
                        base_resp = f"[System Error: {str(e)}]"
                        tuned_resp = f"[System Error: {str(e)}]"
                        status_msg = f"Error: Error: {str(e)[:100]}"
                    
                    return (
                        "\n".join(conversation_history) if conversation_history else "No messages yet.",
                        base_resp,
                        tuned_resp,
                        base_stats,
                        tuned_stats,
                        token_info,
                        status_msg
                    )
                
                def clear_conversation():
                    """Clear the conversation history."""
                    global conversation_history
                    conversation_history = []
                    return "No messages yet.", "*Conversation cleared*", "*Conversation cleared*", "Stats: -", "Stats: -", "Tokens: Input 0 | Output 0 | Total 0", "Conversation history cleared."
                
                def compare_models(question, system_prompt, temperature, top_p, max_tokens, current_state):
                    """Run the same question multiple times to compare consistency."""
                    if not question or not question.strip():
                        return "Please enter a question to compare"
                    
                    results = []
                    for i in range(3):
                        try:
                            config = current_state.get('training_config', {})
                            output_dir = current_state.get('output_dir')
                            
                            # Get model_id from config, or use actual model from adapter
                            model_id = config.get('model_id')
                            if not model_id and output_dir:
                                # Try to read actual model from adapter_config.json
                                adapter_config_paths = [
                                    Path(output_dir) / "adapters" / "adapter_config.json",
                                    Path(output_dir) / "best_adapter" / "adapter_config.json",
                                ]
                                for config_path in adapter_config_paths:
                                    if config_path.exists():
                                        try:
                                            with open(config_path, 'r') as f:
                                                adapter_cfg = json.load(f)
                                            actual_model = adapter_cfg.get('model')
                                            if actual_model:
                                                model_id = actual_model
                                                print(f"[COMPARE] Read model from adapter config: {model_id}")
                                                break
                                        except:
                                            pass
                            
                            # Final fallback to default from STUDIO_MODELS
                            if not model_id:
                                default_model = STUDIO_MODELS.get_model(STUDIO_MODELS.DEFAULT_STUDIO_MODEL)
                                model_id = default_model.get('model_id', 'mlx-community/Phi-3-mini-4k_instruct-4bit')
                                print(f"[COMPARE] Using default model: {model_id}")
                            
                            wrapper = ChatWrapper(model_id)
                            resp = wrapper.chat(question.strip(), system_prompt=system_prompt, temperature=temperature, top_p=top_p, max_tokens=int(max_tokens))
                            results.append(f"**Run {i+1}:**\n{resp[:200]}...")
                        except Exception as e:
                            results.append(f"**Run {i+1}:** Error - {str(e)}")
                    
                    return "\n\n".join(results)
                
                # Wire up event handlers
                send_btn.click(
                    fn=ask_models,
                    inputs=[
                        question, system_prompt, temperature, top_p, max_tokens,
                        streaming_toggle, use_fused, state
                    ],
                    outputs=[
                        conversation_display, base_response, tuned_response,
                        base_stats, tuned_stats, token_info, chat_status
                    ]
                )
                
                clear_history_btn.click(
                    fn=clear_conversation,
                    inputs=[],
                    outputs=[
                        conversation_display, base_response, tuned_response,
                        base_stats, tuned_stats, token_info, chat_status
                    ]
                )
                
                compare_btn.click(
                    fn=compare_models,
                    inputs=[question, system_prompt, temperature, top_p, max_tokens, state],
                    outputs=[base_response]
                )
                
                # Auto-refresh status when Chat tab is selected
                chat_tab.select(
                    fn=refresh_chat_status,
                    inputs=[state],
                    outputs=[chat_status]
                )
            
            # Add Results tab auto-refresh
            results_tab.select(
                fn=refresh_results_status,
                inputs=[state],
                outputs=[results_status]
            )
            
            # Tab 6: Models
            with gr.TabItem("Models"):
                gr.Markdown("## Manage Models and Tokens")
                gr.Markdown("Add custom models from HuggingFace and manage your access tokens.")
                
                # Refresh Models Section
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("---")
                        gr.Markdown("**Refresh Model List**")
                        gr.Markdown("*Click to reload available models after adding custom models*")
                        
                        models_refresh_status = gr.Textbox(
                            label="Status",
                            value="Click refresh to update model list",
                            interactive=False,
                            show_label=False
                        )
                        
                        models_refresh_btn = gr.Button("🔄 Refresh Models", variant="secondary", size="sm")
                
                # Custom Model Section
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("---")
                        gr.Markdown("**Add Custom Model from HuggingFace**")
                        
                        models_custom_model_input = gr.Textbox(
                            label="Model URL or ID",
                            placeholder="e.g., mlx-community/Llama-3.2-1B-Instruct-4bit",
                            value=""
                        )
                        
                        with gr.Row():
                            models_verify_model_btn = gr.Button("Verify Model", size="sm")
                            models_add_custom_model_btn = gr.Button("Add to My Models", size="sm", interactive=False)
                        
                        models_custom_model_status = gr.Textbox(
                            label="Status",
                            value="",
                            interactive=False,
                            show_label=False
                        )
                        
                        models_custom_model_details = gr.Textbox(
                            label="Model Details",
                            value="",
                            interactive=False,
                            visible=True,
                            lines=6
                        )
                
                # HuggingFace Token Section
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("---")
                        gr.Markdown("**HuggingFace Token (Optional)**")
                        gr.Markdown("*Required for gated models (e.g., Meta Llama)*")
                        
                        models_hf_token_input = gr.Textbox(
                            label="Token",
                            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                            value="",
                            type="password"
                        )
                        
                        with gr.Row():
                            models_save_token_btn = gr.Button("Save Token", size="sm")
                            models_clear_token_btn = gr.Button("Clear Token", size="sm")
                        
                        models_hf_token_status = gr.Textbox(
                            label="Token Status",
                            value="No token configured",
                            interactive=False,
                            show_label=False
                        )
                
                # Models Tab Handlers
                def models_verify_custom_model(user_input):
                    """Verify a custom HuggingFace model."""
                    try:
                        from edukaai_studio.ui.hf_model_validator import HFModelValidator, format_model_info_for_display
                        from edukaai_studio.ui.user_config import get_hf_token
                        
                        # Debug: log what user entered
                        print(f"[VERIFY MODEL] Raw input: '{user_input}'")
                        
                        # Clean input: remove anything after # (comments) and strip whitespace
                        user_input_clean = user_input.split('#')[0].strip() if user_input else user_input
                        print(f"[VERIFY MODEL] Cleaned input: '{user_input_clean}'")
                        
                        # Get token if configured
                        token = get_hf_token()
                        validator = HFModelValidator(token=token)
                        
                        model_id = validator.parse_model_input(user_input_clean)
                        print(f"[VERIFY MODEL] Parsed model_id: '{model_id}'")
                        
                        if not model_id:
                            return "Error: Invalid format. Use: organization/model-name", "", gr.update(interactive=False)
                        
                        # Check if already exists
                        from edukaai_studio.ui.hf_model_validator import UserModelsManager
                        mgr = UserModelsManager()
                        if mgr.model_exists(model_id):
                            return f"OK: Model already in your list: {model_id}", "", gr.update(interactive=False)
                        
                        # Validate the model
                        info = validator.validate_model(model_id)
                        print(f"[VERIFY MODEL] Validation result - name: '{info.name}', id: '{info.model_id}', error: {info.error}")
                        
                        if info.error:
                            return f"Error: {info.error}", "", gr.update(interactive=False)
                        
                        # Format details for display
                        details = format_model_info_for_display(info)
                        
                        if info.mlx_compatible:
                            status = f"OK: {info.name} verified and ready to add!"
                            return status, details, gr.update(interactive=True)
                        else:
                            status = f"Warning: {info.name} found but not MLX-optimized"
                            return status, details, gr.update(interactive=True)
                            
                    except Exception as e:
                        import traceback
                        print(f"[VERIFY MODEL] Exception: {e}\n{traceback.format_exc()}")
                        return f"Error: Error: {str(e)}", "", gr.update(interactive=False)
                
                def models_add_custom_model(user_input, current_state):
                    """Add verified model to user's list."""
                    try:
                        from edukaai_studio.ui.hf_model_validator import HFModelValidator, UserModelsManager
                        from edukaai_studio.ui.user_config import get_hf_token
                        
                        # Clean input: remove anything after # (comments) and strip whitespace
                        user_input_clean = user_input.split('#')[0].strip() if user_input else user_input
                        print(f"[ADD MODEL] Cleaned input: '{user_input_clean}'")
                        
                        token = get_hf_token()
                        validator = HFModelValidator(token=token)
                        model_id = validator.parse_model_input(user_input_clean)
                        
                        if not model_id:
                            return "Error: Invalid input", current_state
                        
                        # Validate and get info
                        info = validator.validate_model(model_id)
                        
                        if info.error:
                            return f"Error: {info.error}", current_state
                        
                        # Add to user models
                        mgr = UserModelsManager()
                        if mgr.model_exists(model_id):
                            return "OK: Model already in your list!", current_state
                        
                        success = mgr.add_model(info)
                        print(f"[ADD MODEL] Added model: {model_id}, success: {success}")
                        
                        if success:
                            return f"OK: Added {info.name}! Available in Configure tab.", current_state
                        else:
                            return "Error: Failed to add model (may already exist)", current_state
                            
                    except Exception as e:
                        import traceback
                        print(f"[ADD MODEL] Exception: {e}\n{traceback.format_exc()}")
                        return f"Error: Error: {str(e)}", current_state
                
                models_verify_model_btn.click(
                    fn=models_verify_custom_model,
                    inputs=[models_custom_model_input],
                    outputs=[models_custom_model_status, models_custom_model_details, models_add_custom_model_btn]
                )
                
                models_add_custom_model_btn.click(
                    fn=models_add_custom_model,
                    inputs=[models_custom_model_input, state],
                    outputs=[models_custom_model_status, state]
                )
                
                # Token Handlers
                def models_save_hf_token(token_input):
                    """Save HF token to config."""
                    try:
                        from edukaai_studio.ui.user_config import set_hf_token, mask_token
                        
                        if not token_input or not token_input.strip():
                            return "No token entered"
                        
                        token = token_input.strip()
                        
                        # Basic validation: should start with 'hf_'
                        if not token.startswith('hf_'):
                            return "Error: Invalid token format. Should start with 'hf_'"
                        
                        # Save token
                        set_hf_token(token)
                        
                        masked = mask_token(token)
                        return f"OK: Token saved: {masked}"
                        
                    except Exception as e:
                        return f"Error: Error saving token: {str(e)}"
                
                def models_clear_hf_token():
                    """Clear HF token from config."""
                    try:
                        from edukaai_studio.ui.user_config import clear_hf_token as clear_token
                        clear_token()
                        return "Token cleared"
                    except Exception as e:
                        return f"Error: Error: {str(e)}"
                
                def models_init_hf_token_status():
                    """Initialize token status on load."""
                    try:
                        from edukaai_studio.ui.user_config import has_hf_token, get_hf_token, mask_token
                        if has_hf_token():
                            token = get_hf_token()
                            masked = mask_token(token)
                            return f"OK: Token configured: {masked}"
                        return "No token configured"
                    except:
                        return "No token configured"
                
                models_save_token_btn.click(
                    fn=models_save_hf_token,
                    inputs=[models_hf_token_input],
                    outputs=[models_hf_token_status]
                )
                
                models_clear_token_btn.click(
                    fn=models_clear_hf_token,
                    outputs=[models_hf_token_status]
                )
                
                # Initialize token status on load
                app.load(fn=models_init_hf_token_status, outputs=[models_hf_token_status])
                
                # Refresh Models Handler
                def models_refresh_model_list():
                    """Refresh the model dropdown choices."""
                    try:
                        from edukaai_studio.ui.hf_model_validator import UserModelsManager
                        
                        # Reload user models
                        mgr = UserModelsManager()
                        user_models = mgr.get_all_models()
                        
                        count = len(user_models)
                        if count > 0:
                            return f"OK: Model list refreshed! {count} custom model(s) available. Go to Configure tab to use them."
                        else:
                            return "OK: Model list refreshed! No custom models yet."
                            
                    except Exception as e:
                        return f"Error: Error refreshing models: {str(e)}"
                
                models_refresh_btn.click(
                    fn=models_refresh_model_list,
                    outputs=[models_refresh_status]
                )
            
            # New Training Session Handler
            def start_new_training_session(current_state):
                """
                Reset all state to start a new training session.
                Archives old training data to history.
                """
                print("[NEW SESSION] Starting new training session...")
                
                # Archive current training if exists
                if current_state.get('training_complete') and current_state.get('output_dir'):
                    archive_dir = Path("training_history")
                    archive_dir.mkdir(exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = current_state.get('model_name', 'unknown').replace(' ', '_')
                    archive_name = f"{model_name}_{timestamp}"
                    
                    # Create archive entry
                    archive_info = {
                        'timestamp': timestamp,
                        'model_name': current_state.get('model_name', 'unknown'),
                        'output_dir': current_state.get('output_dir'),
                        'completion_time': current_state.get('completion_time'),
                    }
                    
                    archive_file = archive_dir / f"{archive_name}.json"
                    try:
                        with open(archive_file, 'w') as f:
                            json.dump(archive_info, f, indent=2)
                        print(f"[NEW SESSION] Archived to {archive_file}")
                    except Exception as e:
                        print(f"[NEW SESSION] Archive error: {e}")
                
                # Clear state file
                if STATE_FILE.exists():
                    try:
                        STATE_FILE.unlink()
                        print(f"[NEW SESSION] Cleared state file")
                    except Exception as e:
                        print(f"[NEW SESSION] Error clearing state: {e}")
                
                # Return completely fresh state
                fresh_state = {
                    'uploaded_file': None,
                    'selected_model': STUDIO_MODELS.DEFAULT_STUDIO_MODEL,
                    'selected_preset': 'balanced',
                    'training_config': {},
                    'training_active': False,
                    'training_complete': False,
                    'output_dir': None,
                    'train_losses': {},
                    'val_losses': {},
                    'best_loss': float('inf'),
                    'best_iter': 0,
                    'log_lines': [],
                    'monitor': None,
                    'completion_time': None,
                    'model_name': None,
                }
                
                print("[NEW SESSION] New session initialized")
                
                # Return updated state and UI reset values
                return (
                    fresh_state,
                    "Step 1: Upload your training data (JSONL or Alpaca JSON)",
                    "Select model and preset",
                    "Ready to train",
                    "No training yet",
                    "Warning: No training completed",
                    "No messages yet.",
                    "*Upload data to begin*",
                    "*Upload data to begin*",
                    gr.Button(visible=False),  # adapter_file
                    gr.Button(visible=False),  # fused_file
                    gr.Button(visible=False),  # gguf_file
                )
            
            # Wire up New Training button
            new_training_btn.click(
                fn=start_new_training_session,
                inputs=[state],
                outputs=[
                    state,
                    file_status,
                    status_text,
                    train_status,
                    results_status,
                    chat_status,
                    conversation_display,
                    base_response,
                    tuned_response,
                    adapter_file,
                    fused_file,
                    gguf_file,
                ]
            )
        
        # Training selector event handlers
        def on_training_selected(selected_folder, current_state):
            """When user selects a training from dropdown."""
            if not selected_folder:
                return current_state, gr.update(visible=False)
            
            # Update state with selected training
            new_state = {
                **current_state,
                'output_dir': selected_folder,
                'training_complete': True,
                'training_active': False
            }
            
            # Show delete button when training is selected
            return new_state, gr.update(visible=True)
        
        # Connect dropdown selection
        training_selector.change(
            fn=on_training_selected,
            inputs=[training_selector, state],
            outputs=[state, delete_training_btn]
        )
        
        # Delete training handler
        def delete_selected_training(current_state):
            """Delete the currently selected training folder."""
            output_dir = current_state.get('output_dir')
            if not output_dir:
                return current_state, gr.update(choices=[], value=None), gr.update(visible=False)
            
            try:
                import shutil
                folder_path = Path(output_dir)
                folder_name = folder_path.name
                
                if folder_path.exists():
                    shutil.rmtree(folder_path)
                    print(f"[DELETE] Removed training folder: {folder_name}")
                
                # Update dropdown choices
                from edukaai_studio.ui.training_scanner import scan_output_folders, get_training_choices
                trainings = scan_output_folders()
                choices = get_training_choices(trainings)
                
                # Clear the deleted training from state
                new_state = {
                    **current_state,
                    'output_dir': None,
                    'training_complete': False
                }
                
                return new_state, gr.update(choices=choices, value=None), gr.update(visible=False)
                
            except Exception as e:
                print(f"[DELETE] Error deleting training: {e}")
                return current_state, gr.update(), gr.update(visible=True)
        
        delete_training_btn.click(
            fn=delete_selected_training,
            inputs=[state],
            outputs=[state, training_selector, delete_training_btn]
        )
        
        # Auto-populate on app load
        def init_training_selector():
            """Initialize selector on app start."""
            from edukaai_studio.ui.training_scanner import scan_output_folders, get_training_choices
            try:
                trainings = scan_output_folders()
                choices = get_training_choices(trainings)
                return gr.update(choices=choices)
            except:
                return gr.update(choices=[])
        
        # Set initial choices
        app.load(fn=init_training_selector, outputs=[training_selector])
        
        gr.Markdown("---")
        gr.Markdown("EdukaAI Studio | Fine-tuning made simple")
        
        return app


def main():
    print("=" * 60)
    print("Starting EdukaAI Studio")
    print("Real MLX Training Version")
    print("=" * 60)
    if SERVER.SHOW_LOCALHOST_URL:
        print(f"Open browser at: {SERVER.LOCALHOST_URL}")
    print(f"Server binding: {SERVER.HOST}:{SERVER.PORT}")
    print("=" * 60)
    
    app = create_ui()
    app.launch(
        server_name=SERVER.HOST,
        server_port=SERVER.PORT,
        share=SERVER.SHARE,
        quiet=SERVER.QUIET
    )


if __name__ == "__main__":
    main()
