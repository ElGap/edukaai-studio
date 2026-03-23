"""Configure Tab for EdukaAI Studio.

Handles model selection and training configuration.
"""

import gradio as gr
from typing import Dict, Any, Tuple
from edukaai_studio.config import STUDIO_MODELS, TRAINING


def configure_training(
    model_id: str,
    preset_name: str,
    iterations: int,
    learning_rate: str,
    lora_rank: int,
    lora_alpha: int,
    max_seq_length: int,
    early_stopping: int,
    validation_split: int,
    grad_accumulation: int,
    current_state: Dict[str, Any]
) -> Tuple:
    """Configure training with selected parameters.
    
    Args:
        model_id: Selected model ID
        preset_name: Training preset name
        iterations: Number of training iterations
        learning_rate: Learning rate as string
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        max_seq_length: Maximum sequence length
        early_stopping: Early stopping patience
        validation_split: Validation split percentage
        grad_accumulation: Gradient accumulation steps
        current_state: Current application state
        
    Returns:
        Tuple of (status_message, new_state)
    """
    if not current_state.get('uploaded_file'):
        return "Warning: Upload training data first", current_state
    
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
                    break
        except Exception as e:
            print(f"[CONFIGURE] Error checking user models: {e}")
    
    if not model:
        return f"Error: Invalid model selected ({model_id})", current_state
    
    # Debug: log the received parameters
    print(f"[CONFIGURE DEBUG] Received parameters:")
    print(f"[CONFIGURE DEBUG]   model_id: {model_id}")
    print(f"[CONFIGURE DEBUG]   preset_name: {preset_name}")
    print(f"[CONFIGURE DEBUG]   iterations: {iterations}")
    print(f"[CONFIGURE DEBUG]   learning_rate: {learning_rate}")
    print(f"[CONFIGURE DEBUG]   lora_rank: {lora_rank}")
    print(f"[CONFIGURE DEBUG]   lora_alpha: {lora_alpha}")
    print(f"[CONFIGURE DEBUG]   max_seq_length: {max_seq_length}")
    print(f"[CONFIGURE DEBUG]   early_stopping: {early_stopping}")
    print(f"[CONFIGURE DEBUG]   validation_split: {validation_split}")
    print(f"[CONFIGURE DEBUG]   grad_accumulation: {grad_accumulation}")
    
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
    
    print(f"[CONFIGURE DEBUG] Saved training_config: {new_state['training_config']}")
    
    msg = f"OK: Configured: {model.get('name')} ({iterations} steps, rank={lora_rank}, lr={learning_rate})"
    return msg, new_state


def update_params_from_preset(preset_name: str):
    """Update all parameter sliders when preset changes.
    
    Args:
        preset_name: Name of the preset (quick, balanced, maximum)
        
    Returns:
        List of values for sliders
    """
    from edukaai_studio.config import TrainingPresets
    
    preset_map = {
        "quick": TrainingPresets.QUICK,
        "balanced": TrainingPresets.BALANCED,
        "maximum": TrainingPresets.MAXIMUM
    }
    
    preset = preset_map.get(preset_name, TrainingPresets.BALANCED)
    
    print(f"[PRESET] Applying {preset_name} preset: {preset}")
    
    return [
        preset.get("iterations", 600),
        preset.get("learning_rate", "1e-4"),
        preset.get("lora_rank", 16),
        preset.get("lora_alpha", 32),
        preset.get("grad_accumulation", 32),
        preset.get("early_stopping", 2),
    ]


def refresh_model_list():
    """Reload predefined and user models for the dropdown.
    
    Returns:
        Gradio update for dropdown choices
    """
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


def create_configure_tab(state: gr.State, tabs: gr.Tabs) -> Dict[str, Any]:
    """Create the Configure tab.
    
    Args:
        state: Gradio state object
        tabs: Gradio tabs container
        
    Returns:
        Dictionary of component references
    """
    components = {}
    
    with gr.TabItem("2. Configure") as configure_tab:
        gr.Markdown("## Step 2: Configure Training")
        
        with gr.Row():
            with gr.Column(scale=2):
                components['model_dropdown'] = gr.Dropdown(
                    label="Model",
                    choices=[(m["name"], m["id"]) for m in STUDIO_MODELS.get_all_models()],
                    value=STUDIO_MODELS.DEFAULT_STUDIO_MODEL
                )
            
            with gr.Column(scale=2):
                components['preset_dropdown'] = gr.Dropdown(
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
                components['iterations_slider'] = gr.Slider(
                    minimum=TRAINING.MIN_ITERATIONS, 
                    maximum=TRAINING.MAX_ITERATIONS, 
                    step=TRAINING.ITERATION_STEP, 
                    value=TRAINING.DEFAULT_ITERATIONS,
                    label="Training Steps (Iterations)"
                )
            with gr.Column():
                components['learning_rate_dropdown'] = gr.Dropdown(
                    choices=TRAINING.LEARNING_RATE_OPTIONS,
                    value=TRAINING.DEFAULT_LEARNING_RATE,
                    label="Learning Rate"
                )
        
        with gr.Row():
            with gr.Column():
                components['lora_rank_slider'] = gr.Slider(
                    minimum=8, 
                    maximum=64, 
                    step=8, 
                    value=TRAINING.DEFAULT_LORA_RANK,
                    label="LoRA Rank"
                )
            with gr.Column():
                components['lora_alpha_slider'] = gr.Slider(
                    minimum=16, 
                    maximum=128, 
                    step=16, 
                    value=TRAINING.DEFAULT_LORA_ALPHA,
                    label="LoRA Alpha"
                )
        
        with gr.Row():
            with gr.Column():
                components['max_seq_length_slider'] = gr.Slider(
                    minimum=512, maximum=4096, step=256, value=2048,
                    label="Max Sequence Length"
                )
            with gr.Column():
                components['early_stopping_slider'] = gr.Slider(
                    minimum=1, maximum=5, step=1, 
                    value=TRAINING.DEFAULT_EARLY_STOPPING_PATIENCE,
                    label="Early Stopping Patience"
                )
        
        with gr.Row():
            with gr.Column():
                components['validation_split_slider'] = gr.Slider(
                    minimum=TRAINING.MIN_VALIDATION_SPLIT_PCT, 
                    maximum=TRAINING.MAX_VALIDATION_SPLIT_PCT, 
                    step=5, 
                    value=TRAINING.DEFAULT_VALIDATION_SPLIT_PCT,
                    label="Validation Split (%)"
                )
            with gr.Column():
                components['grad_accum_slider'] = gr.Slider(
                    minimum=4, maximum=128, step=4, 
                    value=TRAINING.DEFAULT_GRAD_ACCUMULATION,
                    label="Gradient Accumulation Steps"
                )
        
        components['status_text'] = gr.Textbox(
            label="Configuration Status",
            value="",
            interactive=False
        )
        
        gr.Markdown("---")
        gr.Markdown("Click 'Save Configuration' to confirm your settings.")
        
        components['save_config_btn'] = gr.Button("Save Configuration", variant="primary")
        
        # Wire up events
        # Note: We don't wire preset_dropdown.change initially to avoid overriding user settings on load
        # User can manually select a preset if they want to apply it
        # components['preset_dropdown'].change(
        #     fn=update_params_from_preset,
        #     inputs=[components['preset_dropdown']],
        #     outputs=[
        #         components['iterations_slider'], 
        #         components['learning_rate_dropdown'], 
        #         components['lora_rank_slider'],
        #         components['lora_alpha_slider'], 
        #         components['grad_accum_slider'], 
        #         components['early_stopping_slider']
        #     ]
        # )
        
        components['save_config_btn'].click(
            fn=configure_training,
            inputs=[
                components['model_dropdown'], 
                components['preset_dropdown'], 
                components['iterations_slider'], 
                components['learning_rate_dropdown'],
                components['lora_rank_slider'], 
                components['lora_alpha_slider'], 
                components['max_seq_length_slider'],
                components['early_stopping_slider'], 
                components['validation_split_slider'], 
                components['grad_accum_slider'], 
                state
            ],
            outputs=[components['status_text'], state]
        )
        
        # Refresh model list when Configure tab is selected
        configure_tab.select(
            fn=refresh_model_list,
            outputs=[components['model_dropdown']]
        )
    
    return components
