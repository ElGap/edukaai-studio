"""Models Tab for EdukaAI Studio.

Handles custom model management and HuggingFace tokens.
"""

import gradio as gr
from typing import Dict, Any, Tuple


def refresh_models_list() -> str:
    """Refresh the model list and return status.
    
    Returns:
        Status message
    """
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
        return f"Error: Failed to refresh models: {str(e)}"


def save_hf_token(token_input: str) -> str:
    """Save HF token to config.
    
    Args:
        token_input: HF token string
        
    Returns:
        Status message
    """
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


def clear_hf_token() -> str:
    """Clear HF token from config.
    
    Returns:
        Status message
    """
    try:
        from edukaai_studio.ui.user_config import clear_hf_token as clear_token
        clear_token()
        return "Token cleared"
    except Exception as e:
        return f"Error: {str(e)}"


def verify_custom_model(user_input: str) -> Tuple:
    """Verify a custom HuggingFace model.
    
    Args:
        user_input: Model URL or ID
        
    Returns:
        Tuple of (status, details, button_state)
    """
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
        return f"Error: {str(e)}", "", gr.update(interactive=False)


def add_custom_model(user_input: str, current_state: Dict[str, Any]) -> Tuple:
    """Add verified model to user's list.
    
    Args:
        user_input: Model URL or ID
        current_state: Current application state
        
    Returns:
        Tuple of (status, state)
    """
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
        return f"Error: {str(e)}", current_state


def create_models_tab(state: gr.State, tabs: gr.Tabs) -> Dict[str, Any]:
    """Create the Models tab.
    
    Args:
        state: Gradio state object
        tabs: Gradio tabs container
        
    Returns:
        Dictionary of component references
    """
    components = {}
    
    with gr.TabItem("Models"):
        gr.Markdown("## Manage Models and Tokens")
        gr.Markdown("Add custom models from HuggingFace and manage your access tokens.")
        
        # Refresh Models Section
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("---")
                gr.Markdown("**Refresh Model List**")
                gr.Markdown("*Click to reload available models after adding custom models*")
                
                components['models_refresh_status'] = gr.Textbox(
                    label="Status",
                    value="Click refresh to update model list",
                    interactive=False,
                    show_label=False
                )
                
                components['models_refresh_btn'] = gr.Button("Refresh Models", variant="secondary", size="sm")
        
        # Custom Model Section
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("---")
                gr.Markdown("**Add Custom Model from HuggingFace**")
                
                components['models_custom_model_input'] = gr.Textbox(
                    label="Model URL or ID",
                    placeholder="e.g., mlx-community/Llama-3.2-1B-Instruct-4bit",
                    value=""
                )
                
                with gr.Row():
                    components['models_verify_model_btn'] = gr.Button("Verify Model", size="sm")
                    components['models_add_custom_model_btn'] = gr.Button("Add to My Models", size="sm", interactive=False)
                
                components['models_custom_model_status'] = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    show_label=False
                )
                
                components['models_custom_model_details'] = gr.Textbox(
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
                
                components['models_hf_token_input'] = gr.Textbox(
                    label="Token",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    value="",
                    type="password"
                )
                
                with gr.Row():
                    components['models_save_token_btn'] = gr.Button("Save Token", size="sm")
                    components['models_clear_token_btn'] = gr.Button("Clear Token", size="sm")
                
                components['models_hf_token_status'] = gr.Textbox(
                    label="Token Status",
                    value="No token configured",
                    interactive=False,
                    show_label=False
                )
        
        # Wire up events
        components['models_refresh_btn'].click(
            fn=refresh_models_list,
            outputs=[components['models_refresh_status']]
        )
        
        components['models_verify_model_btn'].click(
            fn=verify_custom_model,
            inputs=[components['models_custom_model_input']],
            outputs=[
                components['models_custom_model_status'], 
                components['models_custom_model_details'], 
                components['models_add_custom_model_btn']
            ]
        )
        
        components['models_add_custom_model_btn'].click(
            fn=add_custom_model,
            inputs=[components['models_custom_model_input'], state],
            outputs=[components['models_custom_model_status'], state]
        )
        
        components['models_save_token_btn'].click(
            fn=save_hf_token,
            inputs=[components['models_hf_token_input']],
            outputs=[components['models_hf_token_status']]
        )
        
        components['models_clear_token_btn'].click(
            fn=clear_hf_token,
            outputs=[components['models_hf_token_status']]
        )
    
    return components
