"""Chat Tab for EdukaAI Studio.

Handles chat interface for testing fine-tuned models against base models.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import gradio as gr

from edukaai_studio.config import STUDIO_MODELS
from edukaai_studio.core.state import load_state_from_disk, save_state_to_disk
from edukaai_studio.ui.chat_wrapper import ChatWrapper


# Global conversation history storage
conversation_history: List[str] = []


def refresh_chat_status(current_state: Dict[str, Any]) -> Tuple[str, str, str]:
    """Check current training status for chat tab and get model info.
    
    Args:
        current_state: Current application state
        
    Returns:
        Tuple of (status_message, base_model_info, trained_model_info)
    """
    disk_state = load_state_from_disk()
    
    # DEBUG: Log what we received vs what's on disk
    print(f"[CHAT DEBUG] Incoming state: training_complete={current_state.get('training_complete')}, "
          f"training_active={current_state.get('training_active')}, output_dir={current_state.get('output_dir')}")
    if disk_state:
        print(f"[CHAT DEBUG] Disk state: training_complete={disk_state.get('training_complete')}, "
          f"training_active={disk_state.get('training_active')}, output_dir={disk_state.get('output_dir')}")
    else:
        print(f"[CHAT DEBUG] No disk state found")
    
    # ALWAYS prioritize disk state over component state (disk state is source of truth)
    if disk_state:
        # If disk says complete, use disk values (it has the truth)
        if disk_state.get('training_complete'):
            print(f"[CHAT DEBUG] Using disk state - training is complete")
            current_state = {
                **current_state,
                'training_complete': True,
                'training_active': disk_state.get('training_active', False),
                'output_dir': disk_state.get('output_dir', current_state.get('output_dir')),
                'completion_time': disk_state.get('completion_time'),
            }
        # If disk says active but component says inactive, trust disk (training might be running in background)
        elif disk_state.get('training_active') and not current_state.get('training_active'):
            print(f"[CHAT DEBUG] Using disk state - training is active")
            current_state = {
                **current_state,
                'training_active': True,
                'output_dir': disk_state.get('output_dir', current_state.get('output_dir')),
            }
    
    # Final debug log
    print(f"[REFRESH] Status: complete={current_state.get('training_complete')}, "
          f"active={current_state.get('training_active')}, dir={current_state.get('output_dir')}")
    # Check if we're using base model only (no training)
    chat_model_id = current_state.get('chat_model_id')
    has_training_config = bool(current_state.get('training_config', {}).get('model_id'))
    
    # Get model info from config or chat selection
    if has_training_config:
        config = current_state.get('training_config', {})
        base_model_id = config.get('model_id', 'Unknown')
        base_model_name = config.get('model_name', 'Unknown Model')
    elif chat_model_id and chat_model_id != "__base__":
        # Try to get model info from STUDIO_MODELS
        try:
            from edukaai_studio.config import STUDIO_MODELS
            model = STUDIO_MODELS.get_model(chat_model_id)
            if model:
                base_model_id = model.get('model_id', chat_model_id)
                base_model_name = model.get('name', 'Unknown Model')
            else:
                base_model_id = chat_model_id
                base_model_name = chat_model_id
        except Exception:
            base_model_id = chat_model_id
            base_model_name = chat_model_id
    else:
        base_model_id = 'Not selected'
        base_model_name = 'No model selected'
    
    base_model_info = f"{base_model_name}\n({base_model_id})"
    
    # Check for fine-tuned model
    trained_model_info = "Not available"
    
    if current_state.get('training_complete'):
        output_dir = current_state.get('output_dir', '')
        
        if not output_dir or not Path(output_dir).exists():
            print(f"[CHAT] Output dir from state not found: {output_dir}")
            print(f"[CHAT] Searching for actual output directory in outputs/...")
            
            outputs_base = Path("outputs")
            if outputs_base.exists():
                for subdir in outputs_base.iterdir():
                    if subdir.is_dir():
                        has_adapters = (subdir / "adapters" / "adapters.safetensors").exists() or \
                                      (subdir / "best_adapter" / "adapters.safetensors").exists()
                        
                        if has_adapters:
                            output_dir = str(subdir)
                            print(f"[CHAT] OK: Found valid output directory: {output_dir}")
                            current_state['output_dir'] = output_dir
                            save_state_to_disk(current_state)
                            break
        
        if output_dir and Path(output_dir).exists():
            # Check if fused model exists
            fused_path = Path(output_dir) / "fused_model"
            if fused_path.exists():
                trained_model_info = f"Fused Model\n({output_dir}/fused_model)"
            if output_dir and Path(output_dir).exists():
                # Check for adapter
                adapter_paths = [
                    Path(output_dir) / "adapters" / "adapters.safetensors",
                    Path(output_dir) / "best_adapter" / "adapters.safetensors",
                ]
                adapter_found = None
                for ap in adapter_paths:
                    if ap.exists():
                        adapter_found = ap
                        break
                
                if adapter_found:
                    trained_model_info = f"LoRA Adapter\n({adapter_found})"
                else:
                    trained_model_info = f"Output exists but no model found\n({output_dir})"
                
                adapter_exists = (Path(output_dir) / "adapters" / "adapters.safetensors").exists() or \
                               (Path(output_dir) / "best_adapter" / "adapters.safetensors").exists()
            else:
                adapter_exists = False
                trained_model_info = f"Output directory not found\n({output_dir})"
            if adapter_exists:
                return "OK: Ready! Fine-tuned model available.", base_model_info, trained_model_info
            else:
                return "Warning: Training complete but adapter not found. Only base model available.", base_model_info, trained_model_info
        else:
            return "OK: Training complete! (Output directory not found - try refreshing)", base_model_info, trained_model_info
    elif current_state.get('training_active'):
        return "Loading: Training in progress... Check Train tab for details.", base_model_info, trained_model_info
    elif has_training_config or (chat_model_id and chat_model_id != "__base__"):
        # Using base model only
        return "OK: Using base model only. Select 'Use Base Model Only' from dropdown above to chat without training.", base_model_info, trained_model_info
    else:
        return "Warning: No model selected. Please select a model from the dropdowns above to start chatting.", base_model_info, trained_model_info


def ask_models(
    question: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    use_fused: bool,
    current_state: Dict[str, Any]
) -> Tuple:
    """Enhanced chat function with parameter controls and both model comparison.
    
    Args:
        question: User's question
        system_prompt: System prompt for the models
        temperature: Temperature for generation
        top_p: Top-p for nucleus sampling
        max_tokens: Maximum tokens to generate
        use_fused: Whether to use fused model instead of base+adapter
        current_state: Current application state
        
    Returns:
        Tuple of outputs for Gradio components
    """
    global conversation_history
    
    if not question or not question.strip():
        return (
            "No messages yet." if not conversation_history else "\n".join(conversation_history),
            "*No question provided*",
            "*No question provided*",
            "Stats: -",
            "Stats: -",
            "Tokens: Input 0 | Output 0 | Total 0",
            "Please enter a message to start the conversation."
        )
    
    # Check disk state
    disk_state = load_state_from_disk()
    is_complete = current_state.get('training_complete') or (disk_state.get('training_complete') if disk_state else False)
    output_dir = current_state.get('output_dir') or (disk_state.get('output_dir') if disk_state else None)
    
    # Auto-discover if needed
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
    
    # Check if we're using base model only (no training required)
    chat_model_id = current_state.get('chat_model_id')
    using_base_only = chat_model_id == "__base__" or (
        not is_complete and 
        not output_dir and 
        current_state.get('training_config', {}).get('model_id')
    )
    
    if not is_complete and not using_base_only:
        return (
            "No messages yet." if not conversation_history else "\n".join(conversation_history),
            "Training not complete. Please complete training first or select 'Use Base Model Only'.",
            "Training not complete. Please complete training first or select 'Use Base Model Only'.",
            "Stats: -",
            "Stats: -",
            "Tokens: Input 0 | Output 0 | Total 0",
            "Warning: Training must be completed before using chat, or select a base model from the dropdowns above."
        )
    
    config = current_state.get('training_config', {})
    model_id = config.get('model_id')
    
    # If using base model only and no training config, try to get from chat_model_id
    if not model_id and using_base_only:
        # Check if we have a base model selected in state
        model_id = current_state.get('chat_model_id')
        if model_id and model_id != "__base__":
            print(f"[CHAT] Using base model from chat_model_id: {model_id}")
        else:
            # Fallback to default model
            from edukaai_studio.config import STUDIO_MODELS
            default_model = STUDIO_MODELS.get_model(STUDIO_MODELS.DEFAULT_STUDIO_MODEL)
            model_id = default_model.get('model_id', 'mlx-community/Phi-3-mini-4k_instruct-4bit')
            print(f"[CHAT] Using default base model: {model_id}")
    
    # Try to read actual model from adapter config
    if not model_id and output_dir:
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
                except Exception:
                    pass
    
    # Final fallback
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
        tuned_base_model = None
        
        # Read adapter_config.json to get correct base model
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
        
        # Use correct base model
        if tuned_base_model:
            actual_base_model = tuned_base_model
            print(f"[CHAT] Using adapter's base model: {actual_base_model}")
        else:
            actual_base_model = model_id
            print(f"[CHAT] Using current model as base: {actual_base_model}")
        
        if use_fused and output_dir:
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
        status_msg = f"Error: {str(e)[:100]}"
    
    return (
        "\n".join(conversation_history),
        base_resp,
        tuned_resp,
        base_stats,
        tuned_stats,
        token_info,
        status_msg
    )


def clear_conversation() -> Tuple:
    """Clear the conversation history.
    
    Returns:
        Tuple of reset values for chat components
    """
    global conversation_history
    conversation_history = []
    return (
        "No messages yet.",
        "*Conversation cleared*",
        "*Conversation cleared*",
        "Stats: -",
        "Stats: -",
        "Tokens: Input 0 | Output 0 | Total 0",
        "Conversation history cleared."
    )


def compare_models(
    question: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    current_state: Dict[str, Any]
) -> str:
    """Run the same question multiple times to compare consistency.
    
    Args:
        question: User's question
        system_prompt: System prompt
        temperature: Temperature
        top_p: Top-p
        max_tokens: Max tokens
        current_state: Current application state
        
    Returns:
        Comparison results as string
    """
    if not question or not question.strip():
        return "Please enter a question to compare"
    
    results = []
    for i in range(3):
        try:
            config = current_state.get('training_config', {})
            output_dir = current_state.get('output_dir')
            
            model_id = config.get('model_id')
            if not model_id and output_dir:
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
                        except Exception:
                            pass
            
            if not model_id:
                default_model = STUDIO_MODELS.get_model(STUDIO_MODELS.DEFAULT_STUDIO_MODEL)
                model_id = default_model.get('model_id', 'mlx-community/Phi-3-mini-4k_instruct-4bit')
                print(f"[COMPARE] Using default model: {model_id}")
            
            wrapper = ChatWrapper(model_id)
            resp = wrapper.chat(
                question.strip(),
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=int(max_tokens)
            )
            results.append(f"**Run {i+1}:**\n{resp[:200]}...")
        except Exception as e:
            results.append(f"**Run {i+1}:** Error - {str(e)}")
    
    return "\n\n".join(results)


def create_chat_tab(state: gr.State, tabs: gr.Tabs) -> Dict[str, Any]:
    """Create the Chat tab.
    
    Args:
        state: Gradio state object
        tabs: Gradio tabs container
        
    Returns:
        Dictionary of component references
    """
    components = {}
    
    with gr.TabItem("5. Chat") as chat_tab:
        gr.Markdown("## Step 5: Chat and Compare")
        gr.Markdown("Test your fine-tuned model against the base model with advanced controls")
        
        components['chat_status'] = gr.Textbox(
            label="Status",
            value="Training must be complete to use chat",
            interactive=False
        )
        
        # Model Information Section
        with gr.Accordion("Model Information", open=True):
            # NEW: Model selector dropdown
            with gr.Row():
                components['model_selector'] = gr.Dropdown(
                    label="Select Fine-tuned Model",
                    choices=[
                        ("⏳ Click Refresh to load models", "__loading__")
                    ],
                    value="__loading__",
                    interactive=True,
                    info="Choose any of your trained models to chat with"
                )
                components['refresh_models_btn'] = gr.Button(
                    "🔄 Refresh Models",
                    size="sm"
                )
            
            # NEW: Base model selector (only used when "Use Base Model Only" is selected)
            with gr.Row():
                components['base_model_selector'] = gr.Dropdown(
                    label="Select Base Model (for base-only chat)",
                    choices=[("⏳ Click Refresh to load base models", "__loading__")],
                    value="__loading__",
                    interactive=True,
                    info="Select which base model to use when not using a fine-tuned model"
                )
                components['refresh_base_models_btn'] = gr.Button(
                    "🔄 Refresh Base Models",
                    size="sm"
                )
            
            with gr.Row():
                with gr.Column():
                    components['base_model_info'] = gr.Textbox(
                        label="Base Model",
                        value="Not loaded",
                        interactive=False,
                        lines=2
                    )
                with gr.Column():
                    components['trained_model_info'] = gr.Textbox(
                        label="Selected Model",
                        value="Not available",
                        interactive=False,
                        lines=2
                    )
        
        # Advanced Parameters Section
        with gr.Accordion("Advanced Parameters", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    components['system_prompt'] = gr.Textbox(
                        label="System Prompt",
                        value="You are a helpful assistant. Answer the user's question accurately and concisely.",
                        lines=3,
                        placeholder="Define the model's behavior and personality..."
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        components['temperature'] = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=0.7,
                            step=0.1,
                            info="Lower = more deterministic, Higher = more creative"
                        )
                    with gr.Row():
                        components['top_p'] = gr.Slider(
                            label="Top-p (Nucleus Sampling)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            info="Probability threshold for token selection"
                        )
                    with gr.Row():
                        components['max_tokens'] = gr.Slider(
                            label="Max Tokens",
                            minimum=10,
                            maximum=2000,
                            value=500,
                            step=10,
                            info="Maximum response length"
                        )
                    with gr.Row():
                        components['use_fused'] = gr.Checkbox(
                            label="Use Fused Model",
                            value=False,
                            info="Load fused model instead of base+adapter (faster, more memory)"
                        )
        
        # Token Counter Display
        with gr.Row():
            components['token_info'] = gr.Textbox(
                label="Token Information",
                value="Tokens: Input 0 | Output 0 | Total 0",
                interactive=False
            )
        
        # Conversation History
        with gr.Accordion("Conversation History", open=False):
            components['conversation_display'] = gr.Textbox(
                label="Conversation Log",
                lines=10,
                max_lines=50,
                value="Conversation history will appear here...",
                interactive=False
            )
        
        # Input Area
        with gr.Row():
            components['question'] = gr.Textbox(
                label="Your Message",
                placeholder="Type your question or message here...",
                lines=2,
                scale=4
            )
            components['send_btn'] = gr.Button("Send", variant="primary", scale=1)
        
        # Model Response Display
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Model (Original)")
                components['base_response'] = gr.Textbox(
                    label="Base Model Response",
                    value="Base model response will appear here",
                    lines=8,
                    interactive=False
                )
                components['base_stats'] = gr.Textbox(
                    label="Base Model Stats",
                    value="Stats: -",
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("### Fine-Tuned Model (Trained)")
                components['tuned_response'] = gr.Textbox(
                    label="Fine-Tuned Model Response",
                    value="Fine-tuned model response will appear here",
                    lines=8,
                    interactive=False
                )
                components['tuned_stats'] = gr.Textbox(
                    label="Fine-Tuned Model Stats",
                    value="Stats: -",
                    interactive=False
                )
        
        with gr.Row():
            components['refresh_chat_btn'] = gr.Button("Refresh Status", variant="secondary", size="sm")
            components['clear_history_btn'] = gr.Button("Clear History", variant="secondary", size="sm")
            components['compare_btn'] = gr.Button("Compare Models", variant="secondary", size="sm")
        
        # Wire up event handlers
        components['send_btn'].click(
            fn=ask_models,
            inputs=[
                components['question'],
                components['system_prompt'],
                components['temperature'],
                components['top_p'],
                components['max_tokens'],
                components['use_fused'],
                state
            ],
            outputs=[
                components['conversation_display'],
                components['base_response'],
                components['tuned_response'],
                components['base_stats'],
                components['tuned_stats'],
                components['token_info'],
                components['chat_status']
            ]
        )
        
        components['clear_history_btn'].click(
            fn=clear_conversation,
            inputs=[],
            outputs=[
                components['conversation_display'],
                components['base_response'],
                components['tuned_response'],
                components['base_stats'],
                components['tuned_stats'],
                components['token_info'],
                components['chat_status']
            ]
        )
        
        components['compare_btn'].click(
            fn=compare_models,
            inputs=[
                components['question'],
                components['system_prompt'],
                components['temperature'],
                components['top_p'],
                components['max_tokens'],
                state
            ],
            outputs=[components['base_response']]
        )
        
        components['refresh_chat_btn'].click(
            fn=refresh_chat_status,
            inputs=[state],
            outputs=[
                components['chat_status'],
                components['base_model_info'],
                components['trained_model_info']
            ]
        )
        
        # Auto-refresh status when Chat tab is selected
        chat_tab.select(
            fn=refresh_chat_status,
            inputs=[state],
            outputs=[
                components['chat_status'],
                components['base_model_info'],
                components['trained_model_info']
            ]
        )
        
        # NEW: Refresh models dropdown
        def get_available_models():
            """Get list of available trained models."""
            try:
                from edukaai_studio.core.trained_models_registry import get_registry
                registry = get_registry()
                models = registry.list_models()
                
                choices = [("📦 Use Base Model Only", "__base__")]
                
                for model in models[:20]:  # Limit to 20 most recent
                    created = model.created_at[:10] if len(model.created_at) >= 10 else model.created_at
                    label = f"📅 {created} - {model.base_model_name[:20]} ({model.iterations}it, loss {model.best_loss:.3f})"
                    choices.append((label, model.id))
                
                if not models:
                    choices = [("⚠️ No trained models found. Train a model first!", "__none__")]
                
                return gr.update(choices=choices, value=choices[0][1] if choices else "__base__")
            except Exception as e:
                return gr.update(
                    choices=[("⚠️ Error loading models", "__error__")],
                    value="__error__"
                )
        
        def get_available_base_models():
            """Get list of available base models from config."""
            try:
                from edukaai_studio.config import STUDIO_MODELS
                
                models = STUDIO_MODELS.get_all_models()
                choices = []
                
                for model in models:
                    model_id = model.get('id')
                    model_name = model.get('name', 'Unknown')
                    choices.append((model_name, model_id))
                
                return gr.update(choices=choices, value=choices[0][1] if choices else None)
            except Exception as e:
                print(f"[CHAT] Error loading base models: {e}")
                return gr.update(
                    choices=[("⚠️ Error loading base models", "__error__")],
                    value="__error__"
                )
        
        def on_model_selected(model_id: str, base_model_id: str, current_state: Dict[str, Any]) -> Tuple[str, str, str, str, Dict[str, Any]]:
            """Handle model selection from dropdown."""
            if model_id == "__base__":
                # Use base model only - update state with selected base model
                try:
                    from edukaai_studio.config import STUDIO_MODELS
                    model = STUDIO_MODELS.get_model(base_model_id)
                    
                    if model:
                        # Update state with base model info
                        current_state = {
                            **current_state,
                            'training_config': {
                                'model_id': model.get('model_id'),
                                'model_name': model.get('name'),
                            },
                            'chat_model_id': base_model_id,
                            'chat_model_name': model.get('name'),
                            'chat_model_path': None,
                            'chat_adapter_path': None,
                        }
                        
                        return (
                            f"✅ Using base model: {model.get('name')}",
                            model.get('name'),
                            "None (Base model only)",
                            "Ready - No training required!",
                            current_state
                        )
                    else:
                        return (
                            "⚠️ Base model not found",
                            "Unknown",
                            "None",
                            "Please select a valid base model",
                            current_state
                        )
                except Exception as e:
                    return (
                        f"⚠️ Error loading base model: {e}",
                        "Error",
                        "None",
                        "Error occurred",
                        current_state
                    )
            
            if model_id in ["__none__", "__error__", "__loading__"]:
                return (
                    "⚠️ Please select a valid model",
                    "Not available",
                    "Not available",
                    "Select a model to chat",
                    current_state
                )
            
            try:
                from edukaai_studio.core.trained_models_registry import get_registry
                registry = get_registry()
                model = registry.get_model(model_id)
                
                if not model:
                    return "Model not found", "Error", "Error", "Error", current_state
                
                # Update state with model info
                current_state = {
                    **current_state,
                    'training_config': {
                        'model_id': model.base_model_id,
                        'model_name': model.base_model_name,
                    },
                    'chat_model_id': model_id,
                    'chat_model_path': model.output_dir,
                    'chat_model_name': model.base_model_name,
                    'chat_adapter_path': model.exports.get('adapter'),
                    'training_complete': True,
                    'output_dir': model.output_dir,
                }
                
                info_text = f"""{model.base_model_name}
Iterations: {model.iterations}
Best Loss: {model.best_loss:.4f}
Dataset: {Path(model.dataset_path).name}"""
                
                return (
                    f"✅ Loaded: {model.base_model_name}",
                    model.base_model_id,
                    info_text,
                    "Ready to chat",
                    current_state
                )
            except Exception as e:
                return f"Error: {e}", "Error", "Error", "Error", current_state
        
        # Wire up model selector
        components['refresh_models_btn'].click(
            fn=get_available_models,
            outputs=[components['model_selector']]
        )
        
        components['refresh_base_models_btn'].click(
            fn=get_available_base_models,
            outputs=[components['base_model_selector']]
        )
        
        components['model_selector'].change(
            fn=on_model_selected,
            inputs=[components['model_selector'], components['base_model_selector'], state],
            outputs=[
                components['chat_status'],
                components['base_model_info'],
                components['trained_model_info'],
                components['chat_status'],  # Update twice for clarity
                state
            ]
        )
        
        # Also trigger when base model selector changes (if using base-only mode)
        components['base_model_selector'].change(
            fn=on_model_selected,
            inputs=[components['model_selector'], components['base_model_selector'], state],
            outputs=[
                components['chat_status'],
                components['base_model_info'],
                components['trained_model_info'],
                components['chat_status'],
                state
            ]
        )
    
    return components
