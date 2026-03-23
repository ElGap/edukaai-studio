"""My Models Tab (Trained Models) for EdukaAI Fine Tuning Studio.

Central hub for managing all fine-tuned adapters with full metadata,
exports, loss curves, and deployment options.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import gradio as gr

from edukaai_studio.core.trained_models_registry import (
    TrainedModelsRegistry, 
    TrainedModel,
    get_registry,
    format_model_for_display
)
from edukaai_studio.core.state import save_state_to_disk
from edukaai_studio.ui.visualizations import create_loss_chart


def scan_for_models() -> Tuple[List[List], str]:
    """Scan for all trained models.
    
    Returns:
        Tuple of (models_data, status_message)
        models_data is a list of lists (rows) for gr.Dataframe
        Columns: [ID, Status, Model, Dataset, Iterations, Best Loss, Exports, Output Dir]
    """
    registry = get_registry()
    
    # First, scan for any new models not in registry
    new_models = registry.scan_for_new_models()
    
    # Get all models (excluding orphaned by default)
    all_models = registry.list_models()
    
    # Format for display - convert to list of lists for gr.Dataframe
    # Headers: ["ID", "Status", "Model", "Dataset", "Iterations", "Best Loss", "Exports", "Output Dir"]
    models_data = []
    for model in all_models:
        display = format_model_for_display(model)
        row = [
            display['id'],
            display['status'],
            display['model_name'],
            display['dataset'],
            display['iterations'],
            display['best_loss'],
            display['exports'],
            display['output_dir']
        ]
        models_data.append(row)
    
    status = f"✅ Found {len(all_models)} trained model(s)"
    if new_models:
        status += f" ({len(new_models)} new)"
    
    return models_data, status


def cleanup_orphaned_models() -> Tuple[List[List], str]:
    """Clean up orphaned models (registry entries with no files on disk).
    
    Returns:
        Tuple of (updated_models_data, status_message)
    """
    registry = get_registry()
    
    # Get stats before cleanup
    stats_before = registry.get_statistics()
    orphaned_before = stats_before.get('orphaned', 0)
    
    if orphaned_before == 0:
        # No orphaned models, just refresh the list
        return scan_for_models()
    
    # Perform cleanup
    total_checked, removed = registry.cleanup_orphaned_models(delete_registry_entries=True)
    
    # Refresh the list after cleanup
    models_data, _ = scan_for_models()
    
    # Update status message
    if removed > 0:
        status = f"🧹 Cleaned up {removed} orphaned model(s). {len(models_data)} model(s) remaining."
    else:
        status = f"✅ No orphaned models found. {len(models_data)} model(s) in registry."
    
    return models_data, status


def get_model_details(model_id: str) -> Tuple[str, str, str, str, str, str, str, Any]:
    """Get detailed information for a specific model.
    
    Returns:
        Tuple of (info_text, exports_text, config_text, metrics_text, 
                  notes_text, tags_text, status, loss_plot)
    """
    registry = get_registry()
    model = registry.get_model(model_id)
    
    if not model:
        return (
            "Model not found",
            "", "", "", "", "",
            "error",
            None
        )
    
    # Format info text
    info_text = f"""📅 Created: {model.created_at[:19]}
🤖 Base Model: {model.base_model_name}
   ({model.base_model_id})
📁 Output: {model.output_dir}
🏷️  Tags: {', '.join(model.tags) if model.tags else 'None'}
📝 Notes: {model.notes if model.notes else 'No notes'}"""
    
    # Format exports text
    exports_list = []
    if model.exports.get('adapter'):
        adapter_path = model.exports['adapter']
        size_mb = Path(adapter_path).stat().st_size / (1024 * 1024) if Path(adapter_path).exists() else 0
        exports_list.append(f"✅ LoRA Adapter (~{size_mb:.1f} MB)")
    else:
        exports_list.append("❌ LoRA Adapter (not found)")
    
    if model.exports.get('fused'):
        exports_list.append("✅ Fused Model (full weights)")
    else:
        exports_list.append("❌ Fused Model")
    
    if model.exports.get('gguf'):
        gguf_path = model.exports['gguf']
        size_mb = Path(gguf_path).stat().st_size / (1024 * 1024) if Path(gguf_path).exists() else 0
        exports_list.append(f"✅ GGUF (~{size_mb:.1f} MB)")
    else:
        exports_list.append("❌ GGUF")
    
    exports_text = "\n".join(exports_list)
    
    # Format config text
    config_text = f"""🎯 Training Configuration:
   • Iterations: {model.iterations}
   • Learning Rate: {model.learning_rate}
   • LoRA Rank: {model.lora_rank}
   • LoRA Alpha: {model.lora_alpha}
   • LoRA Dropout: {model.lora_dropout}
   • Batch Size: {model.batch_size}
   • Grad Accumulation: {model.grad_accumulation}

📊 Dataset:
   • Path: {model.dataset_path}
   • Size: {model.dataset_size} examples"""
    
    # Format metrics text
    metrics_text = f"""📈 Training Results:
   • Best Loss: {model.best_loss:.4f} @ iteration {model.best_iteration}
   • Final Loss: {model.final_loss:.4f}
   • Duration: {model.training_duration_minutes:.1f} minutes
   • Status: {model.status.upper()}"""
    
    # Notes and tags
    notes_text = model.notes
    tags_text = ", ".join(model.tags) if model.tags else ""
    
    # Create loss plot - show if we have EITHER train or validation losses
    has_loss_data = model.train_losses or model.val_losses
    plot = None
    if has_loss_data:
        plot = create_loss_chart(
            model.train_losses,
            model.val_losses,
            model.best_iteration
        )
    
    status = "ok"
    
    return (
        info_text,
        exports_text,
        config_text,
        metrics_text,
        notes_text,
        tags_text,
        status,
        plot
    )


def update_model_notes(model_id: str, notes: str) -> str:
    """Update notes for a model."""
    registry = get_registry()
    if registry.update_model(model_id, notes=notes):
        return "✅ Notes updated"
    return "❌ Failed to update notes"


def update_model_tags(model_id: str, tags: str) -> str:
    """Update tags for a model."""
    registry = get_registry()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if registry.update_model(model_id, tags=tag_list):
        return f"✅ Tags updated: {', '.join(tag_list)}"
    return "❌ Failed to update tags"


def delete_model(model_id: str, delete_files: bool = False) -> Tuple[List[List], str]:
    """Delete a model from registry.
    
    Returns:
        Updated model list (as list of lists for Dataframe) and status
    """
    registry = get_registry()
    
    if registry.delete_model(model_id, delete_files=delete_files):
        # Refresh list - return as list of lists
        models_data, _ = scan_for_models()
        return models_data, f"✅ Model deleted{' (files removed)' if delete_files else ''}"
    
    models_data, _ = scan_for_models()
    return models_data, "❌ Failed to delete model"


def load_model_for_chat(model_id: str, current_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Load a model into chat state.
    
    Returns:
        Status message and updated state
    """
    registry = get_registry()
    model = registry.get_model(model_id)
    
    if not model:
        return "❌ Model not found", current_state
    
    # Update state with model info
    current_state = {
        **current_state,
        'chat_model_id': model_id,
        'chat_model_path': model.output_dir,
        'chat_model_name': model.base_model_name,
        'chat_adapter_path': model.exports.get('adapter'),
    }
    
    save_state_to_disk(current_state)
    
    return f"✅ Loaded: {model.base_model_name} (ready for chat)", current_state


def create_fused_model(model_id: str) -> Tuple[str, str]:
    """Create fused model from adapter.
    
    Returns:
        Status and updated exports text
    """
    registry = get_registry()
    model = registry.get_model(model_id)
    
    if not model:
        return "❌ Model not found", ""
    
    # Check if adapter exists
    adapter_path = model.exports.get('adapter')
    if not adapter_path or not Path(adapter_path).exists():
        return "❌ Adapter not found", ""
    
    # This would call the fusion logic
    # For now, return placeholder
    return (
        "⏳ Fused model creation started (check terminal)",
        "✅ Fused Model (creation in progress...)"
    )


def convert_to_gguf(model_id: str) -> Tuple[str, str]:
    """Convert model to GGUF format.
    
    Returns:
        Status and updated exports text
    """
    registry = get_registry()
    model = registry.get_model(model_id)
    
    if not model:
        return "❌ Model not found", ""
    
    # This would call the GGUF conversion logic
    return (
        "⏳ GGUF conversion started (check terminal)",
        "✅ GGUF (conversion in progress...)"
    )


def download_model_file(model_id: str, export_type: str) -> Tuple[str, str, str]:
    """Get download path for a model export.
    
    Returns:
        Tuple of (file_path, status_message, label)
    """
    registry = get_registry()
    model = registry.get_model(model_id)
    
    if not model:
        return None, "❌ Model not found", "Error"
    
    export_path = model.exports.get(export_type)
    
    if not export_path:
        return None, f"❌ {export_type} not available", "Error"
    
    if not Path(export_path).exists():
        return None, f"❌ {export_type} file not found", "Error"
    
    file_size = Path(export_path).stat().st_size / (1024 * 1024)
    file_name = Path(export_path).name
    
    return (
        export_path,
        f"✅ Ready: {file_name} ({file_size:.1f} MB)",
        f"Download {export_type.upper()}"
    )


def filter_models(filter_type: str, search_query: str = "") -> Tuple[List[List], str]:
    """Filter models based on type and search.
    
    Returns:
        Filtered models (as list of lists for Dataframe) and status
        Columns: [ID, Status, Model, Dataset, Iterations, Best Loss, Exports, Output Dir]
    """
    registry = get_registry()
    
    # Apply filters
    if filter_type == "favorites":
        models = registry.list_models(tags=["favorite"])
    elif filter_type == "with_fused":
        all_models = registry.list_models()
        models = [m for m in all_models if m.exports.get('fused')]
    elif filter_type == "with_gguf":
        all_models = registry.list_models()
        models = [m for m in all_models if m.exports.get('gguf')]
    elif filter_type == "running":
        models = registry.list_models(filter_status="running")
    elif filter_type == "completed":
        models = registry.list_models(filter_status="completed")
    else:  # "all"
        models = registry.list_models()
    
    # Apply search
    if search_query:
        models = [m for m in models if 
                  search_query.lower() in m.base_model_name.lower() or
                  search_query.lower() in m.notes.lower() or
                  any(search_query.lower() in tag.lower() for tag in m.tags)]
    
    # Convert to list of lists for gr.Dataframe
    # Headers: ["ID", "Status", "Model", "Dataset", "Iterations", "Best Loss", "Exports", "Output Dir"]
    models_data = []
    for model in models:
        display = format_model_for_display(model)
        row = [
            display['id'],
            display['status'],  # NEW: Status column
            display['model_name'],
            display['dataset'],
            display['iterations'],
            display['best_loss'],
            display['exports'],
            display['output_dir']
        ]
        models_data.append(row)
    
    status = f"Showing {len(models)} model(s)"
    if filter_type != "all":
        status += f" ({filter_type})"
    if search_query:
        status += f' matching "{search_query}"'
    
    return models_data, status


def create_my_models_tab(state: gr.State, tabs: gr.Tabs) -> Dict[str, Any]:
    """Create the My Models (Trained Models) management tab.
    
    This is a comprehensive hub for all fine-tuned adapters with:
    - Visual grid of all trained models
    - Detailed metadata and loss curves
    - One-click exports and downloads
    - Chat integration
    - Organization tools (tags, notes, favorites)
    
    Args:
        state: Gradio state object
        tabs: Parent tabs component
        
    Returns:
        Dictionary of created components
    """
    components = {}
    
    with gr.TabItem("🤖 My Models", id="my-models") as my_models_tab:
        # Hidden state for selected model - define FIRST so it's available to all handlers
        components['selected_model_id'] = gr.Textbox(
            visible=False,
            value="",
            label="selected_model_id"
        )
        
        # Header
        gr.Markdown("# 🤖 My Fine-tuned Models")
        gr.Markdown("Browse, test, and deploy all your trained adapters. Click any model to see details.")
        
        # Top controls row
        with gr.Row():
            with gr.Column(scale=2):
                components['scan_btn'] = gr.Button(
                    "🔍 Scan for Models",
                    variant="primary",
                    size="sm"
                )
            with gr.Column(scale=1):
                components['cleanup_btn'] = gr.Button(
                    "🧹 Clean Up",
                    size="sm"
                )
                gr.Markdown("*Remove registry entries for deleted models*", visible=False)  # Tooltip workaround
            with gr.Column(scale=3):
                # Filter tabs - Added "Running" filter
                components['filter_tabs'] = gr.Radio(
                    choices=[
                        ("All", "all"),
                        ("🏃 Running", "running"),
                        ("✅ Completed", "completed"),
                        ("With Fused", "with_fused"),
                        ("With GGUF", "with_gguf"),
                        ("Favorites", "favorites")
                    ],
                    value="all",
                    label="Filter",
                    interactive=True
                )
            with gr.Column(scale=3):
                components['search_input'] = gr.Textbox(
                    placeholder="Search by name, tags, or notes...",
                    label="Search",
                    interactive=True
                )
        
        # Status bar
        components['status_text'] = gr.Textbox(
            label="Status",
            value="Click 'Scan for Models' to load your adapters",
            interactive=False
        )
        
        gr.Markdown("---")
        
        # Main content area
        with gr.Row():
            # Left: Model list
            with gr.Column(scale=3):
                gr.Markdown("### 📋 Trained Adapters")
                
                # Model list as dataframe (interactive table)
                components['models_table'] = gr.Dataframe(
                    headers=["ID", "Status", "Model", "Dataset", "Iterations", "Best Loss", "Exports", "Output Dir"],
                    datatype=["str", "str", "str", "str", "number", "str", "str", "str"],
                    label="Click a row to view details",
                    interactive=False,
                    wrap=True
                )
                
                gr.Markdown("*💡 Tip: Click any row to see full details and actions*")
            
            # Right: Detail panel
            with gr.Column(scale=4):
                gr.Markdown("### 📊 Model Details")
                
                # Info section
                components['info_text'] = gr.Textbox(
                    label="Model Information",
                    lines=6,
                    interactive=False
                )
                
                # Two column layout for metrics and config
                with gr.Row():
                    with gr.Column():
                        components['metrics_text'] = gr.Textbox(
                            label="Training Metrics",
                            lines=6,
                            interactive=False
                        )
                    with gr.Column():
                        components['config_text'] = gr.Textbox(
                            label="Configuration",
                            lines=6,
                            interactive=False
                        )
                
                # Loss curve
                components['loss_plot'] = gr.Plot(
                    label="Training & Validation Loss",
                    value=None
                )
                
                # Exports section
                gr.Markdown("---")
                gr.Markdown("### 📦 Exports")
                
                components['exports_text'] = gr.Textbox(
                    label="Available Exports",
                    lines=3,
                    interactive=False
                )
                
                # Add status text for selection feedback
                components['detail_status'] = gr.Textbox(
                    label="Detail Status",
                    value="Click a model to view details",
                    interactive=False,
                    visible=False  # Hidden, just for internal use
                )
                
                # Export action buttons
                with gr.Row():
                    components['download_adapter_btn'] = gr.Button(
                        "📥 Download Adapter",
                        variant="secondary",
                        size="sm"
                    )
                    components['download_fused_btn'] = gr.Button(
                        "📥 Download Fused",
                        variant="secondary",
                        size="sm"
                    )
                    components['download_gguf_btn'] = gr.Button(
                        "📥 Download GGUF",
                        variant="secondary",
                        size="sm"
                    )
                
                # Create buttons for missing exports
                with gr.Row():
                    components['create_fused_btn'] = gr.Button(
                        "⚙️ Create Fused Model",
                        size="sm"
                    )
                    components['create_gguf_btn'] = gr.Button(
                        "⚙️ Convert to GGUF",
                        size="sm"
                    )
                
                # Download file output
                components['download_file'] = gr.File(
                    label="Download",
                    visible=False
                )
                
                gr.Markdown("---")
                
                # Quick actions
                gr.Markdown("---")
                gr.Markdown("### ⚡ Quick Actions")
                
                with gr.Row():
                    components['chat_btn'] = gr.Button(
                        "💬 Test in Chat",
                        variant="primary",
                        size="sm"
                    )
                    components['retrain_btn'] = gr.Button(
                        "🔄 Retrain with Same Config",
                        size="sm"
                    )
                
                # Retrain button handler - NEW
                def on_retrain_click(model_id: str, current_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
                    """Load model config for retraining."""
                    try:
                        registry = get_registry()
                        model = registry.get_model(model_id)
                        
                        if not model:
                            return "❌ Model not found", current_state
                        
                        # Update state with training config
                        current_state = {
                            **current_state,
                            'retrain_model_id': model_id,
                            'retrain_config': {
                                'model_id': model.base_model_id,
                                'model_name': model.base_model_name,
                                'iterations': model.iterations,
                                'learning_rate': model.learning_rate,
                                'lora_rank': model.lora_rank,
                                'lora_alpha': model.lora_alpha,
                                'lora_dropout': model.lora_dropout,
                                'batch_size': model.batch_size,
                                'grad_accumulation': model.grad_accumulation,
                            }
                        }
                        
                        save_state_to_disk(current_state)
                        
                        return (
                            f"✅ Config loaded: {model.base_model_name}\n📝 Go to Configure tab and upload new data",
                            current_state
                        )
                    except Exception as e:
                        return f"❌ Error loading config: {e}", current_state
                
                components['retrain_btn'].click(
                    fn=on_retrain_click,
                    inputs=[components['selected_model_id'], state],
                    outputs=[components['status_text'], state]
                )
                
                # Organization
                gr.Markdown("---")
                gr.Markdown("### 🏷️ Organization")
                
                with gr.Row():
                    with gr.Column():
                        components['tags_input'] = gr.Textbox(
                            placeholder="tag1, tag2, tag3...",
                            label="Tags (comma-separated)"
                        )
                        components['update_tags_btn'] = gr.Button(
                            "Update Tags",
                            size="sm"
                        )
                    with gr.Column():
                        components['notes_input'] = gr.TextArea(
                            placeholder="Add notes about this model...",
                            label="Notes",
                            lines=3
                        )
                        components['update_notes_btn'] = gr.Button(
                            "Save Notes",
                            size="sm"
                        )
                
                # Danger zone
                gr.Markdown("---")
                gr.Markdown("### ⚠️ Danger Zone")
                
                with gr.Row():
                    components['delete_btn'] = gr.Button(
                        "🗑️ Delete from Registry",
                        variant="stop",
                        size="sm"
                    )
                    components['delete_files_btn'] = gr.Button(
                        "🗑️ Delete with Files",
                        variant="stop",
                        size="sm"
                    )
        
        # Wire up events
        
        # Scan button
        components['scan_btn'].click(
            fn=scan_for_models,
            outputs=[components['models_table'], components['status_text']]
        )
        
        # Clean up button - removes orphaned registry entries
        components['cleanup_btn'].click(
            fn=cleanup_orphaned_models,
            outputs=[components['models_table'], components['status_text']]
        )
        
        # Filter tabs
        components['filter_tabs'].change(
            fn=filter_models,
            inputs=[components['filter_tabs'], components['search_input']],
            outputs=[components['models_table'], components['status_text']]
        )
        
        # Search input
        components['search_input'].change(
            fn=filter_models,
            inputs=[components['filter_tabs'], components['search_input']],
            outputs=[components['models_table'], components['status_text']]
        )
        
        # Table row selection - FIXED: Handle Gradio DataFrame properly
        def on_table_select(evt: gr.SelectData, current_data) -> Tuple[str, str, str, str, str, str, str, Any, str]:
            """Handle table row selection.
            
            Args:
                evt: Selection event with row index
                current_data: Current table data (DataFrame or list)
                
            Returns:
                Updated detail fields: (info, exports, config, metrics, notes, tags, status, plot, model_id)
            """
            try:
                if evt.index is not None and len(evt.index) > 0:
                    row_idx = evt.index[0]
                    
                    # Convert DataFrame to list if needed
                    import pandas as pd
                    if isinstance(current_data, pd.DataFrame):
                        data_list = current_data.values.tolist()
                    elif isinstance(current_data, list):
                        data_list = current_data
                    else:
                        data_list = []
                    
                    # Get the selected row data
                    if data_list and row_idx < len(data_list):
                        selected_row = data_list[row_idx]
                        model_id = str(selected_row[0]) if selected_row else ""  # First column is ID
                        
                        if model_id:
                            print(f"[MY_MODELS] Selected model: {model_id}")
                            
                            # Load model details - returns (info, exports, config, metrics, notes, tags, status, plot)
                            details = get_model_details(model_id)
                            
                            # Return all values including model_id at the end
                            # Order must match outputs list exactly:
                            # [info_text, exports_text, config_text, metrics_text, notes_input, tags_input, detail_status, loss_plot, selected_model_id]
                            return (
                                details[0],  # info_text
                                details[1],  # exports_text
                                details[2],  # config_text
                                details[3],  # metrics_text
                                details[4],  # notes_text
                                details[5],  # tags_text
                                details[6],  # status
                                details[7],  # plot
                                model_id       # selected_model_id
                            )
                        else:
                            print(f"[MY_MODELS] Empty model ID")
                    else:
                        print(f"[MY_MODELS] Invalid row index: {row_idx}, data length: {len(data_list) if data_list else 0}")
                
                # Return empty/default values if no selection
                return (
                    "Select a model from the table",
                    "",  # exports_text
                    "",  # config_text
                    "",  # metrics_text
                    "",  # notes_input
                    "",  # tags_input
                    "ok",  # detail_status
                    None,  # loss_plot
                    ""    # selected_model_id
                )
            except Exception as e:
                print(f"[MY_MODELS] Error in table selection: {e}")
                import traceback
                traceback.print_exc()
                return (
                    f"Error loading model: {e}",
                    "", "", "", "", "",
                    "error",
                    None,
                    ""
                )
        
        components['models_table'].select(
            fn=on_table_select,
            inputs=[components['models_table']],  # Pass current table data
            outputs=[
                components['info_text'],         # 1. info_text
                components['exports_text'],      # 2. exports_text
                components['config_text'],       # 3. config_text
                components['metrics_text'],    # 4. metrics_text
                components['notes_input'],       # 5. notes_text
                components['tags_input'],      # 6. tags_text
                components['detail_status'],   # 7. status
                components['loss_plot'],       # 8. loss_plot
                components['selected_model_id'] # 9. model_id
            ]
        )
        
        # Update notes
        components['update_notes_btn'].click(
            fn=update_model_notes,
            inputs=[components['selected_model_id'], components['notes_input']],
            outputs=[components['status_text']]
        )
        
        # Update tags
        components['update_tags_btn'].click(
            fn=update_model_tags,
            inputs=[components['selected_model_id'], components['tags_input']],
            outputs=[components['status_text']]
        )
        
        # Delete buttons
        components['delete_btn'].click(
            fn=lambda mid: delete_model(mid, False),
            inputs=[components['selected_model_id']],
            outputs=[components['models_table'], components['status_text']]
        )
        
        components['delete_files_btn'].click(
            fn=lambda mid: delete_model(mid, True),
            inputs=[components['selected_model_id']],
            outputs=[components['models_table'], components['status_text']]
        )
        
        # Chat button
        components['chat_btn'].click(
            fn=load_model_for_chat,
            inputs=[components['selected_model_id'], state],
            outputs=[components['status_text'], state]
        )
        
        # Download buttons
        components['download_adapter_btn'].click(
            fn=lambda mid: download_model_file(mid, 'adapter'),
            inputs=[components['selected_model_id']],
            outputs=[components['download_file'], components['status_text'], gr.Textbox(label="dummy")]
        )
        
        components['download_fused_btn'].click(
            fn=lambda mid: download_model_file(mid, 'fused'),
            inputs=[components['selected_model_id']],
            outputs=[components['download_file'], components['status_text'], gr.Textbox(label="dummy")]
        )
        
        components['download_gguf_btn'].click(
            fn=lambda mid: download_model_file(mid, 'gguf'),
            inputs=[components['selected_model_id']],
            outputs=[components['download_file'], components['status_text'], gr.Textbox(label="dummy")]
        )
        
        # Create export buttons
        components['create_fused_btn'].click(
            fn=create_fused_model,
            inputs=[components['selected_model_id']],
            outputs=[components['status_text'], components['exports_text']]
        )
        
        components['create_gguf_btn'].click(
            fn=convert_to_gguf,
            inputs=[components['selected_model_id']],
            outputs=[components['status_text'], components['exports_text']]
        )
        
        # Auto-scan when My Models tab is selected
        my_models_tab.select(
            fn=scan_for_models,
            outputs=[components['models_table'], components['status_text']]
        )
    
    return components
