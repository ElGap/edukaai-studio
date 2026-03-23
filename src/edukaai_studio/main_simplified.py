"""EdukaAI Fine Tuning Studio - Main Application

Entry point that assembles all tabs into a cohesive application.
"""

import gradio as gr
from pathlib import Path

# Import configuration
from edukaai_studio.config import SERVER

# Import core state management
from edukaai_studio.core.state import get_initial_state, save_state_to_disk

# Import UI tabs
from edukaai_studio.ui.tabs import (
    create_upload_tab,
    create_configure_tab,
    create_train_tab,
    create_results_tab,
    create_chat_tab,
    create_models_tab,
    create_my_models_tab,
)

# Import UI components
try:
    from edukaai_studio.ui.training_monitor import TrainingMonitor
    from edukaai_studio.ui.chat_wrapper import ChatWrapper
except ImportError as e:
    print(f"Import warning: {e}")
    TrainingMonitor = None
    ChatWrapper = None


def create_ui():
    """Create and configure the Gradio UI."""

    # Initialize state
    initial_state = get_initial_state()

    with gr.Blocks(title="EdukaAI Fine Tuning Studio") as app:
        # Global state
        state = gr.State(initial_state)

        # Header
        gr.Markdown("# EdukaAI Fine Tuning Studio")
        gr.Markdown("### Fine-tune language models on Apple Silicon - Made Simple")

        # Tabs container
        tabs = gr.Tabs()

        with tabs:
            # Create all tabs
            upload_components = create_upload_tab(state, tabs)
            configure_components = create_configure_tab(state, tabs)
            train_components = create_train_tab(state, tabs)
            results_components = create_results_tab(state, tabs)
            models_components = create_models_tab(state, tabs)
            my_models_components = create_my_models_tab(state, tabs)
            chat_components = create_chat_tab(state, tabs)

        # Footer
        gr.Markdown("---")
        gr.Markdown("EdukaAI Fine Tuning Studio | Apple Silicon Fine-Tuning Made Simple")

    return app


def main():
    """Main entry point."""
    print("=" * 60)
    print("Starting EdukaAI Fine Tuning Studio")
    print("Real MLX Training for Apple Silicon")
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
