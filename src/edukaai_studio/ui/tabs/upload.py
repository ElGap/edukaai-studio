"""Upload Tab for EdukaAI Studio.

Handles file upload, validation, and data preview.
"""

import json
import random
import gradio as gr
from typing import Dict, Any, Tuple


def process_uploaded_file(file_path: str, preview_mode: str, current_state: Dict[str, Any]) -> Tuple:
    """Process uploaded training data file.
    
    Args:
        file_path: Path to uploaded file
        preview_mode: "First 5" or "Random 5"
        current_state: Current application state
        
    Returns:
        Tuple of (status, preview_data, preview_visible, radio_visible, new_state, button_visible)
    """
    if not file_path:
        return (
            "No file selected", 
            None, 
            gr.Dataframe(visible=False), 
            gr.Radio(visible=False), 
            current_state, 
            gr.Button(visible=False)
        )
    
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
            return (
                "Error: No valid examples found", 
                None, 
                gr.Dataframe(visible=False), 
                gr.Radio(visible=False), 
                current_state, 
                gr.Button(visible=False)
            )
        
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
            return (
                "Error: File must have 'instruction' and 'output' fields (or 'prompt'/'response')", 
                None, 
                gr.Dataframe(visible=False), 
                gr.Radio(visible=False), 
                current_state, 
                gr.Button(visible=False)
            )
        
        # Extract sample data for preview
        preview_data = _extract_preview_data(examples, preview_mode)
        
        new_state = {**current_state, 'uploaded_file': file_path}
        format_type = "Alpaca" if has_input else "standard"
        success_msg = f"OK: {count} {format_type} examples validated and ready"
        
        return (
            success_msg, 
            preview_data, 
            gr.Dataframe(visible=True), 
            gr.Radio(visible=True), 
            new_state, 
            gr.Button(visible=True)
        )
        
    except json.JSONDecodeError as e:
        return (
            f"Error: Invalid JSON format - {str(e)}", 
            None, 
            gr.Dataframe(visible=False), 
            gr.Radio(visible=False), 
            current_state, 
            gr.Button(visible=False)
        )
    except Exception as e:
        return (
            f"Error: {str(e)}", 
            None, 
            gr.Dataframe(visible=False), 
            gr.Radio(visible=False), 
            current_state, 
            gr.Button(visible=False)
        )


def _extract_preview_data(examples: list, preview_mode: str) -> list:
    """Extract sample data for preview display.
    
    Args:
        examples: List of training examples
        preview_mode: "First 5" or "Random 5"
        
    Returns:
        List of [instruction, input, output] for DataFrame
    """
    if preview_mode == "Random 5" and len(examples) > 5:
        samples = random.sample(examples, 5)
    else:
        samples = examples[:5]
    
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
    
    return preview_data


def refresh_preview_data(preview_mode: str, current_state: Dict[str, Any]) -> Tuple:
    """Refresh data preview when mode changes.
    
    Args:
        preview_mode: "First 5" or "Random 5"
        current_state: Current application state
        
    Returns:
        Tuple of (preview_data, preview_visible)
    """
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
        
        preview_data = _extract_preview_data(examples, preview_mode)
        
        return preview_data, gr.Dataframe(visible=True)
    except Exception:
        return None, gr.Dataframe(visible=False)


def create_upload_tab(state: gr.State, tabs: gr.Tabs) -> Dict[str, Any]:
    """Create the Upload tab.
    
    Args:
        state: Gradio state object
        tabs: Gradio tabs container
        
    Returns:
        Dictionary of component references
    """
    components = {}
    
    with gr.TabItem("1. Upload"):
        gr.Markdown("## Step 1: Upload Training Data")
        
        components['file_upload'] = gr.File(
            label="Training Data (JSONL or Alpaca JSON)",
            file_types=[".jsonl", ".json"]
        )
        
        components['file_status'] = gr.Textbox(
            label="Status",
            value="Please upload a JSONL or Alpaca JSON file",
            interactive=False
        )
        
        # Data preview component with proper styling
        components['data_preview'] = gr.Dataframe(
            label="Data Preview (5 samples)",
            headers=["Instruction", "Input", "Output"],
            visible=False,
            interactive=False,
            elem_classes=["data-preview-table"]
        )
        
        # Add CSS to ensure proper contrast for data preview
        gr.HTML("""
        <style>
        .data-preview-table table {
            background-color: white !important;
            color: #1f2937 !important;
        }
        .data-preview-table th {
            background-color: #f3f4f6 !important;
            color: #1f2937 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #d1d5db !important;
        }
        .data-preview-table td {
            background-color: white !important;
            color: #1f2937 !important;
            border-bottom: 1px solid #e5e7eb !important;
        }
        .data-preview-table tr:hover td {
            background-color: #f9fafb !important;
        }
        </style>
        """)
        
        components['preview_type'] = gr.Radio(
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
        gr.Markdown("### ✅ Next Step")
        
        # Navigation button with helpful guidance
        components['go_to_configure_btn'] = gr.Button(
            "✅ Data Uploaded! Go to Configure Tab →",
            variant="primary",
            size="lg",
            visible=False
        )
        
        def navigate_to_configure():
            """Show helpful message about next step."""
            return """✅ Data uploaded successfully!

📋 NEXT STEP: Click on the "⚙️ Configure" tab at the top of the page to configure your training settings.

The Configure tab will let you:
• Select your base model
• Adjust training parameters  
• Start the training process"""
        
        # Wire up events
        components['file_upload'].change(
            fn=process_uploaded_file,
            inputs=[components['file_upload'], components['preview_type'], state],
            outputs=[
                components['file_status'], 
                components['data_preview'], 
                components['data_preview'], 
                components['preview_type'], 
                state, 
                components['go_to_configure_btn']
            ]
        )
        
        components['preview_type'].change(
            fn=refresh_preview_data,
            inputs=[components['preview_type'], state],
            outputs=[components['data_preview'], components['data_preview']]
        )
        
        # Navigation button click handler - updates status with helpful message
        components['go_to_configure_btn'].click(
            fn=navigate_to_configure,
            outputs=[components['file_status']]
        )
    
    return components
