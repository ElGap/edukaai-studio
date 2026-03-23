"""Results Tab for EdukaAI Studio.

Handles training results display and model downloads with adapter, fused, and GGUF support.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import gradio as gr

from edukaai_studio.core.state import load_state_from_disk, save_state_to_disk


def refresh_results_status(current_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Check current training status and update display.
    
    Args:
        current_state: Current application state (may be stale)
        
    Returns:
        Tuple of (status_message, updated_state)
    """
    # ALWAYS load fresh state from disk first (ignore potentially stale Gradio state)
    disk_state = load_state_from_disk()
    
    if disk_state:
        # Use disk state as base, merge with current state for any additional fields
        fresh_state = {
            **current_state,  # Keep any extra fields from Gradio state
            'training_complete': disk_state.get('training_complete', current_state.get('training_complete', False)),
            'training_active': disk_state.get('training_active', current_state.get('training_active', False)),
            'output_dir': disk_state.get('output_dir', current_state.get('output_dir')),
            'completion_time': disk_state.get('completion_time', current_state.get('completion_time')),
            'model_name': disk_state.get('model_name', current_state.get('model_name')),
        }
        current_state = fresh_state
        
        if disk_state.get('training_complete'):
            print(f"[REFRESH] Loaded fresh state from disk: training_complete=True")
    
    # Check if training is complete
    is_complete = current_state.get('training_complete', False)
    output_dir = current_state.get('output_dir')
    completion_time = current_state.get('completion_time')
    
    print(f"[REFRESH] Status: complete={is_complete}, active={current_state.get('training_active')}, dir={output_dir}")
    
    if is_complete:
        # Verify the output_dir from state exists and is valid
        if not output_dir:
            print(f"[REFRESH] No output_dir in state, cannot determine results location")
            return "Warning: Training completed but output directory not recorded. Please check the outputs/ folder.", current_state
        
        if not Path(output_dir).exists():
            print(f"[REFRESH] Output dir from state not found: {output_dir}")
            
            # Search for alternative directories that might be the actual output
            # This handles cases where the path format or timestamp differs
            outputs_base = Path("outputs")
            if outputs_base.exists():
                # Get model name from the path (e.g., "Qwen_2.5_7B" from "outputs/Qwen_2.5_7B_20260323_115946")
                expected_dir = Path(output_dir).name
                model_prefix = expected_dir.rsplit('_', 2)[0] if '_' in expected_dir else expected_dir
                
                print(f"[REFRESH] Searching for directories matching: {model_prefix}")
                
                matching_dirs = []
                for subdir in outputs_base.iterdir():
                    if subdir.is_dir() and model_prefix.lower() in subdir.name.lower():
                        has_adapters = (subdir / "adapters" / "adapters.safetensors").exists() or \
                                      (subdir / "best_adapter" / "adapters.safetensors").exists()
                        if has_adapters:
                            matching_dirs.append(subdir)
                
                if matching_dirs:
                    # Use the most recently modified matching directory
                    most_recent = max(matching_dirs, key=lambda p: p.stat().st_mtime)
                    output_dir = str(most_recent)
                    print(f"[REFRESH] Found matching directory: {output_dir}")
                    current_state['output_dir'] = output_dir
                    save_state_to_disk(current_state)
                else:
                    print(f"[REFRESH] No matching directories found")
                    return f"Warning: Output directory not found: {output_dir}. Training may have failed or files were moved.", current_state
            else:
                return f"Warning: Output directory not found: {output_dir}. Training may have failed or files were moved.", current_state
        
        # We have a valid output_dir from state - verify it has the expected files
        print(f"[REFRESH] Using output_dir: {output_dir}")
        
        if output_dir and Path(output_dir).exists():
            adapter_exists = (Path(output_dir) / "adapters" / "adapters.safetensors").exists() or \
                           (Path(output_dir) / "best_adapter" / "adapters.safetensors").exists()
            fused_exists = (Path(output_dir) / "fused_model").exists()
            gguf_exists = len(list(Path(output_dir).rglob("*.gguf"))) > 0
            
            status_parts = []
            if adapter_exists:
                status_parts.append("Adapter OK")
            if fused_exists:
                status_parts.append("Fused Model OK")
            if gguf_exists:
                status_parts.append("GGUF OK")
            
            if status_parts:
                files_status = ", ".join(status_parts)
                return f"OK: Training complete! {files_status} | Output: {output_dir}", current_state
            else:
                return f"Warning: Training complete but files not found at: {output_dir}", current_state
        else:
            return "OK: Training complete! (Output directory not found - try refreshing)", current_state
    elif current_state.get('training_active'):
        return "Loading: Training in progress... Check Train tab for details.", current_state
    else:
        return "Warning: No training completed. Go to Train tab to start training.", current_state


def download_adapter(current_state: Dict[str, Any]) -> Tuple:
    """Download LoRA adapter file.
    
    Args:
        current_state: Current application state
        
    Returns:
        Tuple of (file_path, status_message, file_component)
    """
    # Check disk state for training completion
    disk_state = load_state_from_disk()
    is_complete = current_state.get("training_complete") or (disk_state.get("training_complete") if disk_state else False)
    output_dir = current_state.get("output_dir") or (disk_state.get("output_dir") if disk_state else None)
    
    if not is_complete:
        return None, "Error: Training not complete", gr.File(visible=False)
    
    if not output_dir:
        return None, "Error: Output directory not found", gr.File(visible=False)
    
    # Check multiple possible locations for the adapter file
    possible_paths = [
        Path(output_dir) / "adapters" / "adapters.safetensors",
        Path(output_dir) / "best_adapter" / "adapters.safetensors",
        Path(output_dir) / "adapters.safetensors",
    ]
    
    for adapter_path in possible_paths:
        if adapter_path.exists():
            return str(adapter_path), f"OK: Click the file above to download adapter ({adapter_path.stat().st_size / 1024 / 1024:.1f} MB)", gr.File(visible=True, label=f"Adapter: {adapter_path.name}")
    
    # Auto-discover
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
                        current_state['output_dir'] = str(subdir)
                        save_state_to_disk(current_state)
                        return str(possible_path), f"OK: Click the file above to download adapter ({possible_path.stat().st_size / 1024 / 1024:.1f} MB)", gr.File(visible=True, label=f"Adapter: {possible_path.name}")
    
    return None, f"Error: Adapter not found. Searched: {[str(p) for p in possible_paths]}", gr.File(visible=False)


def download_fused(current_state: Dict[str, Any]) -> Tuple:
    """Download fused model directory as zip.
    
    Args:
        current_state: Current application state
        
    Returns:
        Tuple of (file_path, status_message, file_component)
    """
    disk_state = load_state_from_disk()
    is_complete = current_state.get("training_complete") or (disk_state.get("training_complete") if disk_state else False)
    output_dir = current_state.get("output_dir") or (disk_state.get("output_dir") if disk_state else None)
    
    if not is_complete:
        return None, "Error: Training not complete", gr.File(visible=False)
    
    if not output_dir:
        return None, "Error: Output directory not found", gr.File(visible=False)
    
    fused_path = Path(output_dir) / "fused_model"
    if fused_path.exists():
        zip_path = Path(output_dir) / "fused_model.zip"
        try:
            if zip_path.exists():
                zip_path.unlink()
            shutil.make_archive(str(zip_path.with_suffix('')), 'zip', fused_path)
            print(f"[DOWNLOAD] Created zip archive: {zip_path}")
            file_size_mb = zip_path.stat().st_size / 1024 / 1024
            return str(zip_path), f"OK: Click the file above to download fused model ({file_size_mb:.1f} MB)", gr.File(visible=True, label=f"Fused Model: {zip_path.name} ({file_size_mb:.1f} MB)")
        except Exception as e:
            print(f"[DOWNLOAD] Error creating zip: {e}")
            return None, f"Error creating zip archive: {str(e)}", gr.File(visible=False)
    
    # Auto-discover
    print(f"[DOWNLOAD] Fused model not found at {fused_path}, searching...")
    outputs_base = Path("outputs")
    if outputs_base.exists():
        for subdir in outputs_base.iterdir():
            if subdir.is_dir():
                possible_path = subdir / "fused_model"
                if possible_path.exists():
                    print(f"[DOWNLOAD] OK: Auto-discovered fused model at: {possible_path}")
                    current_state['output_dir'] = str(subdir)
                    save_state_to_disk(current_state)
                    zip_path = subdir / "fused_model.zip"
                    try:
                        if zip_path.exists():
                            zip_path.unlink()
                        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', possible_path)
                        file_size_mb = zip_path.stat().st_size / 1024 / 1024
                        return str(zip_path), f"OK: Click the file above to download fused model ({file_size_mb:.1f} MB)", gr.File(visible=True, label=f"Fused Model: {zip_path.name} ({file_size_mb:.1f} MB)")
                    except Exception as e:
                        return None, f"Error creating zip: {str(e)}", gr.File(visible=False)
    
    return None, f"Error: Fused model not found at: {fused_path}", gr.File(visible=False)


def download_gguf(current_state: Dict[str, Any]) -> Tuple:
    """Download or create GGUF model file on-demand for llama.cpp compatibility.
    
    Args:
        current_state: Current application state
        
    Returns:
        Tuple of (file_path, status_message, file_component)
    """
    disk_state = load_state_from_disk()
    is_complete = current_state.get("training_complete") or (disk_state.get("training_complete") if disk_state else False)
    output_dir = current_state.get("output_dir") or (disk_state.get("output_dir") if disk_state else None)
    
    if not is_complete:
        return None, "Error: Training not complete", gr.File(visible=False)
    
    if not output_dir:
        return None, "Error: Output directory not found", gr.File(visible=False)
    
    output_path = Path(output_dir)
    
    # Check if GGUF already exists
    print(f"[GGUF] Checking for existing GGUF files in {output_path}...")
    
    if output_path.exists():
        for gguf_file in output_path.rglob("*.gguf"):
            print(f"[GGUF] OK: Found existing GGUF file: {gguf_file}")
            file_size_mb = gguf_file.stat().st_size / 1024 / 1024
            return str(gguf_file), f"OK: GGUF model ready ({file_size_mb:.1f} MB) - click above to download", gr.File(visible=True, label=f"GGUF Model: {gguf_file.name} ({file_size_mb:.1f} MB)")
    
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
        except Exception:
            pass
    
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
                "--outtype", "q4_k_m",
            ]
        else:
            cmd = [
                sys.executable, convert_script,
                str(fused_path),
                str(gguf_output),
            ]
        
        print(f"[GGUF] Running command: {' '.join(cmd)}")
        
        # Run conversion with progress tracking
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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
            return str(gguf_output), f"OK: GGUF model created on-demand ({file_size_mb:.1f} MB) - click above to download", gr.File(visible=True, label=f"GGUF Model: {gguf_output.name} ({file_size_mb:.1f} MB)")
        else:
            # Try to find the created file
            for gguf_file in output_path.rglob("*.gguf"):
                file_size_mb = gguf_file.stat().st_size / 1024 / 1024
                print(f"[GGUF] OK: Found created GGUF: {gguf_file}")
                return str(gguf_file), f"OK: GGUF model created on-demand ({file_size_mb:.1f} MB)", gr.File(visible=True, label=f"GGUF Model: {gguf_file.name} ({file_size_mb:.1f} MB)")
            
            return None, "Error: Conversion appeared to succeed but no GGUF file found", gr.File(visible=False)
            
    except subprocess.TimeoutExpired:
        print(f"[GGUF] Conversion timed out after 10 minutes")
        return None, "Error: GGUF conversion timed out (10 minutes). The model may be too large or the conversion is stuck.", gr.File(visible=False)
    except Exception as e:
        print(f"[GGUF] Error during conversion: {e}")
        import traceback
        print(f"[GGUF] Traceback: {traceback.format_exc()}")
        return None, f"Error: Error during GGUF conversion: {str(e)}", gr.File(visible=False)


def create_results_tab(state: gr.State, tabs: gr.Tabs) -> Dict[str, Any]:
    """Create the Results tab.
    
    Args:
        state: Gradio state object
        tabs: Gradio tabs container
        
    Returns:
        Dictionary of component references
    """
    components = {}
    
    with gr.TabItem("4. Results") as results_tab:
        gr.Markdown("## Step 4: Training Results")
        
        components['results_status'] = gr.Textbox(
            label="Status",
            value="Complete training to see download options",
            interactive=False
        )
        
        gr.Markdown("### Download Trained Model")
        gr.Markdown("Download your fine-tuned model in various formats.")
        
        with gr.Row():
            with gr.Column():
                components['adapter_btn'] = gr.Button(
                    "Download LoRA Adapter", variant="secondary"
                )
                components['adapter_file'] = gr.File(
                    label="LoRA Adapter Download",
                    visible=False,
                    interactive=False
                )
            
            with gr.Column():
                components['fused_btn'] = gr.Button(
                    "Download Fused Model", variant="secondary"
                )
                components['fused_file'] = gr.File(
                    label="Fused Model Download",
                    visible=False,
                    interactive=False
                )
            
            with gr.Column():
                components['gguf_btn'] = gr.Button(
                    "Download GGUF Model", variant="secondary"
                )
                components['gguf_file'] = gr.File(
                    label="GGUF Model Download",
                    visible=False,
                    interactive=False
                )
        
        components['refresh_results_btn'] = gr.Button(
            "Refresh Status", variant="secondary", size="sm"
        )
        
        # Wire up events
        components['refresh_results_btn'].click(
            fn=refresh_results_status,
            inputs=[state],
            outputs=[components['results_status'], state]
        )
        
        components['adapter_btn'].click(
            fn=download_adapter,
            inputs=[state],
            outputs=[components['adapter_file'], components['results_status'], components['adapter_file']]
        )
        
        components['fused_btn'].click(
            fn=download_fused,
            inputs=[state],
            outputs=[components['fused_file'], components['results_status'], components['fused_file']]
        )
        
        components['gguf_btn'].click(
            fn=download_gguf,
            inputs=[state],
            outputs=[components['gguf_file'], components['results_status'], components['gguf_file']]
        )
        
        # Auto-refresh when tab is selected
        results_tab.select(
            fn=refresh_results_status,
            inputs=[state],
            outputs=[components['results_status'], state]
        )
    
    return components
