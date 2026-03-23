"""Core business logic for EdukaAI Studio.

Handles state management and application logic.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def get_state_file() -> Path:
    """Get the state file path, creating directory if needed."""
    # Use project root for state file (more reliable than src/data)
    project_root = Path(__file__).parent.parent.parent.parent
    state_file = project_root / ".studio_state.json"
    return state_file


def save_state_to_disk(state: Dict[str, Any]) -> bool:
    """Save critical training state to disk.
    
    Args:
        state: Current application state
        
    Returns:
        True if save successful, False otherwise
    """
    try:
        state_file = get_state_file()
        
        data = {
            'training_complete': state.get('training_complete', False),
            'training_active': state.get('training_active', False),
            'output_dir': state.get('output_dir', None),
            'completion_time': state.get('completion_time', None),
            'model_name': state.get('model_name', None),
        }
        
        # Write to file and ensure it's flushed to disk
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()  # Flush buffer
            import os
            os.fsync(f.fileno())  # Force write to disk
        
        # Verify the write by reading it back
        with open(state_file, 'r') as f:
            verify_data = json.load(f)
        
        if verify_data.get('training_complete') != data['training_complete']:
            print(f"[STATE SAVE ERROR] Verification failed!")
            print(f"  Expected: training_complete={data['training_complete']}")
            print(f"  Got: training_complete={verify_data.get('training_complete')}")
            return False
        
        print(f"[STATE SAVE] Saved to {state_file}")
        print(f"[STATE SAVE] Content: training_complete={data['training_complete']}, "
              f"training_active={data['training_active']}, output_dir={data['output_dir']}")
        return True
        
    except Exception as e:
        print(f"[STATE SAVE ERROR] {e}")
        import traceback
        print(traceback.format_exc())
        return False


def load_state_from_disk() -> Optional[Dict[str, Any]]:
    """Load training state from disk.
    
    Returns:
        State dictionary if found, None otherwise
    """
    try:
        state_file = get_state_file()
        
        if not state_file.exists():
            print(f"[STATE LOAD] No state file found at {state_file}")
            return None
        
        with open(state_file, 'r') as f:
            data = json.load(f)
        
        print(f"[STATE LOAD] Loaded from {state_file}")
        print(f"[STATE LOAD] Content: training_complete={data.get('training_complete')}, "
              f"training_active={data.get('training_active')}, output_dir={data.get('output_dir')}")
        return data
        
    except Exception as e:
        print(f"[STATE LOAD ERROR] {e}")
        import traceback
        print(traceback.format_exc())
        return None


def clear_state_file() -> bool:
    """Clear the state file.
    
    Returns:
        True if cleared successfully, False otherwise
    """
    try:
        state_file = get_state_file()
        
        if state_file.exists():
            state_file.unlink()
            print(f"[STATE CLEAR] Cleared {state_file}")
        
        return True
        
    except Exception as e:
        print(f"[STATE CLEAR ERROR] {e}")
        return False


def get_initial_state() -> Dict[str, Any]:
    """Get the initial application state.
    
    Checks for existing training sessions and returns appropriate state.
    
    Returns:
        Initial state dictionary
    """
    print("[INIT] Checking for existing training sessions...")
    
    # Try to load saved state
    saved_state = load_state_from_disk()
    
    if saved_state:
        # Check if the output directory still exists
        output_dir = saved_state.get('output_dir')
        if output_dir and Path(output_dir).exists():
            print(f"[INIT] Found completed training in state file: {output_dir}")
            return {
                'uploaded_file': None,
                'training_config': {},
                'training_active': False,
                'training_complete': saved_state.get('training_complete', False),
                'output_dir': output_dir,
                'train_losses': {},
                'val_losses': {},
                'best_loss': float('inf'),
                'best_iter': 0,
                'log_lines': [],
                'monitor': None,
                'completion_time': saved_state.get('completion_time'),
                'model_name': saved_state.get('model_name'),
            }
        else:
            # Output directory doesn't exist, clear stale state
            print(f"[INIT] Stale training state detected: {output_dir}")
            clear_state_file()
    
    # Return default initial state
    return {
        'uploaded_file': None,
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
