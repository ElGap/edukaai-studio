"""
Training Scanner Module

Simple scanner to discover and list all training output folders.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


def scan_output_folders(outputs_dir: str = "outputs") -> List[Dict]:
    """
    Scan outputs directory and return list of valid training folders.
    
    Args:
        outputs_dir: Path to outputs directory (default: "outputs")
        
    Returns:
        List of training info dictionaries, sorted by timestamp (newest first)
    """
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        return []
    
    trainings = []
    
    for folder in outputs_path.iterdir():
        if not folder.is_dir():
            continue
            
        # Check if it's a valid training folder
        has_adapter = (folder / "best_adapter" / "adapters.safetensors").exists() or \
                     (folder / "adapters" / "adapters.safetensors").exists()
        has_fused = (folder / "fused_model" / "model.safetensors").exists()
        has_summary = (folder / "training_summary.json").exists()
        
        if has_adapter or has_fused or has_summary:
            # Get metadata from summary if available
            if has_summary:
                try:
                    with open(folder / "training_summary.json", 'r') as f:
                        summary = json.load(f)
                    
                    trainings.append({
                        'id': folder.name,
                        'folder': str(folder),
                        'model': summary.get('model', 'Unknown'),
                        'model_name': _extract_model_name(summary.get('model', '')),
                        'best_loss': summary.get('best_val_loss', 0),
                        'iterations': summary.get('training_config', {}).get('iterations', 0),
                        'timestamp': summary.get('timestamp', ''),
                        'formatted_date': _format_timestamp(summary.get('timestamp', '')),
                        'has_fused': has_fused,
                        'status': 'completed'
                    })
                except Exception as e:
                    # If summary parsing fails, add basic info
                    trainings.append({
                        'id': folder.name,
                        'folder': str(folder),
                        'model': 'Unknown',
                        'model_name': folder.name,
                        'best_loss': 0,
                        'iterations': 0,
                        'timestamp': '',
                        'formatted_date': 'Unknown',
                        'has_fused': has_fused,
                        'status': 'unknown'
                    })
            else:
                # No summary but has model files
                trainings.append({
                    'id': folder.name,
                    'folder': str(folder),
                    'model': 'Unknown',
                    'model_name': folder.name,
                    'best_loss': 0,
                    'iterations': 0,
                    'timestamp': '',
                    'formatted_date': 'Unknown',
                    'has_fused': has_fused,
                    'status': 'completed'
                })
    
    # Sort by timestamp (newest first), put unknown dates at end
    def sort_key(t):
        if t['timestamp']:
            return t['timestamp']
        return '0000-00-00T00:00:00'
    
    return sorted(trainings, key=sort_key, reverse=True)


def get_training_choices(trainings: List[Dict]) -> List[tuple]:
    """
    Convert training list to dropdown choices.
    
    Returns:
        List of (display_text, folder_path) tuples
    """
    choices = []
    
    for t in trainings:
        # Format: "Model Name - Data Type (Loss: X.XX, 100 iters)"
        loss_str = f"Loss: {t['best_loss']:.2f}" if t['best_loss'] > 0 else "Loss: N/A"
        iter_str = f"{t['iterations']} iters" if t['iterations'] > 0 else ""
        
        if t['formatted_date'] != 'Unknown':
            display = f"{t['model_name']} - {t['formatted_date']} ({loss_str}, {iter_str})"
        else:
            display = f"{t['model_name']} - {t['id'][:30]} ({loss_str})"
        
        choices.append((display, t['folder']))
    
    return choices


def _extract_model_name(full_model_id: str) -> str:
    """Extract short model name from full HuggingFace ID."""
    if not full_model_id:
        return "Unknown"
    
    # Remove common prefixes
    name = full_model_id.split('/')[-1] if '/' in full_model_id else full_model_id
    
    # Clean up common suffixes
    name = name.replace('-4bit', '').replace('-8bit', '')
    name = name.replace('-instruct', '').replace('-chat', '')
    
    # Capitalize and format
    name = name.replace('_', ' ').replace('-', ' ')
    name = ' '.join(word.capitalize() for word in name.split())
    
    return name


def _format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp to readable date."""
    if not timestamp_str:
        return 'Unknown'
    
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %H:%M")
    except:
        return 'Unknown'


def load_training_info(folder_path: str) -> Optional[Dict]:
    """
    Load detailed info about a specific training folder.
    
    Args:
        folder_path: Path to training folder
        
    Returns:
        Dictionary with training info or None if invalid
    """
    folder = Path(folder_path)
    if not folder.exists():
        return None
    
    info = {
        'folder': str(folder),
        'id': folder.name,
        'has_adapter': False,
        'has_fused': False,
        'has_summary': False,
        'summary': {}
    }
    
    # Check for files
    info['has_adapter'] = (folder / "best_adapter" / "adapters.safetensors").exists() or \
                         (folder / "adapters" / "adapters.safetensors").exists()
    info['has_fused'] = (folder / "fused_model").exists()
    info['has_summary'] = (folder / "training_summary.json").exists()
    
    # Load summary if available
    if info['has_summary']:
        try:
            with open(folder / "training_summary.json", 'r') as f:
                info['summary'] = json.load(f)
        except:
            pass
    
    return info


if __name__ == "__main__":
    # Test the scanner
    print("Testing training scanner...")
    trainings = scan_output_folders()
    print(f"Found {len(trainings)} training(s)")
    
    for t in trainings[:3]:
        print(f"\n  {t['id']}")
        print(f"    Model: {t['model_name']}")
        print(f"    Loss: {t['best_loss']:.4f}")
        print(f"    Date: {t['formatted_date']}")
    
    if trainings:
        choices = get_training_choices(trainings)
        print(f"\nDropdown choices:")
        for display, path in choices[:3]:
            print(f"  - {display}")
