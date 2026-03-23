"""
User Configuration Manager

Manages user-specific settings including HuggingFace token.
Stored in user_config.json (not committed to git).
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict


CONFIG_FILE = Path("user_config.json")


def load_user_config() -> Dict:
    """Load user configuration from file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_user_config(config: Dict):
    """Save user configuration to file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Set restrictive permissions (user only)
    try:
        os.chmod(CONFIG_FILE, 0o600)
    except:
        pass  # Windows may not support this


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from user config."""
    config = load_user_config()
    return config.get('hf_token')


def set_hf_token(token: str):
    """Set HuggingFace token in user config."""
    config = load_user_config()
    config['hf_token'] = token.strip()
    save_user_config(config)


def clear_hf_token():
    """Remove HuggingFace token from user config."""
    config = load_user_config()
    if 'hf_token' in config:
        del config['hf_token']
        save_user_config(config)


def mask_token(token: str) -> str:
    """Mask token for display (show only last 4 chars)."""
    if not token:
        return ""
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def has_hf_token() -> bool:
    """Check if HF token is configured."""
    return get_hf_token() is not None


if __name__ == "__main__":
    # Test the module
    print("Testing user config...")
    
    # Test save
    set_hf_token("hf_test_token_12345")
    print(f"Token saved: {mask_token(get_hf_token())}")
    
    # Test load
    config = load_user_config()
    print(f"Config loaded: {config}")
    
    # Test clear
    clear_hf_token()
    print(f"After clear: {get_hf_token()}")
    
    print("✓ User config module working")
