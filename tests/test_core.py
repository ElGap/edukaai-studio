"""Tests for EdukaAI Studio core functionality."""

import sys
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edukaai_studio.core.state import (
    get_initial_state,
    save_state_to_disk,
    load_state_from_disk,
    clear_state_file,
)


class TestStateManagement:
    """Test state management functionality."""
    
    def test_get_initial_state_returns_dict(self):
        """Test that get_initial_state returns a dictionary."""
        state = get_initial_state()
        assert isinstance(state, dict)
        assert 'uploaded_file' in state
        assert 'training_config' in state
    
    def test_save_and_load_state(self, tmp_path):
        """Test saving and loading state."""
        # Mock the state file location for testing
        from edukaai_studio import core
        original_file = core.state.STATE_FILE
        core.state.STATE_FILE = tmp_path / "test_state.json"
        
        try:
            test_state = {
                'training_complete': True,
                'output_dir': '/tmp/test',
                'model_name': 'Test Model',
            }
            
            # Save state
            result = save_state_to_disk(test_state)
            assert result is True
            
            # Load state
            loaded = load_state_from_disk()
            assert loaded is not None
            assert loaded['training_complete'] is True
            assert loaded['output_dir'] == '/tmp/test'
            
            # Clear state
            result = clear_state_file()
            assert result is True
            
            # Verify cleared
            loaded = load_state_from_disk()
            assert loaded is None
            
        finally:
            core.state.STATE_FILE = original_file


class TestConfiguration:
    """Test configuration loading."""
    
    def test_config_imports(self):
        """Test that configuration can be imported."""
        from edukaai_studio.config import STUDIO_MODELS, SERVER, TRAINING
        
        assert STUDIO_MODELS is not None
        assert SERVER is not None
        assert TRAINING is not None
    
    def test_training_config_values(self):
        """Test training configuration has expected values."""
        from edukaai_studio.config import TRAINING
        
        assert TRAINING.DEFAULT_ITERATIONS > 0
        assert TRAINING.DEFAULT_LORA_RANK > 0
        assert TRAINING.DEFAULT_LEARNING_RATE is not None


class TestModelValidator:
    """Test HuggingFace model validation."""
    
    def test_hf_model_validator_imports(self):
        """Test that HF model validator can be imported."""
        from edukaai_studio.ui.hf_model_validator import HFModelValidator
        assert HFModelValidator is not None
    
    def test_parse_model_input_valid(self):
        """Test parsing valid model inputs."""
        from edukaai_studio.ui.hf_model_validator import HFModelValidator
        
        validator = HFModelValidator()
        
        # Test direct ID
        result = validator.parse_model_input("org/model-name")
        assert result == "org/model-name"
        
        # Test HuggingFace URL
        result = validator.parse_model_input("https://huggingface.co/org/model-name")
        assert result == "org/model-name"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
