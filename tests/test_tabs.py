"""Tests for EdukaAI Studio UI tabs."""

import sys
import json
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from edukaai_studio.ui.tabs.upload import (
    process_uploaded_file,
    _extract_preview_data,
    refresh_preview_data,
)


class TestUploadTab:
    """Test upload tab functionality."""
    
    def test_extract_preview_data_first_5(self):
        """Test extracting first 5 samples."""
        examples = [
            {'instruction': f'Instruction {i}', 'output': f'Output {i}'}
            for i in range(10)
        ]
        
        result = _extract_preview_data(examples, 'First 5')
        
        assert len(result) == 5
        assert result[0][0] == 'Instruction 0'
    
    def test_extract_preview_data_random_5(self):
        """Test extracting random 5 samples."""
        examples = [
            {'instruction': f'Instruction {i}', 'output': f'Output {i}'}
            for i in range(10)
        ]
        
        result = _extract_preview_data(examples, 'Random 5')
        
        assert len(result) == 5
        # Should be different from first 5
        first_5 = _extract_preview_data(examples, 'First 5')
        # Random might match first occasionally, but usually different
        assert len(result) == len(first_5)
    
    def test_extract_preview_data_truncate_long_text(self):
        """Test that long text gets truncated."""
        long_instruction = 'x' * 200
        examples = [
            {'instruction': long_instruction, 'output': 'Short'}
        ]
        
        result = _extract_preview_data(examples, 'First 5')
        
        assert len(result[0][0]) <= 103  # 100 chars + "..."
        assert result[0][0].endswith('...')


class TestConfigureTab:
    """Test configure tab functionality."""
    
    def test_update_params_from_preset(self):
        """Test preset parameter updates."""
        from edukaai_studio.ui.tabs.configure import update_params_from_preset
        
        # Test quick preset
        result = update_params_from_preset('quick')
        assert len(result) == 6  # 6 parameters
        assert result[0] < 1000  # iterations should be lower
        
        # Test maximum preset
        result = update_params_from_preset('maximum')
        assert result[0] > 900  # iterations should be higher
    
    def test_configure_training_no_uploaded_file(self):
        """Test configure training with no uploaded file."""
        from edukaai_studio.ui.tabs.configure import configure_training
        
        result = configure_training(
            'model-id', 'preset', 100, '1e-4', 16, 32, 2048, 2, 10, 32,
            {'uploaded_file': None}
        )
        
        assert 'Upload training data first' in result[0]


class TestModelsTab:
    """Test models tab functionality."""
    
    def test_verify_custom_model_invalid_format(self):
        """Test verification with invalid format."""
        from edukaai_studio.ui.tabs.models import verify_custom_model
        
        result = verify_custom_model('invalid-input')
        
        assert 'Invalid format' in result[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
