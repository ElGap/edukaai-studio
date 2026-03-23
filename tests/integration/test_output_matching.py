"""Test for output directory matching and state synchronization."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import time


class TestOutputDirectoryMatching:
    """Test that results tab correctly finds the actual output directory."""
    
    def test_results_finds_matching_directory_when_state_path_missing(self):
        """If state has wrong timestamp, find actual directory by model name.
        
        Scenario:
        - State has: outputs/Qwen_2.5_7B_20260323_115946
        - Actual: outputs/Qwen2.5_7B_Instruct_4bit_20260323_115720
        - Should find and use the actual directory
        """
        from edukaai_studio.ui.tabs.results import refresh_results_status
        from edukaai_studio.core.state import save_state_to_disk
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock outputs directory
            outputs_dir = Path(tmpdir) / "outputs"
            outputs_dir.mkdir()
            
            # Create actual model directory (simulating real training output)
            actual_dir = outputs_dir / "Qwen2.5_7B_Instruct_4bit_20260323_115720"
            actual_dir.mkdir()
            
            # Create adapter file to make it valid
            adapters_dir = actual_dir / "adapters"
            adapters_dir.mkdir()
            (adapters_dir / "adapters.safetensors").touch()
            
            # Wait a moment to ensure different timestamps
            time.sleep(0.1)
            
            # State has wrong/wrongly-formatted path
            wrong_state = {
                'training_complete': True,
                'training_active': False,
                'output_dir': str(outputs_dir / "Qwen_2.5_7B_20260323_115946"),  # Wrong timestamp
            }
            
            # Patch the outputs path
            with patch('edukaai_studio.ui.tabs.results.Path', return_value=outputs_dir):
                # This should find the actual directory despite wrong timestamp
                result = refresh_results_status(wrong_state)
                
                # Should return success message with actual directory
                status_msg = result[0]
                updated_state = result[1]
                
                # Verify it found the model
                assert "OK: Training complete!" in status_msg or "Adapter OK" in status_msg
                # Verify state was updated with correct path
                assert "Qwen2.5_7B_Instruct_4bit" in updated_state['output_dir']
    
    def test_results_uses_exact_path_when_it_exists(self):
        """If state path exists exactly, use it without searching."""
        from edukaai_studio.ui.tabs.results import refresh_results_status
        
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs"
            outputs_dir.mkdir()
            
            # Create exact directory from state
            exact_dir = outputs_dir / "Phi_3_mini_20260323_120000"
            exact_dir.mkdir()
            adapters_dir = exact_dir / "adapters"
            adapters_dir.mkdir()
            (adapters_dir / "adapters.safetensors").touch()
            
            state = {
                'training_complete': True,
                'training_active': False,
                'output_dir': str(exact_dir),
            }
            
            with patch('edukaai_studio.ui.tabs.results.Path', return_value=outputs_dir):
                result = refresh_results_status(state)
                
                status_msg = result[0]
                updated_state = result[1]
                
                # Should use exact path from state
                assert "OK: Training complete!" in status_msg
                assert updated_state['output_dir'] == str(exact_dir)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
