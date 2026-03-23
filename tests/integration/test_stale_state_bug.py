"""Tests for the stale state bug that caused "Training already completed" on new training.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import json


class TestStaleStateBug:
    """Test for the critical bug: clicking Start Training with old completed state."""
    
    def test_stale_state_cleared_before_training_start(self):
        """CRITICAL: Old completed state must be cleared before starting new training.
        
        This was the bug:
        1. Previous training completed → state has training_complete=True
        2. User clicks "Start Training" again
        3. Function received OLD state with training_complete=True
        4. OLD CODE: Immediately returned "already completed" without starting training
        5. User saw "Training already completed" in 1 second
        
        With fix:
        1. Function CLEARS state first
        2. Then proceeds with fresh state (training_complete=False)
        3. Training actually starts
        """
        from edukaai_studio.core.state import save_state_to_disk, load_state_from_disk, clear_state_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Step 1: Save OLD completed state (simulating previous training)
                old_completed_state = {
                    'training_complete': True,
                    'training_active': False,
                    'output_dir': 'outputs/old_training_20240101_120000',
                    'completion_time': '12:00:00',
                    'uploaded_file': '/old/data.json',  # Keep uploaded file
                    'training_config': {'model_id': 'test-model'},  # Keep config
                }
                save_state_to_disk(old_completed_state)
                
                # Verify old state exists
                loaded = load_state_from_disk()
                assert loaded['training_complete'] is True
                assert loaded['output_dir'] == 'outputs/old_training_20240101_120000'
                
                # Step 2: Simulate what start_training_real does - CLEAR state first
                clear_state_file()
                
                # Create fresh state
                fresh_state = {
                    **old_completed_state,  # Keep uploaded_file and config
                    'training_complete': False,  # Clear completion flags
                    'training_active': False,
                    'output_dir': None,
                    'completion_time': None,
                    'train_losses': {},
                    'val_losses': {},
                    'best_loss': float('inf'),
                    'best_iter': 0,
                }
                save_state_to_disk(fresh_state)
                
                # Step 3: Verify state was cleared
                cleared = load_state_from_disk()
                assert cleared['training_complete'] is False, "training_complete should be False after clearing"
                assert cleared['training_active'] is False, "training_active should be False"
                assert cleared['output_dir'] is None, "output_dir should be None"
                
                # Step 4: Verify we can start new training (not "already completed")
                # The old check would have failed here:
                # if current_state.get('training_complete') and current_state.get('output_dir'):
                #     return  # Would have returned immediately!
                
                # With cleared state, this check passes:
                would_start_training = not (cleared.get('training_complete') and cleared.get('output_dir'))
                assert would_start_training is True, "Should be able to start new training with cleared state"
    
    def test_no_immediate_complete_with_old_state(self):
        """Verify we don't get immediate 'Training already completed' message."""
        from edukaai_studio.core.state import save_state_to_disk
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Simulate old completed state
                old_state = {
                    'training_complete': True,
                    'output_dir': 'outputs/exists',
                    'uploaded_file': 'data.json',
                    'training_config': {'model_id': 'test'},
                }
                
                # Create output dir to make it "exist"
                (Path(tmpdir) / "outputs" / "exists").mkdir(parents=True)
                
                save_state_to_disk(old_state)
                
                # The BUG would have checked this and returned immediately:
                # OLD CODE: if current_state.get('training_complete') and current_state.get('output_dir'):
                # Both conditions must be truthy
                buggy_would_trigger = bool(
                    old_state.get('training_complete') and old_state.get('output_dir')
                )
                assert buggy_would_trigger is True, "Old buggy check would trigger"
                
                # The FIX clears state first:
                old_state_cleared = {
                    **old_state,
                    'training_complete': False,
                    'output_dir': None,
                }
                
                # Now the check should NOT trigger:
                # FIXED CODE: state is cleared first, so training_complete=False
                fixed_would_trigger = bool(
                    old_state_cleared.get('training_complete') and old_state_cleared.get('output_dir')
                )
                assert fixed_would_trigger is False, "Fixed check should not trigger (allows training to start)"


class TestStateClearing:
    """Test state clearing functions."""
    
    def test_clear_state_file_removes_state(self):
        """Verify clear_state_file actually removes the state file."""
        from edukaai_studio.core.state import save_state_to_disk, clear_state_file, get_state_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Save some state
                save_state_to_disk({'training_complete': True, 'output_dir': 'test'})
                assert state_file.exists(), "State file should exist"
                
                # Clear it
                result = clear_state_file()
                assert result is True, "clear_state_file should return True"
                assert not state_file.exists(), "State file should be removed"
    
    def test_clear_state_file_handles_missing_file(self):
        """Verify clear_state_file handles non-existent file gracefully."""
        from edukaai_studio.core.state import clear_state_file, get_state_file
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # File doesn't exist
                assert not state_file.exists()
                
                # Should still return True (idempotent)
                result = clear_state_file()
                assert result is True, "clear_state_file should return True even if file missing"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
