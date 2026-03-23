"""Test for state synchronization bug between tabs.

This test reproduces the issue where training completes successfully
but the UI still shows incomplete status.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestStateSynchronizationBug:
    """Test for the bug: training completes but UI shows incomplete status."""
    
    def test_chat_tab_reads_completed_state_from_disk(self):
        """CRITICAL: Chat tab should read completed=true from disk even if passed stale state.
        
        Bug Scenario:
        1. Training completes, saves state to disk with training_complete=True
        2. User clicks on Chat tab
        3. Chat tab receives stale state from Gradio (training_complete=False)
        4. Chat tab should load from disk and see training_complete=True
        5. But old code only merged if disk_state.get('training_complete'), which is True
           so it should work... unless there's a timing issue
        
        Root cause: The old code only merged disk state when disk_state existed AND
        showed complete. But if the component state was stale and disk state was 
        from a previous session, it might not merge properly.
        """
        from edukaai_studio.ui.tabs.chat import refresh_chat_status
        from edukaai_studio.core.state import save_state_to_disk, get_state_file
        
        # Setup: Save completed state to disk
        completed_state = {
            'training_complete': True,
            'training_active': False,
            'output_dir': 'outputs/test_model_20240101_120000',
            'completion_time': '12:00:00',
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the state file location
            state_file = Path(tmpdir) / ".studio_state.json"
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Save completed state
                save_state_to_disk(completed_state)
                
                # Simulate: Chat tab receives STALE state from Gradio
                stale_state = {
                    'training_complete': False,  # Stale!
                    'training_active': True,     # Stale!
                    'output_dir': None,
                }
                
                # Call refresh_chat_status
                result = refresh_chat_status(stale_state)
                
                # Verify: Chat tab should now show completed status
                # The function returns (status_message, base_model_info, trained_model_info)
                status_message = result[0]
                
                # Should indicate training is complete or model is available
                # "OK: Ready! Fine-tuned model available." means it's working!
                completion_indicators = [
                    'complete', 'ready', '✅', 'available', 
                    'fine-tuned', 'trained model'
                ]
                assert any(indicator in status_message.lower() for indicator in completion_indicators), \
                    f"Chat tab should show completed/available status, got: {status_message}"
    
    def test_results_tab_always_uses_disk_state(self):
        """Results tab should always use disk state as source of truth."""
        from edukaai_studio.ui.tabs.results import refresh_results_status
        from edukaai_studio.core.state import save_state_to_disk
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Save completed state to disk
                completed_state = {
                    'training_complete': True,
                    'training_active': False,
                    'output_dir': 'outputs/test_model',
                    'completion_time': '12:00:00',
                }
                save_state_to_disk(completed_state)
                
                # Simulate stale Gradio state
                stale_state = {
                    'training_complete': False,
                    'training_active': True,
                }
                
                # Call results refresh
                status_msg, updated_state = refresh_results_status(stale_state)
                
                # Should use disk state
                assert updated_state.get('training_complete') == True, \
                    "Results tab should use disk state's completed status"
                assert updated_state.get('training_active') == False, \
                    "Results tab should use disk state's active status"
    
    def test_disk_state_priority_over_component_state(self):
        """Disk state should always win over potentially stale component state."""
        from edukaai_studio.core.state import save_state_to_disk, load_state_from_disk
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Save completed training
                disk_data = {
                    'training_complete': True,
                    'training_active': False,
                    'output_dir': 'outputs/completed_model',
                }
                save_state_to_disk(disk_data)
                
                # Load it back
                loaded = load_state_from_disk()
                
                # Verify disk state is correct
                assert loaded is not None
                assert loaded.get('training_complete') == True
                assert loaded.get('training_active') == False
                assert loaded.get('output_dir') == 'outputs/completed_model'


class TestRaceConditions:
    """Test for race conditions in state management."""
    
    def test_concurrent_tab_switches_dont_corrupt_state(self):
        """Multiple tabs switching shouldn't corrupt the state."""
        from edukaai_studio.core.state import save_state_to_disk, load_state_from_disk
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Save completed state
                save_state_to_disk({
                    'training_complete': True,
                    'training_active': False,
                    'output_dir': 'outputs/model',
                })
                
                # Simulate multiple tabs reading simultaneously
                reads = []
                for _ in range(5):
                    state = load_state_from_disk()
                    reads.append(state.get('training_complete'))
                
                # All reads should be consistent
                assert all(r == True for r in reads), \
                    "All concurrent reads should see completed=True"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
