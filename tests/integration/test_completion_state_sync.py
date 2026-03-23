"""Test for training completion state synchronization.

This test verifies that when training completes, the state is properly
saved to disk and can be read by other tabs.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import queue


class TestTrainingCompletionStateSync:
    """Test state synchronization after training completion."""
    
    def test_train_tab_saves_completed_state_to_disk(self):
        """Verify train.py saves correct state to disk when training completes."""
        from edukaai_studio.core.state import save_state_to_disk, load_state_from_disk
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Simulate completed training state
                completed_state = {
                    'training_complete': True,
                    'training_active': False,
                    'output_dir': 'outputs/test_model_20240101_120000',
                    'completion_time': '12:00:00',
                    'train_losses': {200: 1.805},
                    'val_losses': {200: 1.805},
                }
                
                # Save state
                result = save_state_to_disk(completed_state)
                assert result is True, "State save should succeed"
                
                # Load and verify
                loaded = load_state_from_disk()
                assert loaded is not None, "Should load state from disk"
                assert loaded.get('training_complete') is True, "training_complete should be True"
                assert loaded.get('training_active') is False, "training_active should be False"
                assert loaded.get('output_dir') == 'outputs/test_model_20240101_120000'
    
    def test_other_tabs_read_completed_state_from_disk(self):
        """Verify other tabs read completed state from disk even with stale component state."""
        from edukaai_studio.ui.tabs.chat import refresh_chat_status
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
                
                # Simulate stale Gradio state (what Gradio passes to the function)
                stale_gradio_state = {
                    'training_complete': False,  # Wrong!
                    'training_active': True,      # Wrong!
                    'output_dir': None,
                }
                
                # Call refresh function
                result = refresh_chat_status(stale_gradio_state)
                
                # Verify the function returned status indicating completion
                status_message = result[0]
                completion_indicators = [
                    'complete', 'ready', '✅', 'available', 
                    'fine-tuned', 'trained model'
                ]
                assert any(indicator in status_message.lower() for indicator in completion_indicators), \
                    f"Should show completed status, got: {status_message}"
    
    def test_results_tab_uses_disk_state(self):
        """Verify results tab uses disk state over stale component state."""
        from edukaai_studio.ui.tabs.results import refresh_results_status
        from edukaai_studio.core.state import save_state_to_disk
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Save completed state
                completed_state = {
                    'training_complete': True,
                    'training_active': False,
                    'output_dir': 'outputs/test_model',
                }
                save_state_to_disk(completed_state)
                
                # Simulate stale state
                stale_state = {
                    'training_complete': False,
                    'training_active': True,
                }
                
                # Call results refresh
                status_msg, updated_state = refresh_results_status(stale_state)
                
                # Verify state was updated from disk
                assert updated_state.get('training_complete') is True, \
                    "Should use disk state's training_complete=True"
                assert updated_state.get('training_active') is False, \
                    "Should use disk state's training_active=False"
    
    def test_disk_state_priority_invariant(self):
        """Invariant: Disk state should always take priority over component state."""
        from edukaai_studio.core.state import save_state_to_disk, load_state_from_disk
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / ".studio_state.json"
            
            with patch('edukaai_studio.core.state.get_state_file', return_value=state_file):
                # Save disk state as complete
                save_state_to_disk({
                    'training_complete': True,
                    'training_active': False,
                    'output_dir': 'outputs/completed',
                })
                
                # Load it
                disk = load_state_from_disk()
                
                # Invariant: Disk values should be authoritative
                assert disk['training_complete'] is True
                assert disk['training_active'] is False
                
                # Component state (simulated stale)
                component = {
                    'training_complete': False,
                    'training_active': True,
                    'output_dir': None,
                }
                
                # Merge logic (what tabs should do)
                merged = {
                    **component,
                    'training_complete': disk.get('training_complete', component['training_complete']),
                    'training_active': disk.get('training_active', component['training_active']),
                    'output_dir': disk.get('output_dir', component['output_dir']),
                }
                
                # Invariant: Disk wins
                assert merged['training_complete'] is True  # From disk
                assert merged['training_active'] is False  # From disk
                assert merged['output_dir'] == 'outputs/completed'  # From disk


class TestTrainingMonitorCompletion:
    """Test that TrainingMonitor properly reports completion."""
    
    def test_monitor_is_complete_returns_true_when_done(self):
        """Verify monitor.is_complete() returns True when training finishes."""
        from edukaai_studio.ui.training_monitor import TrainingMonitor
        
        # Create mock queues
        output_q = queue.Queue()
        progress_q = queue.Queue()
        
        monitor = TrainingMonitor(output_q, progress_q)
        
        # Initially not complete
        assert monitor.is_complete() is False
        
        # Simulate completion
        monitor.training_complete = True
        monitor._running = False
        
        # Now should be complete
        assert monitor.is_complete() is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
